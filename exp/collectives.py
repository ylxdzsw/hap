from __future__ import annotations

import torch
import torch.distributed as dist

def sharded_shape(shape, dim, length):
    return shape[:dim] + (length,) + shape[dim+1:]

class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, dim, sharding_lengths, rank):
        ctx.dim, ctx.sharding_lengths, ctx.rank = dim, sharding_lengths, rank
        out_tensor_slices = [ torch.empty(*sharded_shape(tensor.shape, dim, length)) for length in sharding_lengths ]
        dist.all_gather(out_tensor_slices, tensor)
        return torch.cat(out_tensor_slices, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        # removing this `.contiguous()` call leads to silent wrong result!
        grad_output_slices = torch.split(grad_output.contiguous(), ctx.sharding_lengths, dim=ctx.dim)
        grad = torch.empty_like(grad_output_slices[ctx.rank])
        dist.reduce_scatter(grad, [ x.contiguous() for x in grad_output_slices ])
        return grad, None

# aliasing will prevent assigning __module__, which is required by fx.node.Node.__repr__, otherwise it crashes
def all_gather(tensor, dim, sharding_lengths, rank): return AllGather.apply(tensor, tensor, dim, sharding_lengths, rank)

class AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        # hack
        if isinstance(tensor, int):
            ctx.hack = True
            return tensor * dist.get_world_size()
        out_tensor = tensor.contiguous()
        dist.all_reduce(out_tensor)
        return out_tensor

    @staticmethod
    def backward(ctx, grad_output):
        if getattr(ctx, 'hack', False):
            return None, None
        grad = grad_output.clone()
        dist.all_reduce(grad)
        return grad

def all_reduce(tensor): return AllReduce.apply(tensor)

class ReduceScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, dim, sharding_lengths, rank):
        ctx.dim, ctx.sharding_lengths, ctx.rank = dim, sharding_lengths, rank
        tensor_slices = torch.split(tensor, sharding_lengths, dim=dim)
        out = torch.empty_like(tensor_slices[rank])
        dist.reduce_scatter(out, [ x.contiguous() for x in tensor_slices ])
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous() # similar to AllGather, this call is important. Both occurences of grad_output require contiguous.
        grad_output_slices = [ torch.empty(sharded_shape(grad_output.shape, ctx.dim, length)) for length in ctx.sharding_lengths ]
        dist.all_gather(grad_output_slices, grad_output)
        return torch.cat(grad_output_slices, dim=ctx.dim), None

def reduce_scatter(tensor, dim, sharding_lengths, rank): return ReduceScatter.apply(tensor, dim, sharding_lengths, rank)

# Not really a collective operator
def dynamic_slice(tensor, dim, sharding_lengths, rank):
    tensor_slices = torch.split(tensor, sharding_lengths, dim=dim)
    return tensor_slices[rank].contiguous()

# Actually there is an "all_to_all_single" that do the chunking and cating for us: https://github.com/pytorch/pytorch/blob/master/torch/distributed/distributed_c10d.py#L2404
# similar versions exist for other collectives. It should be perferable in terms of performance (and deepspeed uses them)
class AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, split_dim, cat_dim, split_sharding_lengths, cat_sharding_lengths, rank):
        ctx.split_dim, ctx.cat_dim, ctx.split_sharding_lengths, ctx.cat_sharding_lengths, ctx.rank = split_dim, cat_dim, split_sharding_lengths, cat_sharding_lengths, rank
        tensor_slices = torch.split(tensor.contiguous(), split_sharding_lengths, dim=split_dim)
        out_slices = [ torch.empty(sharded_shape(tensor_slices[rank].shape, cat_dim, length)) for length in cat_sharding_lengths ]
        dist.all_to_all(out_slices, [ x.contiguous() for x in tensor_slices ])
        return torch.cat(out_slices, dim=cat_dim)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output_slices = torch.split(grad_output.contiguous(), ctx.cat_sharding_lengths, dim=ctx.cat_dim)
        grad_slices = [ torch.empty_like(sharded_shape(grad_output_slices[ctx.rank].shape, ctx.split_dim, length)) for length in ctx.split_sharding_lengths ]
        dist.all_to_all(grad_slices, [ x.contiguous() for x in grad_output_slices ])
        return torch.cat(grad_slices, dim=ctx.split_dim), None, None

def all_to_all(tensor, split_dim, cat_dim, split_sharding_lengths, cat_sharding_lengths, rank): return AllToAll.apply(tensor, split_dim, cat_dim, split_sharding_lengths, cat_sharding_lengths, rank)

# the "f" function in the Megatron paper, which is identical function in forward, but all-reduce in backward. It is used for tensors that are replicated before entering the funtion (input and parameters)
class Replicate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad = grad_output.clone()
        dist.all_reduce(grad)
        return grad

def replicate(tensor): return Replicate.apply(tensor)
