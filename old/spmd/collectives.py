# this is basically the same as torch.distributed.nn.functional, with the following differences:
# 1. we add .contiguous() calls that is required by NCCL,
# 2. we insert chunk/cat calls and unify the API, such that we can generate calls to them automatically,
# 3. we ignore other backends than NCCL, remove the support for higher order derivatives, and fill-in default arguments like reduce op and collective group.
# Note: NCCL operators are in-place, we must clone the input tensors. Some tensors in the backward pass may be modified in-place but better be defensive here.

from __future__ import annotations

import torch
import torch.distributed as dist

from utils import *

def _sync(op, stream):
    class Func(torch.autograd.Function):
        @staticmethod
        def forward(ctx, tensor, *other_args):
            torch.cuda.current_stream().record_event().wait(stream)
            with torch.cuda.stream(stream):
                result = op.forward(ctx, tensor, *other_args)
            if isinstance(tensor, torch.Tensor):
                tensor.record_stream(stream)
            stream.record_event().wait()
            return result

        @staticmethod
        def backward(ctx, grad_output):
            torch.cuda.current_stream().record_event().wait(stream)
            with torch.cuda.stream(stream):
                result = op.backward(ctx, grad_output)
            if isinstance(grad_output, torch.Tensor):
                grad_output.record_stream(stream)
            stream.record_event().wait()
            return result

    return Func

class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, dim):
        ctx.dim = dim
        out_tensor_slices = [ torch.empty_like(tensor) for _ in range(dist.get_world_size()) ]
        dist.all_gather(out_tensor_slices, tensor)
        return torch.cat(out_tensor_slices, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        # removing this `.contiguous()` call leads to silent wrong result!
        grad_output_slices = torch.chunk(grad_output.contiguous(), dist.get_world_size(), dim=ctx.dim)
        grad = torch.empty_like(grad_output_slices[0])
        dist.reduce_scatter(grad, [ x.contiguous() for x in grad_output_slices ])
        return grad, None

# aliasing will prevent assigning __module__, which is required by fx.node.Node.__repr__, otherwise it crashes
def all_gather(tensor, dim): return AllGather.apply(tensor, dim)
def all_gather_sync(stream, tensor, dim): return _sync(AllGather, stream).apply(tensor, dim)

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
def all_reduce_sync(stream, tensor): return _sync(AllReduce, stream).apply(tensor)

class ReduceScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, dim):
        ctx.dim = dim
        tensor_slices = torch.chunk(tensor, dist.get_world_size(), dim=dim)
        out = torch.empty_like(tensor_slices[0])
        dist.reduce_scatter(out, [ x.contiguous() for x in tensor_slices ])
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous() # similar to AllGather, this call is important. Both occurences of grad_output require contiguous.
        grad_output_slices = [ torch.empty_like(grad_output) for _ in range(dist.get_world_size()) ]
        dist.all_gather(grad_output_slices, grad_output)
        return torch.cat(grad_output_slices, dim=ctx.dim), None

def reduce_scatter(tensor, dim): return ReduceScatter.apply(tensor, dim)
def reduce_scatter_sync(stream, tensor, dim): return _sync(ReduceScatter, stream).apply(tensor, dim)

# Not really a collective operator
def dynamic_slice(tensor, dim):
    tensor_slices = torch.chunk(tensor, dist.get_world_size(), dim=dim)
    return tensor_slices[dist.get_rank()].contiguous()

# Actually there is an "all_to_all_single" that do the chunking and cating for us: https://github.com/pytorch/pytorch/blob/master/torch/distributed/distributed_c10d.py#L2404
# similar versions exist for other collectives. It should be perferable in terms of performance (and deepspeed uses them)
class AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, split_dim, cat_dim):
        ctx.split_dim, ctx.cat_dim = split_dim, cat_dim
        tensor_slices = torch.chunk(tensor.contiguous(), dist.get_world_size(), dim=split_dim)
        out_slices = [ torch.empty_like(tensor_slices[0]) for _ in range(dist.get_world_size()) ]
        dist.all_to_all(out_slices, [ x.contiguous() for x in tensor_slices ])
        return torch.cat(out_slices, dim=cat_dim)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output_slices = torch.chunk(grad_output.contiguous(), dist.get_world_size(), dim=ctx.cat_dim)
        grad_slices = [ torch.empty_like(grad_output_slices[0]) for _ in range(dist.get_world_size()) ]
        dist.all_to_all(grad_slices, [ x.contiguous() for x in grad_output_slices ])
        return torch.cat(grad_slices, dim=ctx.split_dim), None, None

def all_to_all(tensor, split_dim, cat_dim): return AllToAll.apply(tensor, split_dim, cat_dim)
def all_to_all_sync(stream, tensor, split_dim, cat_dim): return _sync(AllToAll, stream).apply(tensor, split_dim, cat_dim)

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
def replicate_sync(stream, tensor): return _sync(Replicate, stream).apply(tensor)
