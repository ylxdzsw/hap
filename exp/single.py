import config
import sys
import torch
import torch.fx
from torch.profiler import profile, record_function, ProfilerActivity

from models import MLP, MLP2, MoE, Transformer
from annotator import annotate
from compiler import compile
from utils import *

def run(rank, model_str):
    import torch.distributed as dist
    dist.init_process_group('nccl', rank=rank)

    torch.manual_seed(0)
    # torch.use_deterministic_algorithms(True)

    if model_str == 'mlp':
        model = MLP(nhid=config.emsize, nlayers=config.nlayers)
    if model_str == 'mlp2':
        model = MLP2(nhid=config.emsize, nlayers=config.nlayers)
    if model_str == 'moe':
        model = MoE(emsize=config.emsize, nhead=4, nhid=config.nhid, dropout=config.dropout, n_expert=config.n_expert, capacity=config.capacity, nlayers=config.nlayers)
    if model_str == 'transformer':
        model = Transformer(emsize=config.emsize, nhead=4, nhid=config.nhid, dropout=config.dropout, nlayers=config.nlayers)

    model = symbolic_trace(model).cuda(rank)
    annotate(model, { 'x': (config.batch_size, config.seqlen, config.emsize) })
    compile(model, load(f"strategy_{model_str}"), rank, config.world_size)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
    test_input = torch.rand(config.batch_size, config.seqlen, config.emsize).cuda(rank) / 6

    for iter in range(10):
        with measure_time(f"iteration {iter}") as wall_time:
            loss = model(test_input)
            aggregated_loss = loss.detach().clone()
            dist.reduce(aggregated_loss, 0)
            if rank == 0:
                print(f"loss {iter}:", aggregated_loss.cpu().numpy())
            # dist.barrier(device_ids=[rank])

            loss.backward()
            # torch.cuda.synchronize()
            optimizer.step()
            dist.barrier()
        if rank == 0:
            print(wall_time)

    with profile(
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        # record_shapes = True,
        # profile_memory = True,
        schedule = torch.profiler.schedule(wait=1, warmup=10, active=4)
    ) as prof:
        for _ in range(15):
            with record_function("forward"):
                loss = model(test_input)
            with record_function("backward"):
                loss.backward()
                torch.cuda.synchronize()
            with record_function("update"):
                optimizer.step()
            dist.barrier()
            prof.step()

    if rank == 0:
        # print(prof.key_averages().table(sort_by="cuda_time_total"))
        prof.export_chrome_trace("trace.json")

if __name__ == '__main__':
    if torch.cuda.device_count() != config.world_size:
        print("forget to set CUDA_VISIBLE_DEVICES")
        raise SystemExit

    import os
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '39281'
    os.environ['WORLD_SIZE'] = str(config.world_size)

    import torch.multiprocessing as mp
    mp.set_start_method('spawn')

    for rank in range(config.world_size):
        mp.Process(target=run, args=(rank, sys.argv[1])).start()

    for p in mp.active_children():
        p.join()
