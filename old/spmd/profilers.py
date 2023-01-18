from __future__ import annotations

import math
import torch
import torch.fx
import collectives
from utils import *

class FlopsProfiler:
    def __init__(self, annotated_model: torch.fx.GraphModule, input_data) -> None:
        optimizer = torch.optim.SGD(annotated_model.parameters(), lr=1e-8)

        for _ in range(11):
            loss = annotated_model(input_data)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()

        with measure_time() as wall_time:
            loss = annotated_model(input_data)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
        total_flops = 0
        for node in annotated_model.graph.nodes:
            total_flops += node.meta.get('flops', 0)
        print(f"Profiling finished. Total flops: {total_flops}, wall time: {wall_time.time}")
        self.device_flops = math.floor(total_flops / wall_time.time)
        print(self.device_flops)

class BandwidthProfiler:
    def __init__(self, config, ranks) -> None:
        self.bandwidth = {}

        for op, op_args in (
            (collectives.all_gather, (0,)),
            (collectives.all_reduce, ()),
            (collectives.reduce_scatter, (0,)),
            (collectives.all_to_all, (0, 1)),
        ):
            estimation = []
            for size in (4*1024*1024, 16*1024*1024, 64*1024*1024, 256*1024*1024):
                ts = [ self.run_collective(config, ranks, op, size, op_args) for _ in range(5) ]
                print((size, sorted(ts)), flush=True)
                estimation.append(size / sorted(ts)[2])
            self.bandwidth[op.__name__] = math.floor(sum(estimation) / len(estimation))
        print(self.bandwidth)

    def run_collective(self, config, ranks, op, size: int, op_args: tuple) -> float:
        import os
        os.environ['MASTER_ADDR'] = str(config.master_addr)
        os.environ['MASTER_PORT'] = str(config.master_port)
        os.environ['WORLD_SIZE'] = str(config.world_size)

        import torch.multiprocessing as mp
        ctx = mp.get_context('spawn')
        queue = ctx.Queue(1)

        if op in (collectives.all_gather, collectives.all_to_all):
            size //= config.world_size # the size argument is the size of the original tensor, we convert it to the size of input in experiment

        for local_rank, global_rank in enumerate(ranks):
            ctx.Process(target=_run_collective_worker, args=(op, size, queue, op_args, global_rank, local_rank)).start()

        for p in mp.active_children():
            p.join()

        return queue.get()

def _run_collective_worker(op, input_slice_size: int, queue, op_args: tuple, global_rank: int, local_rank: int):
    import torch.distributed as dist
    dist.init_process_group('nccl', rank=global_rank)

    tensor = torch.rand(256, input_slice_size // 1024).to(local_rank)
    for _ in range(10):
        with measure_time() as time:
            op(tensor, *op_args)
            torch.cuda.synchronize(local_rank)

    if local_rank == 0:
        queue.put(time.time)
