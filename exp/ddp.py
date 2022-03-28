import config
import sys
import datetime
import torch
import torch.fx
from torch.profiler import profile, record_function, ProfilerActivity

from torch.nn.parallel import DistributedDataParallel as DDP

from utils import *

def run(global_rank, local_rank):
    import torch.distributed as dist
    dist.init_process_group('nccl', rank=global_rank, timeout=datetime.timedelta(hours=2))

    model = symbolic_trace(config.get_model(seed=39)).cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    train_data = config.get_data()[1]

    result_times = []
    for iter in range(100):
        with measure_time(f"iteration {iter}") as wall_time:
            x, y = next(train_data)
            x = x.chunk(config.world_size, 0)[global_rank].cuda(local_rank)
            y = y.chunk(config.world_size, 0)[global_rank].cuda(local_rank)
            loss = model(x, y)
            aggregated_loss = loss.detach().clone() * config.world_size # not sure why we need this but the loss seems to be smaller than expected?
            dist.reduce(aggregated_loss, 0)
            if global_rank == 0:
                print(f"loss {iter}:", aggregated_loss.cpu().numpy())
            # dist.barrier(device_ids=[global_rank])

            loss.backward()
            # torch.cuda.synchronize()
            optimizer.step()
            # dist.barrier()
        if local_rank == 0:
            print(wall_time)
            result_times.append(wall_time.time)
            print("avg:", sum(result_times[-50:]) / len(result_times[-50:]))

    if not config.trace:
        return

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

    if local_rank == 0:
        # print(prof.key_averages().table(sort_by="cuda_time_total"))
        prof.export_chrome_trace("trace.json")

if __name__ == '__main__':
    ranks = [ int(x) for x in sys.argv[1].split(',') ]

    if torch.cuda.device_count() != len(ranks):
        print("forget to set CUDA_VISIBLE_DEVICES")
        raise SystemExit

    import os
    os.environ['MASTER_ADDR'] = str(config.master_addr)
    os.environ['MASTER_PORT'] = str(config.master_port)
    os.environ['WORLD_SIZE'] = str(config.world_size)

    import torch.multiprocessing as mp
    mp.set_start_method('spawn')

    for local_rank, global_rank in enumerate(ranks):
        mp.Process(target=run, args=(global_rank, local_rank)).start()

    for p in mp.active_children():
        p.join()
