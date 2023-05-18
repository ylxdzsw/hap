import config
import sys
import datetime
import time
import torch
import torch.fx
from torch.profiler import profile, record_function, ProfilerActivity
from contextlib import nullcontext
import numpy as np
import hetspmd

from utils import *

def run(global_rank, local_rank):
    import torch.distributed as dist
    dist.init_process_group('nccl', rank=global_rank, timeout=datetime.timedelta(hours=2))

    model = symbolic_trace(config.get_model(seed=39))

    for i, node in enumerate(model.graph.nodes):
        node.meta['id'] = i

    hetspmd.init()

    dgraph = hetspmd.main(model, {
        "input_shape": config.input_shape(),
        "device_flops": [
            config.profiler_data["device_flops"],
        ] * config.world_size,
        "all_reduce_bandwidth": config.profiler_data["all_reduce"],
        "all_gather_bandwidth": config.profiler_data["all_gather"],
        "reduce_scatter_bandwidth": config.profiler_data["reduce_scatter"],
        "all_to_all_bandwidth": config.profiler_data["all_to_all"],
        "rank": global_rank
    })

    # print(dgraph, flush=True)

    dmodel = torch.fx.GraphModule(model, dgraph).cuda(local_rank)
    del model

    optimizer = torch.optim.Adam(dmodel.parameters(), lr=config.lr)
    train_data = config.get_data()[1]

    result_times = []
    strat_time = last_iter_time = time.time()
    total_loss = 0
    for iter in range(config.run_iter):
        optimizer.zero_grad()
        x, y = next(train_data)
        x = x.cuda(local_rank)
        y = y.cuda(local_rank)

        with torch.autocast(device_type="cuda") if config.fp16 else nullcontext() :
            loss = dmodel(x, y)

        aggregated_loss = loss.detach().clone()
        dist.reduce(aggregated_loss, 0)

        if global_rank == 0:
            total_loss += aggregated_loss.cpu().numpy() / config.batch_size / config.seqlen
            if iter % config.log_iter == 0:
                print(f"loss (log ppl) {iter}: {total_loss / config.log_iter:.3f}, wall clock: {time.time() - strat_time:.3f}")
                total_loss = 0
        # dist.barrier(device_ids=[global_rank])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(dmodel.parameters(), 0.5)
        # torch.cuda.synchronize()
        optimizer.step()
        # dist.barrier()
        if config.report_per_iter_time and local_rank == 0:
            iter_duration = time.time() - last_iter_time
            result_times.append(iter_duration)
            last_iter_time += iter_duration
            print("iter time: ", iter_duration)
            print("avg±std:", np.mean(result_times[-config.avg_iter:]), np.std(result_times[-config.avg_iter:]))

    # for epoch in range(config.epoch):
    #     total_loss = 0.
    #     start_time = time.time()
    #     for batch, offset in enumerate(range(0, train_data.size(1) - config.seqlen, config.seqlen)):
    #         loss = model(
    #             x = train_data[:, offset:offset+config.seqlen],
    #             y = train_data[:, offset+1:offset+1+config.seqlen]
    #         ) / config.batch_size / config.seqlen

    #         total_loss += loss.detach()
    #         if batch % config.log_iterval == 0 and batch > 0:
    #             dist.reduce(total_loss, 0)
    #             if global_rank == 0:
    #                 avg_loss = total_loss / config.log_iterval
    #                 elapsed = time.time() - start_time
    #                 print(f"epoch {epoch:3d} | batch {batch:3d} | ppl {math.exp(avg_loss):02.2f} | ms/batch {elapsed*1000/config.log_iterval:5.2f}")
    #             total_loss = 0.
    #             start_time = time.time()

    #         torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)

    #         loss.backward()
    #         optimizer.step()

    if not config.trace:
        return

    x, y = next(train_data)
    x = x.cuda(local_rank)
    y = y.cuda(local_rank)
    with profile(
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        # record_shapes = True,
        # profile_memory = True,
        schedule = torch.profiler.schedule(wait=1, warmup=10, active=4)
    ) as prof:
        for _ in range(15):
            with record_function("forward"):
                with torch.autocast(device_type="cuda") if config.fp16 else nullcontext() :
                    loss = dmodel(x, y)
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

    # if torch.cuda.device_count() != len(ranks):
    #     print("forget to set CUDA_VISIBLE_DEVICES")
    #     raise SystemExit

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
