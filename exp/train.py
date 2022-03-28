import config
import sys
import datetime
import time
import math
import torch
import torch.fx
from torch.profiler import profile, record_function, ProfilerActivity

from annotator import annotate
from compiler import compile
from utils import *

def run(global_rank, local_rank):
    import torch.distributed as dist
    dist.init_process_group('nccl', rank=global_rank, timeout=datetime.timedelta(hours=2))

    # ntokens, train_data, test_data, valid_data = config.get_data()
    # train_data = train_data.cuda(local_rank)

    model = symbolic_trace(config.get_model(seed=39)).cuda(local_rank)
    annotate(model, config.input_shape())
    compile(model, load(f"strategy_{config.model_name}"), global_rank=global_rank, local_rank=local_rank, world_size=config.world_size)

    # optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

    result_times = []
    test_input = torch.rand(config.batch_size, config.seqlen, config.emsize).cuda(local_rank) / 6
    for iter in range(100):
        with measure_time(f"iteration {iter}") as wall_time:
            loss = model(test_input)
            aggregated_loss = loss.detach().clone()
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
