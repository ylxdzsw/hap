import config

import torch
from sys import argv
from annotator import annotate
from profilers import FlopsProfiler, BandwidthProfiler
from utils import *

if __name__ == '__main__':
    if len(argv) >= 2:
        ranks = [ int(x) for x in argv[1].split(',') ]

        if torch.cuda.device_count() != len(ranks):
            print("forget to set CUDA_VISIBLE_DEVICES")
            raise SystemExit

        profiler = BandwidthProfiler(config, ranks)
        # save("bandwidth_profiler", profiler)
        raise SystemExit

    model = symbolic_trace(config.get_model()).cuda(0)
    data = torch.rand(config.batch_size // config.world_size // 2, config.seqlen, config.emsize).cuda() / 6 # the batch size is halved for duplication
    annotate(model, { 'x': tuple(data.shape) })
    profiler = FlopsProfiler(model, data)
    # save("flops_profiler", profiler)
