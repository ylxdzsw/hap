import config

import torch
from sys import argv
from models import MLP, MLP2, MoE, Transformer
from annotator import annotate
from profilers import FlopsProfiler, BandwidthProfiler
from utils import *

if __name__ == '__main__':
    if argv[1] == "mlp":
        model = symbolic_trace(MLP(nhid=config.emsize, nlayers=config.nlayers)).cuda()
        data = torch.rand(config.batch_size // 2, config.seqlen, config.emsize).cuda() / 6 # the batch size is halved for duplication
        annotate(model, { 'x': tuple(data.shape) })
        profiler = FlopsProfiler(model, data)
        save("flops_profiler", profiler)
    elif argv[1] == "comm":
        profiler = BandwidthProfiler()
        save("bandwidth_profiler", profiler)

