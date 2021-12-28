import sys
sys.path.insert(1, f"{sys.path[0]}/../spmd")

import torch
from sys import argv
from models import MLP, MLP2, MoE, Transformer
from annotator import annotate
from profilers import FlopsProfiler, BandwidthProfiler
from utils import *

if __name__ == '__main__':
    if argv[1] == "mlp":
        model = symbolic_trace(MLP(nhid=1024, nlayers=4)).cuda()
        data = torch.rand(64, 256, 1024).cuda() / 6
        annotate(model, { 'x': tuple(data.shape) })
        profiler = FlopsProfiler(model, data)
        save("flops_profiler", profiler)
    elif argv[1] == "comm":
        profiler = BandwidthProfiler()
        save("bandwidth_profiler", profiler)

