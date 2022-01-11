import config

import torch
from sys import argv
from models import MLP, MLP2, MoE, Transformer
from annotator import annotate
from profilers import FlopsProfiler, BandwidthProfiler
from utils import *

if __name__ == '__main__':
    if argv[1] == "comm":
        profiler = BandwidthProfiler()
        save("bandwidth_profiler", profiler)
        raise SystemExit

    if argv[1] == 'mlp':
        model = MLP(nhid=config.emsize, nlayers=config.nlayers)
    if argv[1] == 'mlp2':
        model = MLP2(nhid=config.emsize, nlayers=config.nlayers)
    if argv[1] == 'moe':
        model = MoE(emsize=config.emsize, nhead=config.nheads, nhid=config.nhid, dropout=config.dropout, n_expert=config.n_expert, capacity=config.capacity, nlayers=config.nlayers)
    if argv[1] == 'transformer':
        model = Transformer(emsize=config.emsize, nhead=config.nheads, nhid=config.nhid, dropout=config.dropout, nlayers=config.nlayers)

    model = symbolic_trace(model).cuda(0)
    data = torch.rand(config.batch_size // config.world_size // 2, config.seqlen, config.emsize).cuda() / 6 # the batch size is halved for duplication
    annotate(model, { 'x': tuple(data.shape) })
    profiler = FlopsProfiler(model, data)
    save("flops_profiler", profiler)
