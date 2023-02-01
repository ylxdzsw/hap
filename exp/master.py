import config
import torch
import torch.fx
import hetspmd

from utils import *

model = symbolic_trace(config.get_model(seed=39))#.cuda(0)

print(model)

print(hetspmd.main(list(model.graph.nodes)))

