import config
import torch
import torch.fx
import math
import hetspmd

from utils import *

import collectives

model = symbolic_trace(config.get_model(seed=39))
print(model.code, flush=True)

for i, node in enumerate(model.graph.nodes):
    node.meta['id'] = i

print(hetspmd.main(model, {
    "input_shape": config.input_shape()
}))


