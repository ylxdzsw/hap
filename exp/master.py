import config
import torch
import torch.fx
import math
import hetspmd

from utils import *

import collectives # required in codegen
import operator # required in codegen

model = symbolic_trace(config.get_model(seed=39))
print(model.code, flush=True)

for i, node in enumerate(model.graph.nodes):
    node.meta['id'] = i

dgraph = hetspmd.main(model, {
    "input_shape": config.input_shape()
})

print(dgraph, flush=True)

dmodel = torch.fx.GraphModule(model, dgraph)

print(dmodel.code, flush=True)
