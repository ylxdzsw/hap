import config
import torch
import torch.fx
import math
import hetspmd

from utils import *

model = symbolic_trace(config.get_model(seed=39))
print(model.code, flush=True)

for i, node in enumerate(model.graph.nodes):
    node.meta['id'] = i

hetspmd.init()

dgraph = hetspmd.main(model, {
    "input_shape": config.input_shape(),
    "device_flops": [4139214925014.] * 4,
    "all_reduce_bandwidth": 611692856.,
    "all_gather_bandwidth": 1224592728.,
    "reduce_scatter_bandwidth": 1130230706.,
    "all_to_all_bandwidth": 10701240728.,
    "rank": 0
})

# print(dgraph, flush=True)

dmodel = torch.fx.GraphModule(model, dgraph)
