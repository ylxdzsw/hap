import config
import torch
import torch.fx
import math
import hetspmd

from utils import *

model = symbolic_trace(config.get_model(seed=39))
print(model.code, flush=True)

print("Total Number of Ops:", len(model.graph.nodes))
print("Total parameters:", sum(math.prod(p.size()) for p in model.parameters()))

for i, node in enumerate(model.graph.nodes):
    node.meta['id'] = i

hetspmd.init()

flops = hetspmd.stat(model, {
    "input_shape": config.input_shape()
})

print("Total flops:", flops, flush=True)

dgraph = hetspmd.main(model, {
    "input_shape": config.input_shape(),
    "device_flops": [
        config.profiler_data["device_flops"],
        config.profiler_data["device_flops"],
        config.profiler_data["device_flops"] * 0.5,
        config.profiler_data["device_flops"] * 0.5,
    ],
    "all_reduce_bandwidth": config.profiler_data["all_reduce"],
    "all_gather_bandwidth": config.profiler_data["all_gather"],
    "reduce_scatter_bandwidth": config.profiler_data["reduce_scatter"],
    "all_to_all_bandwidth": config.profiler_data["all_to_all"],
    "rank": 0
})

# print(dgraph, flush=True)

# dmodel = torch.fx.GraphModule(model, dgraph)
