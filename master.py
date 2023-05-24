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

with measure_time() as wall_time:
    dgraph = hetspmd.main(model, {
        "input_shape": config.input_shape(),
        "device_flops": [ 3858755112937 ] * round(config.world_size / 8 * 2) + [ 2149250936815 ] * round(config.world_size / 8 * 6),
        "all_gather_bandwidth": 815418707,
        "all_gather_by_group_call_bandwidth": 549828906,
        "all_reduce_bandwidth": 476774816,
        "reduce_scatter_bandwidth": 876490907,
        "reduce_scatter_by_group_call_bandwidth": 512358434,
        "all_to_all_bandwidth": 7504501871,
        "rank": 0,
    })

print(flush=True)
print(flush=True)
print(flush=True)
print(flush=True)
print(flush=True)
print(flush=True)
print(wall_time, flush=True)

# print(dgraph, flush=True)

# dmodel = torch.fx.GraphModule(model, dgraph)
