import config
import torch
import torch.fx
import numpy as np
import math

from annotator import annotate
from utils import *

model = symbolic_trace(config.get_model(seed=39))

print(model.code)

print("Total parameters:", sum(math.prod(p.size()) for p in model.parameters()))

annotate(model, config.input_shape())
print_annotated_graph(model.graph)

nodes = list(model.graph.nodes)

for i, node in enumerate(nodes):
    node.meta['id'] = i

if config.profile_noise > 0:
    for node in nodes:
        if 'flops' not in node.meta:
            continue
        noise = 2 * (np.random.rand() - 0.5) * config.profile_noise
        node.meta['flops'] = int(node.meta['flops'] * (1 + noise))

    for k in config.profiler_data:
        noise = 2 * (np.random.rand() - 0.5) * config.profile_noise
        config.profiler_data[k] = int(config.profiler_data[k] * (1 + noise))

import spmd

hints = {}

if config.use_hints:
    for node in nodes:
        if node.target == torch.nn.functional.conv2d:
            hints[node.name] = ['gather_1']
        if node.target == torch.nn.functional.embedding:
            hints[node.name] = ['gather_2']
        # TODO: also add einsum? need to match the einsum code

for node in nodes:
    if node.name.startswith("src_mask"):
        hints[node.name] = ['replicate']

with measure_time("gen strategy") as t:
    strategy = spmd.spmd(nodes, config.profiler_data, hints)

print(t)

save(f"strategy_{config.model_name}", strategy)

# from pprint import pprint
# pprint(strategy)
