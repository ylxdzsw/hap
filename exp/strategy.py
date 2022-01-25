import config
import torch
import torch.fx

from annotator import annotate
from utils import *

model = symbolic_trace(config.get_model(seed=39))

print(model.code)

annotate(model, config.input_shape())
print_annotated_graph(model.graph)

raise SystemExit

nodes = list(model.graph.nodes)

for i, node in enumerate(nodes):
    node.meta['id'] = i

import spmd

strategy = spmd.spmd(nodes, config.profiler_data, {})
save(f"strategy_{config.model_name}", strategy)

# from pprint import pprint
# pprint(strategy)
