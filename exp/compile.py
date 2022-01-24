import config

import sys
import torch
import torch.fx

from annotator import annotate
from compiler import compile
from utils import *

model = symbolic_trace(config.get_model(seed=39))
annotate(model, { 'x': (config.batch_size, config.seqlen, config.emsize) })
print_annotated_graph(model.graph)

strategy = load(f"strategy_{sys.argv[1]}")

from pprint import pprint
pprint(strategy)

print(model.code)
compile(model, strategy, global_rank=0, local_rank=0, world_size=config.world_size)
print(model.code)
