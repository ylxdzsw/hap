import config
import torch
import torch.fx
import math
import hetspmd

from annotator import annotate
from utils import *

model = symbolic_trace(config.get_model(seed=39))
print(model.code)

annotate(model, config.input_shape())
for i, node in enumerate(model.graph.nodes):
    node.meta['id'] = i

print(hetspmd.main(model, {}))

