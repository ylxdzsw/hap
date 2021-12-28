import sys
import torch
import torch.fx

sys.path.insert(1, f"{sys.path[0]}/../spmd")

from models import MLP, MLP2, MoE, Transformer
from annotator import annotate
from utils import *

# model = symbolic_trace(MLP(nhid=1024, nlayers=4))
# model = symbolic_trace(MLP2(nhid=1024, nlayers=4))
# model = symbolic_trace(MoE(emsize=1024, nhead=4, nhid=4096, dropout=0.1, n_expert=32, capacity=10, nlayers=4), inline_functions=[torch.nn.functional.multi_head_attention_forward])
model = symbolic_trace(Transformer(emsize=1024, nhead=4, nhid=4096, dropout=0.1, nlayers=4), inline_functions=[torch.nn.functional.multi_head_attention_forward])
annotate(model, { 'x': (64, 256, 1024) })
print_annotated_graph(model.graph)

nodes = list(model.graph.nodes)

for i, node in enumerate(nodes):
    node.meta['id'] = i

import spmd

print(spmd.spmd(nodes, {}, {}))
