import sys
import torch
import torch.fx

sys.path.insert(1, f"{sys.path[0]}/../spmd")

from models import MLP, MLP2, MoE, Transformer
from annotator import annotate
from compiler import compile
from utils import *

if sys.argv[1] == 'mlp':
    model = symbolic_trace(MLP(nhid=2048, nlayers=4))
if sys.argv[1] == 'mlp2':
    model = symbolic_trace(MLP2(nhid=2048, nlayers=4))
if sys.argv[1] == 'moe':
    model = symbolic_trace(MoE(emsize=2048, nhead=4, nhid=4096, dropout=0.1, n_expert=32, capacity=10, nlayers=4), inline_functions=[torch.nn.functional.multi_head_attention_forward])
if sys.argv[1] == 'transformer':
    model = symbolic_trace(Transformer(emsize=2048, nhead=4, nhid=4096, dropout=0.1, nlayers=4), inline_functions=[torch.nn.functional.multi_head_attention_forward])

annotate(model, { 'x': (64, 256, 2048) })
print_annotated_graph(model.graph)

nodes = list(model.graph.nodes)

for i, node in enumerate(nodes):
    node.meta['id'] = i

import spmd

strategy = spmd.spmd(nodes, {}, {})
save(f"strategy_{sys.argv[1]}", strategy)

# from pprint import pprint
# pprint(strategy)
