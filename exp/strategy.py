import config
import sys
import torch
import torch.fx

from models import MLP, MLP2, MoE, Transformer
from annotator import annotate
from compiler import compile
from utils import *

if sys.argv[1] == 'mlp':
    model = symbolic_trace(MLP(nhid=config.emsize, nlayers=config.nlayers))
if sys.argv[1] == 'mlp2':
    model = symbolic_trace(MLP2(nhid=config.emsize, nlayers=config.nlayers))
if sys.argv[1] == 'moe':
    model = symbolic_trace(MoE(emsize=config.emsize, nhead=4, nhid=config.nhid, dropout=config.dropout, n_expert=config.n_expert, capacity=config.capacity, nlayers=config.nlayers))
if sys.argv[1] == 'transformer':
    model = symbolic_trace(Transformer(emsize=config.emsize, nhead=4, nhid=config.nhid, dropout=config.dropout, nlayers=config.nlayers))

print(model.code)

annotate(model, { 'x': (config.batch_size, config.seqlen, config.emsize) })
print_annotated_graph(model.graph)

nodes = list(model.graph.nodes)

for i, node in enumerate(nodes):
    node.meta['id'] = i

import spmd

strategy = spmd.spmd(nodes, {}, {})
save(f"strategy_{sys.argv[1]}", strategy)

# from pprint import pprint
# pprint(strategy)
