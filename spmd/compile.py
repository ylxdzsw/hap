from __future__ import annotations
from typing import Any

import torch
import torch.fx
import torch.distributed as dist

import collectives
from utils import *

# stages: [([communication], [computation])]
# communications and computations are python dictionaries.

def compile(
    module: torch.fx.graph_module.GraphModule,
    strategy: list[tuple[dict, dict]],
    rank: int | None = None,
    world_size: int | None = None
):
    if rank is None:
        rank = dist.get_rank()
    if world_size is None:
        world_size = dist.get_world_size()

    nodes = module.graph.nodes

    new_graph = torch.fx.graph.Graph()
    tensor_dict_1 = {} # (origin_name, form) -> node_in_new_graph
    tensor_dict_2 = {}

    def gen_comp(
        tensor_dict: dict[tuple[int, str], torch.fx.node.Node],
        origin_id: int,
        input_forms: dict[str, str], # original_input_name -> form
        output_forms: list[str],
        companions: list[int],
    ):
        raw_node = nodes[origin_id]
        new_node = new_graph.node_copy(raw_node, lambda n: tensor_dict[(n.name, input_forms[n.name])])
        if len(companions) > 0:
            tensor_dict[(raw_node.name, '')] = new_node
            for companion_id, companion_form in zip(companions, output_forms):
                companion_raw_node = nodes[companion_id]
                companion_new_node = new_graph.node_copy(companion_raw_node, lambda n: tensor_dict[(n.name, '')])
                tensor_dict[(companion_raw_node.name, companion_form)] = companion_new_node
        else:
            assert len(output_forms) == 1
            tensor_dict[(raw_node.name, output_forms[0])] = new_node

    def gen_comm(
        tensor_dict: dict[tuple[int, str], torch.fx.node.Node],
        origin_node_id: int,
        old_form: str,
        new_form: str,
        collective_names: list[str],
        collective_args: list[tuple]
    ):
        origin_name = nodes[origin_node_id].name
        node = tensor_dict[(origin_name, old_form)]
        for collective_name, collective_arg in zip(collective_names, collective_args):
            collective_op = ({
                'all_gather': collectives.all_gather,
                'all_reduce': collectives.all_reduce,
                'reduce_scatter': collectives.reduce_scatter,
                'all_to_all': collectives.all_to_all,
                'dynamic_slice': collectives.dynamic_slice,
                'replicate': collectives.replicate
            })[collective_name]
            node = new_graph.call_function(collective_op, (node, *collective_arg))
        tensor_dict[(origin_name, new_form)] = node

    module.default_stream = torch.cuda.default_stream(rank)
    module.stream_1 = torch.cuda.Stream(rank)
    module.stream_2 = torch.cuda.Stream(rank)

    default_stream = new_graph.get_attr("default_stream")
    stream_1 = new_graph.get_attr("stream_1")
    stream_2 = new_graph.get_attr("stream_2")


    debt_computations = []

    for stage in strategy:
        communications, computations = stage

        if len(debt_computations) >= 0:
            for computation in debt_computations:
                gen_comp(tensor_dict_1, **computation)


        if len(communications) != 0:
            ...

