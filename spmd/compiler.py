from __future__ import annotations
from typing import Any

import operator

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

    nodes = list(module.graph.nodes)

    new_graph = torch.fx.graph.Graph()
    tensor_dict_1 = {} # (origin_name, form) -> node_in_new_graph
    tensor_dict_2 = {}

    module.default_stream = torch.cuda.default_stream(rank)
    module.stream_1 = torch.cuda.Stream(rank)
    module.stream_2 = torch.cuda.Stream(rank)

    default_stream = new_graph.get_attr("default_stream")
    stream_1 = new_graph.get_attr("stream_1")
    stream_2 = new_graph.get_attr("stream_2")

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
                if companion_id is None:
                    continue
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
        collectives: list[str],
    ):
        origin_name = nodes[origin_node_id].name
        node = tensor_dict[(origin_name, old_form)]
        for collective_str in collectives:
            collective_op, collective_args = parse_collective_str(collective_str)
            node = new_graph.call_function(collective_op, (node, *collective_args))
        tensor_dict[(origin_name, new_form)] = node

    def barrier():
        ev1 = new_graph.call_method("record_event", (stream_1,))
        ev2 = new_graph.call_method("record_event", (stream_2,))
        new_graph.call_method("wait_event", (stream_2, ev1))
        new_graph.call_method("wait_event", (stream_1, ev2))

    def default_wait_all():
        new_graph.call_method("wait_stream", (default_stream, stream_1))
        new_graph.call_method("wait_stream", (default_stream, stream_2))

    def all_wait_default():
        new_graph.call_method("wait_stream", (stream_1, default_stream))
        new_graph.call_method("wait_stream", (stream_2, default_stream))

    def gen_placeholder(raw_node):
        # not needed assuming that placeholders are in the first stage
        # default_wait_all()
        # new_graph.call_function(torch.cuda.set_stream, (default_stream,))
        new_input = new_graph.node_copy(raw_node)
        new_input_chunks = new_graph.call_function(torch.chunk, (new_input, 2, 0))
        tensor_dict_1[(raw_node.name, 'full')] = new_graph.call_function(operator.getitem, (new_input_chunks, 0))
        tensor_dict_2[(raw_node.name, 'full')] = new_graph.call_function(operator.getitem, (new_input_chunks, 1))
        all_wait_default()

    def gen_output(raw_node):
        loss_node_name = raw_node.args[0].name
        default_wait_all()
        new_graph.call_function(torch.cuda.set_stream, (default_stream,))
        aggregated_loss = new_graph.call_function(operator.add, (tensor_dict_1[(loss_node_name, 'reduce')], tensor_dict_2[(loss_node_name, 'reduce')]))
        new_graph.output(aggregated_loss)

    debt_computations = []

    output = None
    strategy.append(([], [])) # append an empty stage to flush the debts

    for stage in strategy:
        communications, computations = stage

        syncs = 0 # a simple encoding: 1: wait for stream 1, 2: wait for stream 2, 3: both
        def sync():
            nonlocal syncs
            if syncs == 1:
                new_graph.call_method("wait_stream", (stream_2, stream_1))
            if syncs == 2:
                new_graph.call_method("wait_stream", (stream_1, stream_2))
            if syncs == 3:
                barrier()
            syncs = 0

        # handle inputs and outputs
        for computation in computations:
            raw_node = nodes[computation['origin_id']]
            if raw_node.op == 'placeholder':
                gen_placeholder(raw_node)
            if raw_node.op == 'output':
                assert output is None
                output = raw_node
            if raw_node.op == 'get_attr': # TODO: what if a model has two statments that load the same parameter? We should at least check for this.
                assert len(computation['output_forms']) == 1
                form = computation['output_forms'][0]
                if form.startswith('gather'):
                    dim = int(form[-1])
                    p = module.get_parameter(raw_node.target)
                    p.data = torch.chunk(p.data, world_size, dim)[rank]

        computations = [ comp for comp in computations if nodes[comp['origin_id']].op != 'placeholder' and nodes[comp['origin_id']].op != 'output' ]

        if len(communications) > 0:
            new_graph.call_function(torch.cuda.set_stream, (stream_1,))
            for communication in communications:
                gen_comm(tensor_dict_1, **communication)
            syncs += 1

        if len(debt_computations) > 0:
            new_graph.call_function(torch.cuda.set_stream, (stream_2,))
            for computation in debt_computations:
                gen_comp(tensor_dict_2, **computation)
            syncs += 2

        sync()

        if len(computations) > 0:
            new_graph.call_function(torch.cuda.set_stream, (stream_1,))
            for computation in computations:
                gen_comp(tensor_dict_1, **computation)
            syncs += 1

        if len(communications) > 0:
            new_graph.call_function(torch.cuda.set_stream, (stream_2,))
            for communication in communications:
                gen_comm(tensor_dict_2, **communication)
            syncs += 2

        sync()

        debt_computations = computations

    gen_output(output)

    module.graph = new_graph
    # module.recompile() # should be automatic in the setter

def parse_collective_str(collective_str):
    collective_op_dict = {
        'all_gather': collectives.all_gather,
        'all_reduce': collectives.all_reduce,
        'reduce_scatter': collectives.reduce_scatter,
        'all_to_all': collectives.all_to_all,
        'dynamic_slice': collectives.dynamic_slice,
        'replicate': collectives.replicate
    }
    re_str = "^({})(_\\d+)?(_\\d+)?$".format('|'.join(collective_op_dict.keys()))

    import re
    m = re.search(re_str, collective_str)

    collective_op = collective_op_dict[m.group(1)]
    collective_args = *(int(v[1:]) for v in m.groups()[1:] if v is not None),
    return collective_op, collective_args
