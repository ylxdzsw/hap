# in-place annotate the shape and possible signatures

from __future__ import annotations
from typing import *

import math
import torch
import torch.fx
import operator
import itertools
import models

# polyfill for old python
if not hasattr(math, "prod"):
    def prod(args):
        result = 1
        for a in args:
            result *= a
        return result
    math.prod = prod

def output_dims(node: torch.fx.node.Node):
    if node.meta.get('output_is_tuple'):
        raise Exception("TODO")
    output_shape = node.meta.get('output_shape')
    if output_shape is None:
        return None
    return len(output_shape)

def is_literal_scalar(x):
    return isinstance(x, int) or isinstance(x, float)

# annotate populates the `meta` field of nodes with following items
# output_shape: the shape of the output of this node. Can be a tuple or tuple of tuple (for multiple outputs)
# output_is_tuple: indicates whether the output is a tuple. True if it is, non-exist if not.
# arg_dict: { argument_name: argument_value_or_node }. The normalized kwarg-only argument dict for call_function and call_method
# signatures: [({ argument_name: argument_form }, output_form)]. a list of possible signatures for call_function and call_method.
# is_adaptive: True if the signature covers all cases of input and the choice should be avoid reforming.
# is_output: if the output of this node is an output of the model
# value: the value of this node if it is known at compile time.
# flops: rough estimation about the number of flops. Copy is also counted as a flop.
# is_buffer: if an parameter is actually a buffer that does not require gradients
def annotate(traced_module: torch.fx.graph_module.GraphModule, input_shapes: "dict[str, tuple[int, ...]]"):
    for node in traced_module.graph.nodes:
        if node.op == 'placeholder':
            node.meta['output_shape'] = input_shapes[node.target]
            continue

        if node.op == 'get_attr':
            try:
                p = traced_module.get_parameter(node.target)
            except AttributeError:
                p = traced_module.get_buffer(node.target)
                node.meta['is_buffer'] = True
            node.meta['output_shape'] = tuple(p.shape)
            continue

        if node.op == 'call_function':
            node.meta['arg_dict'] = normalize_arguments(node)
            annotation_rules[node.target](node)
            continue

        if node.op == 'call_method':
            # assume we only call methods of tensors, and if there is a function in the torch namespace, they are equivalent.
            node.meta['arg_dict'] = normalize_arguments(node)
            annotation_rules[getattr(torch.Tensor, node.target)](node)
            continue

        if node.op == 'output':
            assert len(node.args) == 1 # what happens for multiple outputs?
            node.meta['output_shape'] = node.args[0].meta['output_shape']
            if 'output_is_tuple' in node.args[0].meta:
                node.meta['output_is_tuple'] = node.args[0].meta['output_is_tuple']
            # also marks the input
            node.args[0].meta['is_output'] = True
            continue

        raise Exception("unknown node type")

def best_guess_arg_type(x):
    if isinstance(x, torch.fx.node.Node):
        if x.meta.get('output_is_tuple'):
            return Tuple[torch.Tensor, ...]
        else:
            return torch.Tensor
    return type(x)

def normalize_arguments(node: torch.fx.node.Node) -> dict[str]:
    # `node.normalized_arguments` has a bug: https://github.com/pytorch/pytorch/issues/62554
    from torch.fx.operator_schemas import normalize_function

    try:
        f = getattr(torch, node.target) if isinstance(node.target, str) and hasattr(torch, node.target) else node.target

        return normalize_function(f, node.args, node.kwargs,
            tuple( best_guess_arg_type(arg) for arg in node.args ),
            { name: best_guess_arg_type(value) for name, value in node.kwargs.items() },
            normalize_to_only_use_kwargs=True).kwargs
    except:
        pass

    # special rules for those that `torch.fx.operator_schemas.normalize_function` cannot handle
    if node.target is getattr:
        return { 'obj': node.args[0], 'attr': node.args[1] }
    if node.target is operator.getitem:
        return { 'obj': node.args[0], 'item': node.args[1] }
    if node.target in (operator.floordiv, operator.truediv, operator.add, operator.mul):
        return { 'x': node.args[0], 'y': node.args[1] }
    if node.target in ('contiguous', 'clone'):
        return { 'self': node.args[0] }
    if node.target == 'view':
        return { 'self': node.args[0], **{ f"s{i}": arg for i, arg in enumerate(node.args[1:]) } }
    if node.target == models.switch_gating:
        return { 'gate_input': node.args[0], 'n_expert': node.args[1], 'capacity': node.args[2], 'gate_weight': node.args[3] }
    if node.target == torch.einsum: # TODO
        return { 'code': node.args[0], 'x': node.args[1], 'y': node.args[2] }

    raise Exception("don't know how to normalize node", node)


from functools import partial
annotation_rules = {}

def annotation_rule(target, **kwargs):
    def fn(rule):
        annotation_rules[target] = partial(rule, **kwargs)
        return rule
    return fn

@annotation_rule(torch.tanh, input_name='input', transcendental=True)
@annotation_rule(torch.relu, input_name='input', transcendental=False)
@annotation_rule(torch.sigmoid, input_name='input', transcendental=True)
@annotation_rule(torch.nn.functional.relu, input_name='input', transcendental=False)
@annotation_rule(math.sqrt, input_name='input', transcendental=True)
def annotate_elementwise_unary_non_linear(node: torch.fx.node.Node, input_name: str, transcendental: bool):
    input_node = node.meta['arg_dict'][input_name]
    node.meta['output_shape'] = input_node.meta['output_shape']
    node.meta['signatures'] = [ ({ input_name: f"gather_{i}" }, f"gather_{i}") for i in range(output_dims(input_node)) ]

    flops = math.prod(input_node.meta['output_shape'])
    if transcendental: # rough estimation of 3x costs to compute. They are not important anyway.
        flops *= 3
    node.meta['flops'] = flops

@annotation_rule(torch.nn.functional.linear)
def annotate_affine(node: torch.fx.node.Node):
    input_node, weight_node, bias_node = node.meta['arg_dict']['input'], node.meta['arg_dict']['weight'], node.meta['arg_dict']['bias']

    *leading_shapes, reduction_width = input_node.meta['output_shape']
    out_feat, _ = weight_node.meta['output_shape']
    node.meta['output_shape'] = (*leading_shapes, out_feat)

    signatures = []

    # data parallelism
    for i in range(output_dims(node)-1):
        signatures.append(({ 'input': f"gather_{i}", 'weight': 'full', 'bias': 'full' }, f"gather_{i}"))

    # weight slice
    signatures.append(({ 'input': 'full', 'weight': 'gather_0', 'bias': 'gather_0' }, f"gather_{output_dims(node)-1}"))

    if bias_node is None:
        # inner slice
        signatures.append(({ 'input': f"gather_{output_dims(node)-1}", 'weight': 'gather_1' }, 'reduce'))

        # forward reduce. If the out_feat < in_feat, this can save communication by all-reduce on smaller tensors. Even if the size remains the same, postpone the communication may lead to better overlapping
        signatures.append(({ 'input': 'reduce', 'weight': 'full' }, 'reduce'))

    node.meta['signatures'] = signatures

    # each output element requires reduction_width FMA operations and one copy
    node.meta['flops'] = 3 * reduction_width * out_feat * math.prod(leading_shapes)

@annotation_rule(torch.sum, input_name='input', dims_name='dim', keepdim_name='keepdim')
def annotate_reduction(node: torch.fx.node.Node, input_name: str, dims_name: str, keepdim_name: str):
    input_node = node.meta['arg_dict'][input_name]

    # TODO: implement them. note that torch.sum has multiple signatures and thoses arguments may not be present.
    assert dims_name not in node.meta['arg_dict'] or node.meta['arg_dict'][dims_name] is None
    assert keepdim_name not in node.meta['arg_dict'] or node.meta['arg_dict'][keepdim_name] is None

    node.meta['output_shape'] = ()

    if node.target is torch.sum: # for summation, it is always possible to reduce everything together
        node.meta['signatures'] = [ ({ input_name: f"gather_{i}" }, 'reduce' ) for i in range(output_dims(input_node)) ]
        node.meta['signatures'].append(({ input_name: 'reduce' }, 'reduce' ))

    node.meta['flops'] = math.prod(input_node.meta['output_shape'])

@annotation_rule(getattr)
def annotate_getattr(node: torch.fx.node.Node):
    obj, attr = node.meta['arg_dict']['obj'], node.meta['arg_dict']['attr']

    if attr == 'shape':
        value = obj.meta['output_shape']
        node.meta['output_shape'] = tuple(() for s in value)
        node.meta['output_is_tuple'] = True
        node.meta['value'] = value
        node.meta['signatures'] = [ ({ 'obj': f"gather_{dim}", 'attr': 'full' }, tuple( 'reduce' if i == dim else 'full' for i in range(len(value)) )) for dim in range(len(value)) ]
        return

    raise Exception("don't know how to getattr for", attr)

@annotation_rule(operator.getitem)
def annotate_getitem(node: torch.fx.node.Node):
    obj, item = node.meta['arg_dict']['obj'], node.meta['arg_dict']['item']

    assert isinstance(item, int)
    assert obj.meta['output_is_tuple']

    if 'value' in obj.meta:
        node.meta['value'] = obj.meta['value'][item]

    node.meta['output_shape'] = obj.meta['output_shape'][item]

    signatures = []
    for forms in itertools.product(*(itertools.chain([ f"gather_{i}" for i in range(len(shape)) ], ('full', 'reduce')) for shape in obj.meta['output_shape'])):
        if all(form == 'full' for form in forms):
            continue
        signatures.append(({ 'obj': forms }, forms[item]))
    node.meta['signatures'] = signatures
    node.meta['is_adaptive'] = True

@annotation_rule(operator.floordiv)
@annotation_rule(operator.truediv)
@annotation_rule(operator.mul)
@annotation_rule(operator.add)
def annotate_elementwise_binary(node: torch.fx.node.Node):
    x, y = node.meta['arg_dict']['x'], node.meta['arg_dict']['y']

    if 'value' in x.meta and (is_literal_scalar(y) or 'value' in y.meta):
        node.meta['value'] = node.target(x.meta['value'], y if is_literal_scalar(y) else y.meta['value'])


    if is_literal_scalar(y) or y.meta['output_shape'] == ():
        node.meta['output_shape'] = x.meta['output_shape']
        node.meta['signatures'] = [ ({ 'x': f"gather_{i}", 'y': 'full' }, f"gather_{i}" ) for i in range(output_dims(x)) ]
        if node.target in (operator.floordiv, operator.truediv, operator.mul):
            node.meta['signatures'].append(({ 'x': 'reduce', 'y': 'full' }, 'reduce'))
    elif x.meta['output_shape'] == y.meta['output_shape']:
        node.meta['output_shape'] = x.meta['output_shape']
        node.meta['signatures'] = [ ({ 'x': f"gather_{i}", 'y': f"gather_{i}" }, f"gather_{i}" ) for i in range(output_dims(x)) ]
        if node.target in (operator.floordiv, operator.truediv, operator.mul):
            node.meta['signatures'].append(({ 'x': 'reduce', 'y': 'full' }, 'reduce'))
        if node.target is operator.mul:
            node.meta['signatures'].append(({ 'x': 'full', 'y': 'reduce' }, 'reduce'))
    else:
        # try handle simple broadcast that only involves 1-sized dimensions
        if output_dims(x) != output_dims(y):
            raise Exception("TODO")

        output_shape = ()
        signatures = []
        for dim, (x_size, y_size) in enumerate(zip(x.meta['output_shape'], y.meta['output_shape'])):
            if x_size == y_size:
                signatures.append(({ 'x': f"gather_{dim}", 'y': f"gather_{dim}" }, f"gather_{dim}"))
                output_shape = (*output_shape, x_size)
            elif x_size == 1:
                signatures.append(({ 'x': "full", 'y': f"gather_{dim}" }, f"gather_{dim}"))
                output_shape = (*output_shape, y_size)
            elif y_size == 1:
                signatures.append(({ 'x': f"gather_{dim}", 'y': "full" }, f"gather_{dim}"))
                output_shape = (*output_shape, x_size)
            else:
                raise Exception("TODO")

        node.meta['output_shape'] = output_shape
        node.meta['signatures'] = signatures
        if node.target in (operator.floordiv, operator.truediv, operator.mul):
            node.meta['signatures'].append(({ 'x': 'reduce', 'y': 'full' }, 'reduce'))
        if node.target is operator.mul:
            node.meta['signatures'].append(({ 'x': 'full', 'y': 'reduce' }, 'reduce'))

    node.meta['flops'] = math.prod(node.meta['output_shape'])

@annotation_rule(torch.chunk)
@annotation_rule(torch.Tensor.chunk)
def annotate_chunk(node: torch.fx.node.Node):
    input_node, chunks, dim = node.meta['arg_dict']['input'], node.meta['arg_dict']['chunks'], node.meta['arg_dict']['dim']

    assert isinstance(chunks, int)
    assert isinstance(dim, int)

    if dim < 0:
        dim += output_dims(input_node)

    node.meta['output_shape'] = (tuple( x // chunks if i == dim else x for i, x in enumerate(input_node.meta['output_shape']) ),) * chunks
    node.meta['output_is_tuple'] = True
    # don't allow splitting in dim
    node.meta['signatures'] = [ ({ 'input': f"gather_{i}" }, (f"gather_{i}",) * chunks) for i in range(output_dims(input_node)) if i != dim ]
    node.meta['signatures'].append(({ 'input': 'reduce' }, ('reduce',) * chunks))
    node.meta['flops'] = math.prod(input_node.meta['output_shape'])

@annotation_rule(torch.Tensor.contiguous, input_name='self')
@annotation_rule(torch.Tensor.clone, input_name='input')
def annotate_identity(node: torch.fx.node.Node, input_name: str):
    input_node = node.meta['arg_dict'][input_name]
    node.meta['output_shape'] = input_node.meta['output_shape']
    node.meta['signatures'] = [ ({ input_name: f"gather_{i}" }, f"gather_{i}" ) for i in range(output_dims(input_node)) ]
    node.meta['signatures'].append(({ input_name: 'reduce' }, 'reduce'))
    # node.meta['is_adaptive'] = True # TODO: the meaning of is_adaptive is too overloaded.
    node.meta['flops'] = math.prod(input_node.meta['output_shape'])

@annotation_rule(torch.Tensor.view)
def annotate_view(node: torch.fx.node.Node):
    self_node = node.meta['arg_dict']['self']
    dims = output_dims(self_node)
    shape = tuple( node.meta['arg_dict'][f"s{i}"] for i in range(dims) )
    concrete_shape = [ s if isinstance(s, int) else s.meta['value'] for s in shape ]
    for i, s in enumerate(concrete_shape):
        if s == -1:
            concrete_shape[i] = math.prod(self_node.meta['output_shape']) // math.prod( s for s in concrete_shape if s > 0 )
            break
    node.meta['output_shape'] = tuple(concrete_shape)
    node.meta['signatures'] = [({ 'self': 'reduce', **{ f"s{i}": 'full' for i in range(dims) if isinstance(shape[i], torch.fx.node.Node) } }, 'reduce')]
    node.meta['flops'] = math.prod(self_node.meta['output_shape'])
    # TODO: how to encode the other signatures?

@annotation_rule(torch.transpose, input_name="input")
@annotation_rule(torch.Tensor.transpose, input_name="input")
def annotate_transpose(node: torch.fx.node.Node, input_name: str):
    input_node, dim0, dim1 = node.meta['arg_dict'][input_name], node.meta['arg_dict']['dim0'], node.meta['arg_dict']['dim1']
    if dim0 < 0: dim0 += output_dims(input_node)
    if dim1 < 0: dim1 += output_dims(input_node)
    assert isinstance(dim0, int) and isinstance(dim1, int)
    shape = list(input_node.meta['output_shape'])
    shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
    node.meta['output_shape'] = tuple(shape)
    signatures = []
    for i in range(output_dims(input_node)):
        if i == dim0:
            signatures.append(({ input_name: f"gather_{dim0}" }, f"gather_{dim1}" ))
            continue
        if i == dim1:
            signatures.append(({ input_name: f"gather_{dim1}" }, f"gather_{dim0}" ))
            continue
        signatures.append(({ input_name: f"gather_{i}" }, f"gather_{i}" ))
    signatures.append(({ input_name: 'reduce' }, 'reduce'))
    node.meta['signatures'] = signatures
    node.meta['flops'] = math.prod(input_node.meta['output_shape'])

@annotation_rule(torch.bmm)
def annotate_bmm(node: torch.fx.node.Node):
    input_node, mat2_node = node.meta['arg_dict']['input'], node.meta['arg_dict']['mat2']

    B, N, M = input_node.meta['output_shape']
    *_, P = mat2_node.meta['output_shape']
    node.meta['output_shape'] = (B, N, P)

    node.meta['signatures'] = [
        ({ 'input': 'gather_0', 'mat2': 'gather_0' }, 'gather_0' ),
        ({ 'input': 'gather_1', 'mat2': 'full' }, 'gather_1' ),
        ({ 'input': 'full', 'mat2': 'gather_2' }, 'gather_2' ),
        ({ 'input': 'gather_2', 'mat2': 'gather_1' }, 'reduce' ),
        ({ 'input': 'reduce', 'mat2': 'full' }, 'reduce' ),
        ({ 'input': 'full', 'mat2': 'reduce' }, 'reduce' )
    ]

    node.meta['flops'] = 3 * B * N * M * P

@annotation_rule(torch.matmul)
def annotate_matmul(node: torch.fx.node.Node):
    input_node, other_node = node.meta['arg_dict']['input'], node.meta['arg_dict']['other']

    assert(output_dims(input_node) == 4 and output_dims(other_node) == 3) # for MoE

    B, E, C, M = input_node.meta['output_shape']
    *_, M, P = other_node.meta['output_shape']
    node.meta['output_shape'] = (B, E, C, P)

    node.meta['signatures'] = [
        ({ 'input': 'gather_1', 'other': 'gather_1' }, 'gather_1' ),
        ({ 'input': 'gather_2', 'other': 'full' }, 'gather_2' ),
        ({ 'input': 'full', 'other': 'gather_3' }, 'gather_3' ),
        ({ 'input': 'gather_3', 'other': 'gather_2' }, 'reduce' ),
        ({ 'input': 'reduce', 'other': 'full' }, 'reduce' ),
        ({ 'input': 'full', 'other': 'reduce' }, 'reduce' )
    ]

    node.meta['flops'] = 3 * B * E * C * M * P

@annotation_rule(torch.nn.functional.softmax, ratio=5)
@annotation_rule(torch.log_softmax, ratio=15)
def annotate_softmax(node: torch.fx.node.Node, ratio: int):
    input_node, dim = node.meta['arg_dict']['input'], node.meta['arg_dict']['dim']

    assert isinstance(dim, int)

    if dim < 0:
        dim += output_dims(input_node)

    node.meta['output_shape'] = input_node.meta['output_shape']
    node.meta['signatures'] = [({ 'input': f"gather_{i}" }, f"gather_{i}") for i in range(output_dims(node)) if i != dim ]

    node.meta['flops'] = ratio * math.prod(input_node.meta['output_shape'])

@annotation_rule(torch.nn.functional.dropout)
def annotate_dropout(node: torch.fx.node.Node):
    input_node, inplace = node.meta['arg_dict']['input'], node.meta['arg_dict']['inplace']

    assert inplace is False

    node.meta['output_shape'] = input_node.meta['output_shape']
    node.meta['signatures'] = [ ({ 'input': f"gather_{i}" }, f"gather_{i}" ) for i in range(output_dims(input_node)) ]

    # really don't know how to calculate.
    node.meta['flops'] = 20 * math.prod(input_node.meta['output_shape'])

@annotation_rule(torch.nn.functional.layer_norm)
def annotate_layer_norm(node: torch.fx.node.Node):
    input_node, normalized_shape = node.meta['arg_dict']['input'], node.meta['arg_dict']['normalized_shape']

    node.meta['output_shape'] = input_node.meta['output_shape']
    node.meta['signatures'] = [ ({ 'input': f"gather_{i}", 'weight': 'full', 'bias': 'full' }, f"gather_{i}" ) for i in range(output_dims(input_node)-len(normalized_shape)) ]

    node.meta['flops'] = 10 * math.prod(input_node.meta['output_shape'])

@annotation_rule(models.switch_gating)
def annotate_switch_gating(node: torch.fx.node.Node):
    gate_input, n_expert, capacity, gate_weight = node.meta['arg_dict']['gate_input'], node.meta['arg_dict']['n_expert'], node.meta['arg_dict']['capacity'], node.meta['arg_dict']['gate_weight']

    assert isinstance(n_expert, int)
    assert isinstance(capacity, int)

    b, s, d = gate_input.meta['output_shape']
    assert gate_weight.meta['output_shape'] == (d, n_expert)

    node.meta['output_shape'] = ((b, s, n_expert, capacity), (b, s, n_expert, capacity))
    node.meta['output_is_tuple'] = True
    node.meta['signatures'] = [ ({ 'gate_input': 'gather_0', 'gate_weight': 'full' }, ('gather_0', 'gather_0') ) ]

    node.meta['flops'] = 10 * b * s * d * n_expert * capacity  # TODO: need to use profiling to infer this back, or carefully go through the logic

@annotation_rule(torch.einsum)
def annotate_einsum(node: torch.fx.node.Node):
    code, x, y = node.meta['arg_dict']['code'], node.meta['arg_dict']['x'], node.meta['arg_dict']['y']

    # TODO: solve this
    if code == "bsd,bsec->becd":
        b, s, d = x.meta['output_shape']
        _b, _s, e, c = y.meta['output_shape']

        node.meta['output_shape'] = (b, e, c, d)
        node.meta['signatures'] = [
            ({ 'x': 'gather_0', 'y': 'gather_0' }, 'gather_0'),
            ({ 'x': 'gather_1', 'y': 'gather_1' }, 'reduce'),
            ({ 'x': 'gather_2', 'y': 'full' }, 'gather_3'),
            ({ 'x': 'full', 'y': 'gather_2' }, 'gather_1'),
            ({ 'x': 'full', 'y': 'gather_3' }, 'gather_2'),
        ]

        node.meta['flops'] = 3 * b * s * d * e * c
        return

    if code == "edh,becd->bech":
        e, d, h = x.meta['output_shape']
        b, _e, c, _d = y.meta['output_shape']

        node.meta['output_shape'] = (b, e, c, h)
        node.meta['signatures'] = [
            ({ 'x': 'gather_0', 'y': 'gather_1' }, 'gather_1'),
            ({ 'x': 'gather_1', 'y': 'gather_3' }, 'reduce'),
            ({ 'x': 'gather_2', 'y': 'full' }, 'gather_3'),
            ({ 'x': 'full', 'y': 'gather_0' }, 'gather_0'),
            ({ 'x': 'full', 'y': 'gather_2' }, 'gather_2'),
        ]

        node.meta['flops'] = 3 * e * d * h * b * c
        return

    if code == "ehd,bech->becd":
        e, h, d = x.meta['output_shape']
        b, _e, c, _h = y.meta['output_shape']

        node.meta['output_shape'] = (b, e, c, d)
        node.meta['signatures'] = [
            ({ 'x': 'gather_0', 'y': 'gather_1' }, 'gather_1'),
            ({ 'x': 'gather_1', 'y': 'gather_3' }, 'reduce'),
            ({ 'x': 'gather_2', 'y': 'full' }, 'gather_3'),
            ({ 'x': 'full', 'y': 'gather_0' }, 'gather_0'),
            ({ 'x': 'full', 'y': 'gather_2' }, 'gather_2'),
        ]

        node.meta['flops'] = 3 * e * h * d * b * c
        return

    if code == "becd,bsec->bsd":
        b, e, c, d = x.meta['output_shape']
        _b, s, _c, _c = y.meta['output_shape']

        node.meta['output_shape'] = (b, s, d)
        node.meta['signatures'] = [
            ({ 'x': 'gather_0', 'y': 'gather_0' }, 'gather_0'),
            ({ 'x': 'gather_1', 'y': 'gather_2' }, 'reduce'),
            ({ 'x': 'gather_2', 'y': 'gather_3' }, 'reduce'),
            ({ 'x': 'gather_3', 'y': 'full' }, 'gather_2'),
            ({ 'x': 'full', 'y': 'gather_2' }, 'gather_1'),
        ]

        node.meta['flops'] = 3 * b * e * c * d * s
        return

    raise "TODO"

@annotation_rule(torch.nn.functional.multi_head_attention_forward)
def annotate_attention(node: torch.fx.node.Node):
    query = node.meta['arg_dict']['query']
    key = node.meta['arg_dict']['key']
    value = node.meta['arg_dict']['value']

    in_proj_weight = node.meta['arg_dict']['in_proj_weight']
    in_proj_bias = node.meta['arg_dict']['in_proj_bias']

    assert node.meta['arg_dict']['bias_k'] is None
    assert node.meta['arg_dict']['bias_v'] is None

    out_proj_weight = node.meta['arg_dict']['out_proj_weight']
    out_proj_bias = node.meta['arg_dict']['out_proj_bias']

    assert node.meta['arg_dict']['key_padding_mask'] is None

    attn_mask = node.meta['arg_dict']['attn_mask']

    assert node.meta['arg_dict']['use_separate_proj_weight'] is False

    assert node.meta['arg_dict']['static_k'] is None
    assert node.meta['arg_dict']['static_v'] is None

    L, N, E = query.meta['output_shape']
    S, _N, _E = key.meta['output_shape']

    node.meta['output_shape'] = (L, N, E), (N, L, S)

    node.meta['flops'] = 3 * L * N * E * S + 5 * L * S * N + 3 * L * N * E * S

    node.meta['output_is_tuple'] = True

    addition = {}
    if attn_mask is not None:
        addition['attn_mask'] = 'full'

    node.meta['signatures'] = [
        ({ # DP: gather on N dimension
            'query': 'gather_1',
            'key': 'gather_1',
            'value': 'gather_1',
            'in_proj_weight': 'full',
            'in_proj_bias': 'full',
            'out_proj_weight': 'full',
            'out_proj_bias': 'full',
            **addition
        }, ('gather_1', 'gather_0')),
        ({ # The inequivalent Megatron style transform that causes mismatch on the assignment of weights
            'query': 'gather_2',
            'key': 'gather_2',
            'value': 'gather_2',
            'in_proj_weight': 'gather_0',
            'in_proj_bias': 'gather_0',
            'out_proj_weight': 'gather_1',
            'out_proj_bias': 'full', # !!! very wrong
            **addition
        }, ('reduce', 'reduce')) # not sure about the weights, but it is not used anyway
    ]

@annotation_rule(torch.nn.functional.embedding)
def annotate_embedding(node: torch.fx.node.Node):
    input_node = node.meta['arg_dict']['input']
    weight_node = node.meta['arg_dict']['weight']

    N, S = input_node.meta['output_shape']
    T, E = weight_node.meta['output_shape']

    node.meta['output_shape'] = N, S, E
    node.meta['flops'] = 0
    node.meta['signatures'] = [
        ({ 'input': 'gather_0', 'weight': 'full' }, 'gather_0'),
        ({ 'input': 'gather_1', 'weight': 'full' }, 'gather_1'),
        ({ 'input': 'full', 'weight': 'gather_1' }, 'gather_2')
    ]

@annotation_rule(torch.nn.functional.nll_loss)
def annotate_nll(node: torch.fx.node.Node):
    input_node = node.meta['arg_dict']['input']
    target_node = node.meta['arg_dict']['target']

    assert output_dims(target_node) == output_dims(input_node) - 1

    n_extra_dims = output_dims(input_node) - 2

    assert node.meta['arg_dict']['weight'] is None
    assert node.meta['arg_dict']['reduction'] == 'sum'

    node.meta['output_shape'] = ()
    node.meta['signatures'] = [({ 'input': 'gather_0', 'target': 'gather_0' }, 'reduce')]
    for i in range(n_extra_dims):
        node.meta['signatures'].append(({ 'input': f'gather_{i+2}', 'target': f'gather_{i+1}' }, 'reduce'))
    node.meta['flops'] = math.prod(input_node.meta['output_shape'])
