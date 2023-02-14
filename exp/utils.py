from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

@contextmanager
def stub(obj, attr, impl):
    unique_symbol = []
    try:
        original_impl = getattr(obj, attr, unique_symbol)
        setattr(obj, attr, impl)
        yield
    finally:
        # TODO: check if the value has been overwritten?
        if original_impl is not unique_symbol:
            setattr(obj, attr, original_impl)
        else:
            delattr(obj, attr)

@contextmanager
def measure_time(name="", disable_gc=False):
    @dataclass
    class Time:
        time: float
        name: str
        def __repr__(self) -> str:
            return f"task {name} finished in {toc - tic:.4}s"

    import time

    result = Time(-1, name)
    if disable_gc:
        import gc
        gc.collect()
        gc.disable()
    tic = time.time()
    yield result
    toc = time.time()
    if disable_gc:
        gc.enable()
    result.time = toc - tic

def symbolic_trace(module, inline_functions=[]):
    import torch.fx
    class Tracer(torch.fx.Tracer):
        def is_leaf_module(*_): return False

    inline_functions = [ f.__code__ for f in inline_functions ]

    import torch.nn.functional
    origin = torch.nn.functional.has_torch_function

    def f(*args):
        import inspect
        if inspect.currentframe().f_back.f_code in inline_functions:
            return False
        return origin(*args)
    with stub(torch.nn.functional, "has_torch_function", f):
        graph = Tracer().trace(module)

    graph.eliminate_dead_code()
    model = torch.fx.graph_module.GraphModule(module, graph)

    from torch.fx.experimental.normalize import NormalizeArgs, NormalizeOperators
    model = NormalizeArgs(model).transform()
    model = NormalizeOperators(model).transform()

    return model

def save(file: str, var: ...):
    import pickle
    with open(file, 'wb') as f:
        pickle.dump(var, f)

def load(file: str) -> ...:
    import pickle
    with open(file, 'rb') as f:
        return pickle.load(f)

def graph_to_dot(graph, strategy):
    import operator
    import torch

    form_dict = {}
    for node in graph.nodes:
        if node.name not in strategy.keys():
            continue
        form_dict[node.name] = strategy[node.name][1]
        for k, v in node.meta['arg_dict'].items():
            if not isinstance(v, torch.fx.node.Node) or v.name in form_dict.keys():
                continue
            form_dict[v.name] = strategy[node.name][0][k]

    dot = ""

    def resolve(node):
        if node.target in (operator.getitem, torch.transpose, torch.Tensor.transpose, torch.relu, "transpose", "contiguous", "view", "clone"):
            return resolve(list(node.all_input_nodes)[0])
        elif node.target in (getattr,):
            return None
        else:
            return node

    def format_node(node):
        s = ""
        if node.name in form_dict.keys():
            s = form_dict[node.name]
        return f"{node.name}: {s}"

    for node in graph.nodes:
        if resolve(node) == node:
            for arg in node.all_input_nodes:
                resolved = resolve(arg)
                if resolved == None:
                    continue
                dot += f"""
                    "{format_node(resolved)}" -> "{format_node(node)}"
                """

    return f"digraph strategy {{{dot}}}"
