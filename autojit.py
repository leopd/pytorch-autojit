"""Automatically apply torch.jit to a function by just adding @autojit annotation to that function.
Only works for functions that only take positional arguments, which are all torch Tensors.
Re-traces the function for every shape of input.
"""
import time
import torch

def autojit(func):
    """Dynamically jit-compile the annotated function with different shapes of tensors.
    Only works for methods that take only position arguments, and each is a torch.Tensor
    """
    _shape_cache = {}
    def inner(*args):
        shapes = tuple([a.shape for a in args])
        if shapes in _shape_cache:
            jitted_func = _shape_cache[shapes]
        else:
            print(f"JIT tracing {shapes}")
            jitted_func = torch.jit.trace(func, args)
            _shape_cache[shapes] = jitted_func
        return jitted_func(*args)
    return inner
