"""PyTorch utilities for the UC Irvine course on
'ML & Statistics for Physicists'
"""
import functools

import torch.nn


def sizes_as_string(tensors):
    if isinstance(tensors, torch.Tensor):
        return str(tuple(tensors.size()))
    else:
        return ', '.join([sizes_as_string(T) for T in tensors])

def trace_forward(module, input, output, name='', verbose=False):
    """Implement the module forward hook API.
    
    Parameters
    ----------
    input : tuple or tensor
        Input tensor(s) to this module. We save a detached
        copy to this module's `input` attribute.
    output : tuple or tensor
        Output tensor(s) to this module. We save a detached
        copy to this module's `output` attribute.        
    """
    if isinstance(input, tuple):
        module.input = [I.detach() for I in input]
        if len(module.input) == 1:
            module.input = module.input[0]
    else:
        module.input = input.detach()
    if isinstance(output, tuple):
        module.output = tuple([O.detach() for O in output])
        if len(module.output) == 1:
            module.output = module.output[0]
    else:
        module.output = output.detach()
    if verbose:
        print(f'{name}: IN {sizes_as_string(module.input)} OUT {sizes_as_string(module.output)}')
    
def trace_backward(module, grad_in, grad_out, name='', verbose=False):
    """Implement the module backward hook API.

    Parameters
    ----------
    grad_in : tuple or tensor
        Gradient tensor(s) for each input to this module.
        These are the *outputs* from backwards propagation and we
        ignore them.
    grad_out : tuple or tensor
        Gradient tensor(s) for each output to this module.
        Theser are the *inputs* to backwards propagation and
        we save detached views to the module's `grad` attribute.
        If grad_out is a tuple with only one entry, which is usually
        the case, save the tensor directly.
    """
    if isinstance(grad_out, tuple):
        module.grad = tuple([O.detach() for O in grad_out])
        if len(module.grad) == 1:
            module.grad = module.grad[0]
    else:
        module.grad = grad_out.detach()
    if verbose:
        print(f'{name}: GRAD {sizes_as_string(module.grad)}')

def trace(module, active=True, verbose=False):
    if hasattr(module, '_trace_hooks'):
        # Remove all previous tracing hooks.
        for hook in module._trace_hooks:
            hook.remove()
    if not active:
        return
    module._trace_hooks = []
    for name, submodule in module.named_modules():
        if submodule is module:
            continue
        module._trace_hooks.append(submodule.register_forward_hook(
            functools.partial(trace_forward, name=name, verbose=verbose)))
        module._trace_hooks.append(submodule.register_backward_hook(
            functools.partial(trace_backward, name=name, verbose=verbose)))
