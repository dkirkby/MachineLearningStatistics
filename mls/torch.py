"""PyTorch utilities for the UC Irvine course on
'ML & Statistics for Physicists'
"""
import functools
import copy

import numpy as np

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


def get_lr(self, name='lr'):
    lr_grps = [grp for grp in self.param_groups if name in grp]
    if not lr_grps:
        raise ValueError(f'Optimizer has no parameter called "{name}".')
    if len(lr_grps) > 1:
        raise ValueError(f'Optimizer has multiple parameters called "{name}".')
    return lr_grps[0][name]

def set_lr(self, value, name='lr'):
    lr_grps = [grp for grp in self.param_groups if name in grp]
    if not lr_grps:
        raise ValueError(f'Optimizer has no parameter called "{name}".')
    if len(lr_grps) > 1:
        raise ValueError(f'Optimizer has multiple parameters called "{name}".')
    lr_grps[0][name] = value    

# Add get_lr, set_lr methods to all Optimizer subclasses.
torch.optim.Optimizer.get_lr = get_lr
torch.optim.Optimizer.set_lr = set_lr


def lr_scan(loader, model, loss_fn, optimizer, lr_start=1e-6, lr_stop=1., lr_steps=100):
    """Implement the learning-rate scan described in Smith 2015.
    """
    import matplotlib.pyplot as plt
    # Save the model and optimizer states before scanning.
    model_save = copy.deepcopy(model.state_dict())
    optim_save = copy.deepcopy(optimizer.state_dict())
    # Schedule learning rate to increase in logarithmic steps.
    lr_schedule = np.logspace(np.log10(lr_start), np.log10(lr_stop), lr_steps)
    model.train()
    losses = []
    scanning = True
    while scanning:
        for x_in, y_tgt in loader:
            optimizer.set_lr(lr_schedule[len(losses)])
            y_pred = model(x_in)
            loss = loss_fn(y_pred, y_tgt)
            losses.append(loss.data)
            if len(losses) == lr_steps or losses[-1] > 10 * losses[0]:
                scanning = False
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # Restore the model and optimizer state.
    model.load_state_dict(model_save)
    optimizer.load_state_dict(optim_save)
    # Plot the scan results.
    plt.plot(lr_schedule[:len(losses)], losses, '.')
    plt.ylim(0.5 * np.min(losses), 10 * losses[0])
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Learning rate')
    plt.ylabel('Loss')
    # Return an optimizer with set_lr/get_lr methods, and lr set to half of the best value found.
    idx = np.argmin(losses)
    lr_set = 0.5 * lr_schedule[idx]
    print(f'Recommended lr={lr_set:.3g}.')
    optimizer.set_lr(lr_set)
