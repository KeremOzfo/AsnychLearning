import torch
from torch.optim.optimizer import Optimizer
from copy import deepcopy


class SGD(Optimizer):

    def __init__(self, params, lr=0.1, momentum=0,
                 dampening=0, weight_decay=0, args=None):
        defaults = dict(lr=lr, momentum=momentum,
                        dampening=dampening,
                        weight_decay=weight_decay,
                        args=args)
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)

    def step(self, apply_momentum=True):
        """Performs a single optimization step.
        Avoid to use momentum to accumulate the gradients from other workers.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        for group in self.param_groups:
            # retrieve para.
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            for p in group['params']:

                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]

                # add weight decay.
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                # apply local momentum.
                if momentum != 0 and apply_momentum:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf
                p.data.add_(-group['lr'], d_p)
        return None


def define_optimizer(model, lr, momentum, w_decay=0):
    # define the param to optimize.
    params_dict = dict(model.named_parameters())
    # params = [
    #     {
    #         'params': [value],
    #         'name': key,
    #         'weight_decay': w_decay
    #     }
    #     for key, value in params_dict.items()
    # ]
    params = []
    for key, value in params_dict.items():
        a = 1
        if 'bn' in key:
            a = 0
        params.append({
            'params': [value],
            'name': key,
            'weight_decay': w_decay * a
        })

    # define the optimizer.
    return SGD(
        params, lr=lr,
        momentum=momentum)