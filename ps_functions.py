import torch.nn as nn
import numpy as np
import torch
from copy import deepcopy


def synch_weight(model, model_synch):
    for param, param_synch in zip(model.parameters(), model_synch.parameters()):
        param.data = param_synch.data + 0

def weight_dif(model, model_synch,model_dif):
    for param, param_synch, param_dif in zip(model.parameters(), model_synch.parameters(), model_dif.parameters()):
        param_dif.data = param.data - param_synch.data + 0

def weight_divide(model,num):
    for param in model.parameters():
       param.data = param.data/num

# changing the learning rate of SGD
def lr_change(lr_new, optim):
    for group in optim.param_groups:
        group['lr'] = lr_new


def average_model(net_avg, net_toavg):
    # net_toavg is broadcasted
    # net_avg uses net_toavg to update its parameters
    for param_avg, param_toavg in zip(net_avg.parameters(), net_toavg.parameters()):
        param_avg.data.mul_(0.5)
        param_avg.data.add_(0.5, param_toavg.data)
    return None

def average_model2(net_avg,net_toavg):
    # net_toavg is broadcasted
    # net_avg uses net_toavg to update its parameters
    for param_avg, param_toavg in zip(net_avg.parameters(), net_toavg.parameters()):
        param_avg.data.mul_(1/3)
        param_avg.data.add_(2/3, param_toavg.data)
    return None

def model_Bcast(bcast_model, opt, r, alpha):

    for bcast_group, group in zip(bcast_model.param_groups, opt.param_groups):
        for p_bcast, p in zip(bcast_group['params'], group['params']):

            param_state = opt.state[p]
            if 'momentum_buffer' not in param_state:
                buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
            else:
                buf = param_state['momentum_buffer']

            param_state_bcast = bcast_model.state[p_bcast]
            if 'momentum_buffer' not in param_state_bcast:
                param_state_bcast['momentum_buffer'] = torch.zeros_like(p.data)
            buf_bcast = param_state_bcast['momentum_buffer']

            if r==0:
               param_state_bcast['momentum_buffer'] = buf
            elif p.grad is None:
               param_state_bcast['momentum_buffer']  = ((buf_bcast * alpha) + (buf * (1 - alpha))) * 0.9
               print('hmm')
            else:
               param_state_bcast['momentum_buffer'] = (((buf_bcast * alpha) + (buf * (1 - alpha))) * 0.9) + p.grad.data
    return None
def average_momentum(opt_avg, opt):
    # opt is broadcasted
    # opt_avg uses opt to update its momentum
    for group_avg, group in zip(opt_avg.param_groups, opt.param_groups):
        for p_avg, p in zip(group_avg['params'], group['params']):

            param_state = opt.state[p]
            buf = param_state['momentum_buffer']

            param_state_avg = opt_avg.state[p_avg]

            if 'momentum_buffer' not in param_state_avg:
                buf_avg = param_state_avg['momentum_buffer'] = torch.zeros_like(p.data)
            else:
                buf_avg = param_state_avg['momentum_buffer']

            buf_avg.mul_(0.5)
            buf_avg.add_(0.5, buf)
    return None


def momentum_change(opt, opt_prev):
    for group, group_prev in zip(opt.param_groups, opt_prev.param_groups):
        for p, p_prev in zip(group['params'], group_prev['params']):
            param_state = opt.state[p]
            buf = param_state['momentum_buffer']
            param_state_prev = opt_prev.state[p_prev]
            param_state_prev['momentum_buffer']=buf
    return None

def initialize_zero(model):
    for param in model.parameters():
        param.data.mul_(0)
    return None


def weight_accumulate(model, agg_model, num):

    for param, ps_param in zip(model.parameters(), agg_model.parameters()):
        ps_param.data += param.data / num
    return None


def momentum_zero(opt):
    for groupAvg in (opt.param_groups):  # momentum
        for p_avg in groupAvg['params']:
            param_state_avg = opt.state[p_avg]
            if 'momentum_buffer' not in param_state_avg:
                buf_avg = param_state_avg['momentum_buffer'] = torch.zeros_like(p_avg.data)
            else:
                buf_avg = param_state_avg['momentum_buffer']
            buf_avg.mul_(0)
    return None

def momentum_accumulate(opt_avg,opt, N_w):
    for group_avg, group in zip(opt_avg.param_groups, opt.param_groups):
        for p_avg, p in zip(group_avg['params'], group['params']):
            param_state = opt.state[p]
            buf = param_state['momentum_buffer']
            param_state_avg = opt_avg.state[p_avg]
            buf_avg = param_state_avg['momentum_buffer']
            buf_avg.add_(1 / N_w, buf)
    return None

def momentum_Avg(opt_avg,opt):
    for group_avg, group in zip(opt_avg.param_groups, opt.param_groups):
        for p_avg, p in zip(group_avg['params'], group['params']):
            param_state_avg = opt_avg.state[p_avg]
            buf_avg = param_state_avg['momentum_buffer']
            opt.state[p]['momentum_buffer'] = buf_avg
    return None
