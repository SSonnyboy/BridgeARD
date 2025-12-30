import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .attack import pgd_attack_training
EPS = 1e-20

def kl_loss(a,b):
    loss = -a*b + torch.log(b+1e-5)*b
    return loss

def diff_in_weights(proxy_1, proxy_2):
    diff_dict = OrderedDict()
    proxy_1_state_dict = proxy_1.state_dict()
    proxy_2_state_dict = proxy_2.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(
        proxy_1_state_dict.items(), proxy_2_state_dict.items()
    ):
        if len(old_w.size()) <= 1:
            continue
        if "weight" in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = diff_w
    return diff_dict


def add_into_diff(model, diff_step, diff):
    diff_scale = OrderedDict()
    if not diff:
        diff = diff_step
        names_in_diff = diff_step.keys()
        diff_squeue = []
        w_squeue = []
        for name, param in model.named_parameters():
            if name in names_in_diff:
                diff_squeue.append(diff[name].view(-1))
                w_squeue.append(param.view(-1))
        diff_squeue_all = torch.cat(diff_squeue)
        w_squeue_all = torch.cat(w_squeue)
        scale_value = w_squeue_all.norm() / (diff_squeue_all.norm() + EPS)
        for name, param in model.named_parameters():
            if name in names_in_diff:
                diff_scale[name] = scale_value * diff[name]
    else:
        names_in_diff = diff_step.keys()
        for name in names_in_diff:
            diff[name] = diff[name] + diff_step[name]
        diff_squeue = []
        w_squeue = []
        for name, param in model.named_parameters():
            if name in names_in_diff:
                diff_squeue.append(diff[name].view(-1))
                w_squeue.append(param.view(-1))
        diff_squeue_all = torch.cat(diff_squeue)
        w_squeue_all = torch.cat(w_squeue)
        scale_value = w_squeue_all.norm() / (diff_squeue_all.norm() + EPS)
        for name, param in model.named_parameters():
            if name in names_in_diff:
                diff_scale[name] = scale_value * diff[name]
    return diff, diff_scale


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


class WP:
    def __init__(self, model, adv_teacher, proxy_1, proxy_2, proxy_2_optim, config):
        super(WP, self).__init__()
        self.model, self.teacher = model, adv_teacher
        self.proxy_1 = proxy_1
        self.proxy_2 = proxy_2
        self.proxy_2_optim = proxy_2_optim
        self.gamma = config.gamma
        self.wp_iter = config.wp_iter
        self.at_iter = config.at_iter

    def calc_diff(self, inputs_adv, targets):
        diff = OrderedDict()
        diff_scale = OrderedDict()

        for ii in range(self.wp_iter):
            self.proxy_1.load_state_dict(self.model.state_dict())
            self.proxy_2.load_state_dict(self.model.state_dict())

            add_into_weights(self.proxy_1, diff_scale, coeff=1.0 * self.gamma)
            add_into_weights(self.proxy_2, diff_scale, coeff=1.0 * self.gamma)

            self.proxy_2.train()

            output = self.proxy_2(inputs_adv)
            loss = -F.cross_entropy(output, targets)
            self.proxy_2_optim.zero_grad()
            loss.backward()
            self.proxy_2_optim.step()
            diff_step = diff_in_weights(self.proxy_1, self.proxy_2)
            diff, diff_scale = add_into_diff(self.model, diff_step, diff)

        return diff_scale

    def perturb(self, model_aux, aux_opt, temp_adv, diff, inputs_adv, targets):
        add_into_weights(model_aux, diff, coeff=1.0 * self.gamma)
        adv_img_aux = pgd_attack_training(model_aux, inputs_adv, targets, 8./255, 2./255, self.at_iter)
        aux_out = model_aux(adv_img_aux)
        with torch.no_grad():
            teacher_adv_logits = self.teacher(adv_img_aux)
        loss = torch.mean(kl_loss(F.log_softmax(aux_out,dim=1),F.softmax(teacher_adv_logits.detach()/temp_adv,dim=1)))
        aux_opt.zero_grad()
        loss.backward()
        aux_gradient_dict = OrderedDict()
        for name, param in model_aux.named_parameters():
            if param.grad is not None:
                aux_gradient_dict[name] = param.grad.clone()
        return aux_gradient_dict

import copy
import torch
def get_perturbation(config, model, teacher_adv):
    proxy_1 = copy.deepcopy(model)
    proxy_2 = copy.deepcopy(model)
    lr = 0.01  
    proxy_2_optimizer = torch.optim.SGD(proxy_2.parameters(), lr=lr)
    return WP(model, teacher_adv, proxy_1, proxy_2, proxy_2_optimizer, config)

