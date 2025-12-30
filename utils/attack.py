import torch
import torch.nn as nn
import torch.nn.functional as F

def pgd_kl_training(model, x, y, teacher_nat_logits,epsilon, alpha, n_steps):
        # define KL-loss
        device = next(model.parameters()).device
        criterion_kl = nn.KLDivLoss(size_average=False,reduce=False)
        model.eval()

        if n_steps > 0 :
            x = x.detach() + 0.001 * torch.randn(x.shape).to(device).detach()
        for i in range(n_steps):
            x.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x), dim=1),
                                       F.softmax(teacher_nat_logits, dim=1))
                loss_kl = torch.sum(loss_kl)
            grad = torch.autograd.grad(loss_kl, [x])[0]
            x = x.detach() + alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x - epsilon), x + epsilon)
            x = torch.clamp(x, 0, 1).detach()

        return x

def pgd_attack(model, x, y, epsilon, alpha, n_steps, random_start=True):
    device = next(model.parameters()).device
    ce_loss = nn.CrossEntropyLoss()

    if random_start:
        x_adv = x.detach() + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)
    else:
        x_adv = x.detach()

    for _ in range(n_steps):
        x_adv.requires_grad_()
        logits = model(x_adv)
        loss = ce_loss(logits, y.to(device))
        loss.backward()

        grad = x_adv.grad.detach()
        x_adv = x_adv.detach() + alpha * torch.sign(grad)

        x_pert = torch.clamp(x_adv - x, -epsilon, epsilon)
        x_adv = torch.clamp(x + x_pert, 0, 1)

    return x_adv.detach()


def pgd_attack_training(model, x, y,epsilon, alpha, n_steps):
    device = next(model.parameters()).device
    ce_loss = nn.CrossEntropyLoss()

    x_adv = x.detach() + 0.001 * torch.randn(x.shape).to(device).detach()
    x_adv = torch.clamp(x_adv, 0, 1)

    for _ in range(n_steps):
        x_adv.requires_grad_()
        logits = model(x_adv)
        loss = ce_loss(logits, y.to(device))
        loss.backward()

        grad = x_adv.grad.detach()
        x_adv = x_adv.detach() + alpha * torch.sign(grad)

        x_pert = torch.clamp(x_adv - x, -epsilon, epsilon)
        x_adv = torch.clamp(x + x_pert, 0, 1)

    return x_adv.detach()


def ifgsm_attack(model, x, y, epsilon, alpha, n_steps):
    device = next(model.parameters()).device
    ce_loss = nn.CrossEntropyLoss()

    x_adv = x.detach() + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0, 1)

    for _ in range(n_steps):
        x_adv.requires_grad_()
        logits = model(x_adv)
        loss = ce_loss(logits, y.to(device))
        loss.backward()

        grad = x_adv.grad.detach()
        x_adv = x_adv.detach() + alpha * torch.sign(grad)
        x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()
