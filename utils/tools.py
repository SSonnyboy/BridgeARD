import torch
import numpy as np

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def get_lr_scheduler(epoch, epochs, base_lr=0.1, warmup_epochs=0, warmup_start_lr=0):
    if epoch < warmup_epochs:
        return warmup_start_lr + (base_lr - warmup_start_lr) * epoch / warmup_epochs

    t = (epoch - warmup_epochs) / (epochs - warmup_epochs)
    cosine_term = 0.5 + 0.5 * np.cos(np.pi * t)
    exponential_decay = np.exp(-0.01 * t ** 2)
    return base_lr * cosine_term * exponential_decay


def get_teacher_lr(epoch, teacher_warmup_epoch=50, base_lr=0.0001, epochs=300):
    if epoch < teacher_warmup_epoch:
        return 0.0

    t = (epoch - teacher_warmup_epoch) / (epochs - teacher_warmup_epoch)
    cosine_term = 0.5 + 0.5 * np.cos(np.pi * t)
    exponential_decay = np.exp(-0.01 * t ** 2)
    return base_lr * cosine_term * exponential_decay