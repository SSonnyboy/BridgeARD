from .cifar import get_loader_cifar
from .tinynet import get_loader_tinynet


def get_loader(config):
    if config.task in ['cifar10', 'cifar100']:
        return get_loader_cifar(config)
    elif config.task == 'tinynet':
        return get_loader_tinynet(config)
    else:
        raise ValueError(f'Unknown task: {config.task}')
