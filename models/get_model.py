from .stu_models import mobilenet_v2, resnet18, PreActResNet18, PreActResNet34
from .tea_models.resnet import cifar10_resnet56
from .tea_models.wideresnet import WideResNet
from .tea_models.preactnet import PreActResNet34 as PreActResNet34_adv
from .tea_models.mypreact import PreActResNet34 as PreActResNet34_nat

from .tea_models.widecifar100 import WideResNet_22_6, WideResNet_70_16


def get_student_model(model_name="mobilenet_v2", num_classes=10):
    if model_name == "mobilenet_v2":
        return mobilenet_v2(num_classes)
    elif model_name == "resnet18":
        return resnet18(num_classes)
    elif model_name == "preact18":
        return PreActResNet18(num_classes)
    elif model_name == "preact34":
        return PreActResNet34(num_classes)
    else:
        raise ValueError(f"Unknown student model: {model_name}")


def get_teacher_model_clean(num_classes=10):
    if num_classes == 10:
        return cifar10_resnet56()
    elif num_classes == 100:
        return WideResNet_22_6()
    elif num_classes == 200:
        return PreActResNet34_nat(num_classes)
    else:
        raise ValueError(f"Unknown number of classes for teacher model: {num_classes}")


def get_teacher_model_adv(num_classes=10):
    if num_classes == 10:
        return WideResNet(depth=34, num_classes=10, widen_factor=10)
    elif num_classes == 100:
        return WideResNet_70_16()
    elif num_classes == 200:
        return PreActResNet34_adv(num_classes)
    else:
        raise ValueError(f"Unknown number of classes for adversarial teacher model: {num_classes}")
    