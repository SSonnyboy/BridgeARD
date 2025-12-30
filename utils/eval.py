import torch
import torch.nn as nn
from .attack import pgd_attack_training

def evaluate(student, test_loader, device, epsilon, alpha, n_steps):
    student.eval()
    ce_loss = nn.CrossEntropyLoss(reduction='none')
    adv_correct = 0
    nat_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            total += labels.size(0)
            with torch.enable_grad():
                adv_images = pgd_attack_training(student, images, labels, epsilon, alpha, n_steps)

            stu_adv_logits = student(adv_images)
            stu_nat_logits = student(images)

            adv_correct += (stu_adv_logits.argmax(1) == labels).sum().item()
            nat_correct += (stu_nat_logits.argmax(1) == labels).sum().item()

    return {
        'student_adv': adv_correct / total,
        'student_nat': nat_correct / total
    }
