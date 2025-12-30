import os
import torch
import math
import torch.nn.functional as F


def kl_loss(a,b):
    loss = -a*b + torch.log(b+1e-5)*b
    return loss

def scale_to_magnitude(a, b, c):
    if(math.isclose(a, 0, rel_tol=1e-9)): a += 1e-7
    if(math.isclose(b, 0, rel_tol=1e-9)): b += 1e-7
    if(math.isclose(c, 0, rel_tol=1e-9)): c += 1e-7
    magnitude_a = math.floor(math.log10(abs(a)))
    magnitude_b = math.floor(math.log10(abs(b)))
    target_magnitude = min(magnitude_a , magnitude_b)
    magnitude_c = math.floor(math.log10(abs(c)))
    scale_factor = 10 ** (target_magnitude - magnitude_c)
    scaled_c = scale_factor #*c
    return scaled_c

def push_loss(teacher_logits, students_logits, labels,T = 5):#train_batch_labels
    '''print(teacher_logits.shape)
    print(students_logits.shape)
    print(labels.shape)'''
    teacher_predictions = torch.argmax(teacher_logits, dim=1)
    #print(teacher_predictions.shape)
    diff_indices = (teacher_predictions != labels).nonzero(as_tuple=True)[0]
    diff_teacher_logits = teacher_logits[diff_indices]
    diff_student_logits = students_logits[diff_indices]
    #print(diff_student_logits)
    
    return kl_loss(F.log_softmax(diff_student_logits/T,dim=1),F.softmax(diff_teacher_logits.detach(),dim=1))
