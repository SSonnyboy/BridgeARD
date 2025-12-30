import torch
import torch.nn as nn
import copy

import torch.optim as optim
from pathlib import Path
from datasets.loader import get_loader
from models.get_model import get_student_model, get_teacher_model_clean, get_teacher_model_adv
from utils.attack import pgd_attack, pgd_attack_training, pgd_kl_training
from utils.config import get_config_from_args
from utils.checkpoint import load_teacher_model, load_checkpoint, save_checkpoint
from utils.tools import AverageMeter, setup_seed, get_lr_scheduler, get_teacher_lr
from utils.eval import evaluate
import torch.nn.functional as F
# from perturbations import get_perturbation
from utils.wp import get_perturbation
import numpy as np
import argparse

def kl_loss(a,b):
    loss = -a*b + torch.log(b+1e-5)*b
    return loss

def entropy_value(a):
    value = torch.log(a+1e-5)*a
    return value

def get_training_state(temp_adv, temp_nat, temp_learn_rate, weight_learn_rate,
                       weight, best_score, init_loss_nat, init_loss_adv):
    """
    Collect all training state variables into a single dictionary.
    This makes it easier to save and load complete training state.
    """
    return {
        'temp_adv': temp_adv,
        'temp_nat': temp_nat,
        'temp_learn_rate': temp_learn_rate,
        'weight_learn_rate': weight_learn_rate,
        'weight': weight.copy(),
        'best_score': best_score,
        'init_loss_nat': init_loss_nat,
        'init_loss_adv': init_loss_adv,
    }

def restore_training_state(extra_state):
    """
    Restore training state from checkpoint.
    Returns a dictionary with all training state variables.
    """
    return {
        'temp_adv': extra_state.get('temp_adv', 1),
        'temp_nat': extra_state.get('temp_nat', 1),
        'temp_learn_rate': extra_state.get('temp_learn_rate', 0.001),
        'weight_learn_rate': extra_state.get('weight_learn_rate', 0.025),
        'weight': extra_state.get('weight', {'adv_loss': 0.5, 'nat_loss': 0.5}),
        'best_score': extra_state.get('best_score', 0),
        'init_loss_nat': extra_state.get('init_loss_nat', None),
        'init_loss_adv': extra_state.get('init_loss_adv', None),
    }



def train(config, resume_path=None):
    device = torch.device(f'cuda:{config.gpu}' if torch.cuda.is_available() else 'cpu')
    setup_seed(config.seed)

    train_loader, test_loader = get_loader(config)

    student = get_student_model(config.student, config.num_classes).to(device)
    aux_student = get_student_model(config.student, config.num_classes).to(device)

    wa_model = copy.deepcopy(student)
    exp_avg = student.state_dict()

    teacher_nat = get_teacher_model_clean(config.num_classes).to(device)
    teacher_adv = get_teacher_model_adv(config.num_classes).to(device)

    teacher_nat = load_teacher_model(teacher_nat, config.teacher_nat_path, device)
    teacher_adv = load_teacher_model(teacher_adv, config.teacher_adv_path, device)

    teacher_nat.eval()
    teacher_adv.eval()

    student_optimizer = optim.SGD(student.parameters(), lr=config.base_lr, momentum=0.9, weight_decay=2e-4)
    aux_opt = torch.optim.SGD(aux_student.parameters(), lr=0.01)
    
    start_epoch = 0
    eval_epoch = 70 if config.task == "tinynet" else 215
    ema_start_epoch = 20 if config.task == "tinynet" else 100
    
    epsilon = config.epsilon / 255.0
    alpha = config.alpha / 255.0
    checkpoint_dir = Path('outputs') / f'{config.task}_{config.student}_{config.gamma}_{config.wp_iter}_{config.at_iter}_{config.scale}'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Initialize training state
    temp_max = 10
    temp_min = 1
    tau = 0.9995
    # Default training state values
    temp_adv = 1
    temp_nat = 1
    temp_learn_rate = 0.001
    weight_learn_rate = 0.025
    weight = {
        "adv_loss": 1/2.0,
        "nat_loss": 1/2.0,
    }
    init_loss_nat = None
    init_loss_adv = None
    best_score = 0

    factor_step = config.factor / config.epochs
    factor = 0.0
    perturbation = get_perturbation(config, student, teacher_adv)
    result_file = checkpoint_dir / 'results.txt'
    epochs = config.epochs
    # Resume training from checkpoint
    if resume_path and Path(resume_path).exists():
        print(f'Resuming training from checkpoint: {resume_path}')
        student, student_optimizer, start_epoch, extra_state = load_checkpoint(
            student, resume_path, device, student_optimizer, strict=True
        )

        # Restore all training state variables at once
        state_dict = restore_training_state(extra_state)
        temp_adv = state_dict['temp_adv']
        temp_nat = state_dict['temp_nat']
        temp_learn_rate = state_dict['temp_learn_rate']
        weight_learn_rate = state_dict['weight_learn_rate']
        weight = state_dict['weight']
        best_score = state_dict['best_score']
        init_loss_nat = state_dict['init_loss_nat']
        init_loss_adv = state_dict['init_loss_adv']

        start_epoch = start_epoch + 1  # Start from next epoch
        print(f'Resumed from epoch {start_epoch - 1}')
        print(f'  Best score: {best_score:.4f}')
        print(f'  Temperature: adv={temp_adv:.4f}, nat={temp_nat:.4f}')
        print(f'  Weight: adv_loss={weight["adv_loss"]:.4f}, nat_loss={weight["nat_loss"]:.4f}')
    else:
        # Create new result file if not resuming
        with open(result_file, 'w') as f:
            f.write('epoch student_adv student_nat score\n')

    for epoch in range(start_epoch, epochs + 1):
        student.train()
        factor += factor_step
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            student.train()
            img_nat = images + torch.empty_like(images).uniform_(
                -epsilon, epsilon
            )
            img_nat = torch.clamp(img_nat, min=0, max=1).detach()
            diff = None
            adv_images = pgd_attack_training(student, images, labels, epsilon, alpha, config.n_steps)

            # sample noise direction
            delta_adv = adv_images - images
            delta_nat = img_nat - images
            rand1 = torch.rand(1).item()
            delta_imd = rand1 * delta_nat + (1-rand1) * delta_adv
            img_imd = torch.clamp((images + delta_imd), min=0, max=1).detach()

            with torch.no_grad():
                teacher_nat_out = teacher_nat(torch.cat((img_nat, img_imd), dim=0))
                teacher_adv_out = teacher_adv(torch.cat((adv_images, img_imd), dim=0))
                teacher_nat_logits, teacher_nat_logits_imd = teacher_nat_out.chunk(2)
                teacher_adv_logits, teacher_adv_logits_imd = teacher_adv_out.chunk(2)

            aux_student.load_state_dict(student.state_dict())
            diff = perturbation.calc_diff(adv_images, labels)
            aux_gradient_dict = perturbation.perturb(aux_student, aux_opt, temp_adv, diff, adv_images, labels)

            student_out = student(torch.cat((adv_images, img_nat, img_imd), dim=0))
            student_adv_logits, student_nat_logits, student_imd_logits = student_out.chunk(3)
            
            teacher_imd_logits = rand1 * teacher_nat_logits_imd + (1-rand1) * teacher_adv_logits_imd
            temp_imd = rand1 * temp_nat + (1-rand1) * temp_adv
            
            loss_imd = torch.mean(kl_loss(F.log_softmax(student_imd_logits,dim=1),F.softmax(teacher_imd_logits.detach()/temp_imd,dim=1)))
            loss_adv = torch.mean(kl_loss(F.log_softmax(student_adv_logits,dim=1),F.softmax(teacher_adv_logits.detach()/temp_adv,dim=1)))
            loss_nat = torch.mean(kl_loss(F.log_softmax(student_nat_logits,dim=1),F.softmax(teacher_nat_logits.detach()/temp_nat,dim=1)))

            adv_teacher_entropy = torch.mean(entropy_value(F.softmax(teacher_adv_logits.detach()/temp_adv,dim=1)))
            nat_teacher_entropy = torch.mean(entropy_value(F.softmax(teacher_nat_logits.detach()/temp_nat,dim=1)))
            temp_adv = temp_adv - temp_learn_rate * torch.sign((adv_teacher_entropy.detach() / nat_teacher_entropy.detach() - 1)).item()
            temp_nat = temp_nat - temp_learn_rate * torch.sign((nat_teacher_entropy.detach() / adv_teacher_entropy.detach() - 1)).item()
            temp_adv = max(min(temp_max, temp_adv), temp_min)
            temp_nat = max(min(temp_max, temp_nat), temp_min)

            #the model update, loss has been updated above
            # loss_nat, loss_adv = weight["adv_loss"]*kl_Loss1, weight["nat_loss"]*kl_Loss2
            
            total_loss = loss_nat + loss_adv + config.scale * loss_imd
            student_optimizer.zero_grad()
            total_loss.backward()
            for name, param in student.named_parameters():
                if param.grad is not None:
                    param.grad = param.grad + factor * aux_gradient_dict[name].to(param.device)
            # if epoch < 1:
            #     torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1)
            student_optimizer.step()

            # weight average
            if epoch < ema_start_epoch:
                for key, value in student.state_dict().items():
                    exp_avg[key] = value 
            else:
                for key, value in student.state_dict().items():
                    exp_avg[key] = (1 - tau) * value + tau * exp_avg[key]

            if step % 100 == 0:
                print(
                    f'Epoch [{epoch}] Step [{step}/{len(train_loader)}] Loss: {total_loss.item():.4f} '
                    f'loss_nat: {loss_nat.item():.4f} loss_adv: {loss_adv.item():.4f} '
                )
        
        if epoch in config.lr_steps:
            for param_group in student_optimizer.param_groups:
                param_group['lr'] *= 0.1
            temp_learn_rate *= 0.1
            weight_learn_rate *= 0.1

        wa_model.load_state_dict(exp_avg)
        wa_model.eval()

        if epoch % 10 == 0 or epoch == 1 or epoch > eval_epoch:
            results = evaluate(wa_model, test_loader, device, epsilon, alpha, 20)
            score = (results['student_adv'] + results['student_nat']) / 2

            print(f'\nEpoch [{epoch}] Evaluation Results:')
            print(f'Student - Adv: {results["student_adv"]:.4f} Nat: {results["student_nat"]:.4f}')
            print()

            with open(result_file, 'a') as f:
                f.write(
                    f'{epoch} {results["student_adv"]:.4f} {results["student_nat"]:.4f} {score:.4f} \n'
                )

            if score > best_score:
                best_score = score
                # Save best model with complete training state
                extra_state = get_training_state(
                    temp_adv, temp_nat, temp_learn_rate, weight_learn_rate,
                    weight, best_score, init_loss_nat, init_loss_adv
                )
                save_checkpoint(wa_model, student_optimizer, epoch, checkpoint_dir / 'student_best.pth',
                              extra_state=extra_state)

        # Save latest checkpoint with complete training state for resuming
        extra_state = get_training_state(
            temp_adv, temp_nat, temp_learn_rate, weight_learn_rate,
            weight, best_score, init_loss_nat, init_loss_adv
        )
        save_checkpoint(wa_model, student_optimizer, epoch, checkpoint_dir / f'student_latest.pth',
                       extra_state=extra_state)
    print(f'Training completed! Best score: {best_score:.4f}')


if __name__ == '__main__':
    config, args = get_config_from_args()

    # Get resume path from command line
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parsed_args, _ = parser.parse_known_args()
    resume_path = parsed_args.resume

    print('Config:')
    print(config)
    if resume_path:
        print(f'Resume checkpoint: {resume_path}')
    print()

    train(config, resume_path=resume_path)
