#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重构的评估脚本 - 支持AutoAttack、白盒、黑盒三种评估模式
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from datasets import get_loader
from models import get_student_model, get_teacher_model_clean, get_teacher_model_adv
from utils import pgd_attack_training, load_teacher_model, get_config_from_args

# 尝试导入autoattack和torchattacks
try:
    from autoattack import AutoAttack
    HAS_AUTOATTACK = True
except ImportError:
    HAS_AUTOATTACK = False
    logger.warning("AutoAttack not installed. Install with: pip install autoattack")

try:
    import torchattacks
    HAS_TORCHATTACKS = True
except ImportError:
    HAS_TORCHATTACKS = False
    logger.warning("torchattacks not installed. Install with: pip install torchattacks")


class AttackEvaluator:
    def __init__(self, classnum, device, aa_path=None):
        self.device = device
        self.class_num = classnum
        self.aa_path = aa_path

    def attack_pgd(self, model, images, labels, epsilon=8./255, alpha=2./255, n_steps=20):
        device = next(model.parameters()).device
        ce_loss = torch.nn.CrossEntropyLoss().to(device)
        train_ifgsm_data = images.detach() + torch.zeros_like(images).uniform_(-epsilon,epsilon)
        train_ifgsm_data = torch.clamp(train_ifgsm_data,0,1)
        for i in range(n_steps):
            train_ifgsm_data.requires_grad_()
            logits = model(train_ifgsm_data)
            loss = ce_loss(logits,labels.to(device))
            loss.backward()
            train_grad = train_ifgsm_data.grad.detach()
            train_ifgsm_data = train_ifgsm_data + alpha*torch.sign(train_grad)
            train_ifgsm_data = torch.clamp(train_ifgsm_data.detach(),0,1)
            train_ifgsm_pert = train_ifgsm_data - images
            train_ifgsm_pert = torch.clamp(train_ifgsm_pert,-epsilon,epsilon)
            train_ifgsm_data = images + train_ifgsm_pert
            train_ifgsm_data = train_ifgsm_data.detach()
        return train_ifgsm_data
    
    def attack_fgsm(self, model, images, labels, epsilon=8./255):
        device = next(model.parameters()).device
        ce_loss = torch.nn.CrossEntropyLoss().to(device)

        images.requires_grad_()
        logits = model(images)
        loss = ce_loss(logits, labels.to(device))
        loss.backward()
        
        data_grad = images.grad.detach()
        sign_data_grad = data_grad.sign()
        
        perturbed_data = images + epsilon * sign_data_grad
        perturbed_data = torch.clamp(perturbed_data, 0, 1) 
        return perturbed_data


    def attack_cw_inf(self, model, images, labels, epsilon=8./255, confidence=50, lr=2./255, steps=30):
        device = next(model.parameters()).device
        perturbation = torch.zeros_like(images).to(device).requires_grad_()
        for _ in range(steps):
            output = model(images + perturbation)
            target_onehot = F.one_hot(labels, num_classes=self.class_num).float().to(device)
            real = torch.sum(target_onehot * output, dim=1)
            other = torch.max((1 - target_onehot) * output - target_onehot * 10000, dim=1)[0]
            loss = -torch.clamp(real - other + confidence, min=0.).mean()  
            grad = torch.autograd.grad(loss, perturbation)[0]
            perturbation = (perturbation + lr * torch.sign(grad)).clamp(-epsilon, epsilon)
            perturbation = perturbation.detach().requires_grad_()
        adversarial_input = images + perturbation
        adversarial_input = torch.clamp(adversarial_input, 0, 1) 
        return adversarial_input 

    def evaluate_natural(self, model, test_loader, model_name="Model"):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        nat_acc = correct / total
        logger.info(f'{model_name} - Natural Accuracy: {nat_acc:.4f}')
        return {'natural': nat_acc, 'correct': correct, 'total': total}

    def evaluate_whitebox(self, model, test_loader, model_name="Model"):
        logger.info(f"\n{'='*60}")
        logger.info(f"{model_name} - White-box Attack Evaluation")
        logger.info(f"{'='*60}")

        model.eval()
        results = {
            'natural': 0,
            'pgd_trades': 0,      # PGD with small steps (epsilon/4 per step)
            'pgd_standard': 0,    # PGD with normal steps
            'fgsm': 0,
            'cw_linf': 0,
        }
        total = 0

        with torch.enable_grad():  # 需要开启梯度用于攻击
            for batch_idx, (images, labels) in enumerate(test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                batch_size = labels.size(0)
                total += batch_size

                # 自然准确率
                with torch.no_grad():
                    outputs = model(images)
                    _, pred = torch.max(outputs, 1)
                    results['natural'] += (pred == labels).sum().item()

                # PGD-TRADES (小步长)
                adv_trades = self.attack_pgd(model, images.clone(), labels,
                                           epsilon=8./255, alpha=0.003, n_steps=20)
                with torch.no_grad():
                    outputs = model(adv_trades)
                    _, pred = torch.max(outputs, 1)
                    results['pgd_trades'] += (pred == labels).sum().item()

                # PGD-Standard (标准步长)
                adv_standard = self.attack_pgd(model, images.clone(), labels,
                                             epsilon=8./255, alpha=2./255, n_steps=20)
                with torch.no_grad():
                    outputs = model(adv_standard)
                    _, pred = torch.max(outputs, 1)
                    results['pgd_standard'] += (pred == labels).sum().item()

                # FGSM
                adv_fgsm = self.attack_fgsm(model, images.clone(), labels)
                with torch.no_grad():
                    outputs = model(adv_fgsm)
                    _, pred = torch.max(outputs, 1)
                    results['fgsm'] += (pred == labels).sum().item()

                # C&W
                adv_cw = self.attack_cw_inf(model, images.clone(), labels)
                with torch.no_grad():
                    outputs = model(adv_cw)
                    _, pred = torch.max(outputs, 1)
                    results['cw_linf'] += (pred == labels).sum().item()

                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"  Processed {batch_idx + 1} batches...")

        # 计算准确率
        for key in results:
            results[key] /= total

        # 计算 avg 与 NRR
        nat = results['natural']
        extra = {}
        for k in ['pgd_trades', 'pgd_standard', 'fgsm', 'cw_linf']:
            adv = results[k]
            avg = (adv + nat) / 2.0
            nrr = 0.0 if (adv + nat) == 0 else 2.0 * adv * nat / (adv + nat)
            extra[f'{k}_wr'] = avg
            extra[f'{k}_nrr'] = nrr

        # 打印原始指标
        logger.info(f"\n{model_name} - White-box Evaluation Results:")
        logger.info(f"  Natural Accuracy:        {results['natural']:.4f}")
        for k in ['pgd_trades', 'pgd_standard', 'fgsm', 'cw_linf']:
            logger.info(f"  {k.upper()} Accuracy:        {results[k]:.4f}")

        # 打印 avg 和 nrr
        logger.info("\n  --- WR ---")
        for k in ['pgd_trades', 'pgd_standard', 'fgsm', 'cw_linf']:
            logger.info(f"  {k.upper()} WR:         {extra[f'{k}_wr']:.4f}")

        logger.info("\n  --- NRR ---")
        for k in ['pgd_trades', 'pgd_standard', 'fgsm', 'cw_linf']:
            logger.info(f"  {k.upper()} NRR:         {extra[f'{k}_nrr']:.4f}")

        # 合并结果
        results.update(extra)
            
        # # 打印结果
        # logger.info(f"\n{model_name} - White-box Evaluation Results:")
        # logger.info(f"  Natural Accuracy:        {results['natural']:.4f}")
        # logger.info(f"  PGD-TRADES Accuracy:     {results['pgd_trades']:.4f}")
        # logger.info(f"  PGD-Standard Accuracy:   {results['pgd_standard']:.4f}")
        # logger.info(f"  FGSM Accuracy:           {results['fgsm']:.4f}")
        # logger.info(f"  C&W L-inf Accuracy:      {results['cw_linf']:.4f}")
        return results

    def evaluate_blackbox(self, student, teacher, test_loader, student_name="Student"):
        logger.info(f"\n{'='*60}")
        logger.info(f"{student_name} - Black-box Attack Evaluation")
        logger.info(f"{'='*60}")

        student.eval()
        teacher.eval()

        results = {
            'natural': 0,
            'pgd_trades': 0,
            'cw_linf': 0,
            "square": 0,
        }
        total = 0
        square = torchattacks.Square(student, norm='Linf', eps=8/255, n_queries=100)
        with torch.enable_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                batch_size = labels.size(0)
                total += batch_size

                # 自然准确率
                with torch.no_grad():
                    outputs = student(images)
                    _, pred = torch.max(outputs, 1)
                    results['natural'] += (pred == labels).sum().item()

                # 使用teacher生成对抗样本，攻击student
                # PGD-TRADES
                adv_trades = self.attack_pgd(teacher, images.clone(), labels,
                                           epsilon=8./255, alpha=0.003, n_steps=20)
                with torch.no_grad():
                    outputs = student(adv_trades)
                    _, pred = torch.max(outputs, 1)
                    results['pgd_trades'] += (pred == labels).sum().item()

                adv_square = square(images.clone(),labels)
                with torch.no_grad():
                    outputs = student(adv_square)
                    _, pred = torch.max(outputs, 1)
                    results['square'] += (pred == labels).sum().item()

                # C&W
                adv_cw = self.attack_cw_inf(teacher, images.clone(), labels)
                with torch.no_grad():
                    outputs = student(adv_cw)
                    _, pred = torch.max(outputs, 1)
                    results['cw_linf'] += (pred == labels).sum().item()

                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"  Processed {batch_idx + 1} batches...")

        # 计算准确率
        for key in results:
            results[key] /= total

        # 计算 avg 与 NRR
        nat = results['natural']
        extra = {}
        for k in ['pgd_trades', 'cw_linf', 'square']:
            adv = results[k]
            avg = (adv + nat) / 2.0
            nrr = 0.0 if (adv + nat) == 0 else 2.0 * adv * nat / (adv + nat)
            extra[f'{k}_wr'] = avg
            extra[f'{k}_nrr'] = nrr

        # 打印原始指标
        logger.info(f"\n{student_name} - Black-box Evaluation Results (Teacher as Attacker):")
        logger.info(f"  Natural Accuracy:        {results['natural']:.4f}")
        for k in ['pgd_trades', 'cw_linf', 'square']:
            logger.info(f"  {k.upper()} Accuracy:        {results[k]:.4f}")

        # 打印 avg 和 nrr
        logger.info("\n  --- WR ---")
        for k in ['pgd_trades', 'cw_linf', 'square']:
            logger.info(f"  {k.upper()} WR:         {extra[f'{k}_wr']:.4f}")
        logger.info("\n  --- NRR ---")
        for k in ['pgd_trades', 'cw_linf', 'square']:
            logger.info(f"  {k.upper()} NRR:         {extra[f'{k}_nrr']:.4f}")

        # 合并结果
        results.update(extra)

        # # 打印结果
        # logger.info(f"\n{student_name} - Black-box Evaluation Results (Teacher as Attacker):")
        # logger.info(f"  Natural Accuracy:        {results['natural']:.4f}")
        # logger.info(f"  PGD-TRADES Accuracy:     {results['pgd_trades']:.4f}")
        # logger.info(f"  square Accuracy:         {results['square']:.4f}")
        # logger.info(f"  C&W L-inf Accuracy:      {results['cw_linf']:.4f}")
        return results

    def evaluate_autoattack(self, model, test_loader):
        if not HAS_AUTOATTACK:
            logger.warning("AutoAttack not available. Skipping AutoAttack evaluation.")
            return None

        model.eval()
        xs, ys = [], []
        for x, y in test_loader:
            xs.append(x.to(self.device))
            ys.append(y.to(self.device))
        x_test = torch.cat(xs, dim=0)
        y_test = torch.cat(ys, dim=0)

        logger.info(f"Running AutoAttack on {x_test.size(0)} images...")

        # 运行AutoAttack
        adversary = AutoAttack(model, norm='Linf', eps=8./255, version='standard', verbose=True, log_path=self.aa_path, device=self.device)
        adversary.run_standard_evaluation(x_test, y_test, bs=128)
        return None  # AutoAttack直接打印结果


class ExperimentEvaluator:
    """基于实验文件夹的评估器"""

    def __init__(self, exp_folder, config_path='config/cifar10.yaml', exp_root='outputs'):
        """
        初始化评估器

        Args:
            exp_folder: 实验文件夹名称
            config_path: 配置文件路径
            exp_root: 实验文件夹的根目录（'outputs' 或 'checkpoints'）
        """
        self.exp_folder = exp_folder
        self.exp_root = exp_root
        self.exp_path = Path(__file__).parent / exp_root / exp_folder

        if not self.exp_path.exists():
            raise FileNotFoundError(f"Experiment folder not found: {self.exp_path}")

        # 加载配置
        from utils import load_config
        self.config = load_config(config_path)
        self.device = torch.device(f'cuda:{self.config.gpu}' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # 获取数据加载器
        _, self.test_loader = get_loader(self.config)
        logger.info(f"Loaded test data")

        # 初始化模型
        self.student = None
        self.teacher_nat = None
        self.teacher_adv = None

        # 初始化报告
        self.report_lines = []

    def load_student_model(self):
        """从实验文件夹加载学生模型"""
        self.student = get_student_model(self.config.student, self.config.num_classes).to(self.device)

        # 查找最佳模型或最新检查点
        model_path = self._find_best_model()

        if model_path and model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.student.load_state_dict(state_dict)
            logger.info(f"✓ Loaded student model from: {model_path}")
        else:
            logger.warning(f"Student model not found in {self.exp_path}")
            return False

        return True

    def load_teacher_models(self):
        """加载教师模型"""
        self.teacher_nat = get_teacher_model_clean(self.config.num_classes).to(self.device)
        self.teacher_adv = get_teacher_model_adv(self.config.num_classes).to(self.device)

        nat_path = Path(__file__).parent / self.config.teacher_nat_path
        adv_path = Path(__file__).parent / self.config.teacher_adv_path

        if nat_path.exists():
            self.teacher_nat = load_teacher_model(self.teacher_nat, str(nat_path), self.device)
            logger.info(f"✓ Loaded NAT teacher from: {nat_path}")
        else:
            logger.warning(f"NAT teacher not found: {nat_path}")

        if adv_path.exists():
            self.teacher_adv = load_teacher_model(self.teacher_adv, str(adv_path), self.device)
            logger.info(f"✓ Loaded ADV teacher from: {adv_path}")
        else:
            logger.warning(f"ADV teacher not found: {adv_path}")

    def _find_best_model(self):
        """查找最佳模型（best.pth）或最新检查点"""
        best_model = self.exp_path / 'student_best.pth'
        if best_model.exists():
            return best_model

        # 查找最新的检查点
        checkpoints = list(self.exp_path.glob('checkpoint_*.pth'))
        if checkpoints:
            latest = max(checkpoints, key=lambda p: int(p.stem.split('_')[1]))
            return latest

        return None

    def evaluate_whitebox(self, save_report=False):
        """
        运行白盒攻击评估

        Args:
            save_report: 是否保存报告
        """
        if not self.student:
            logger.error("Student model not loaded")
            return None

        if save_report:
            self.report_lines = []
            header = f"""
{'='*70}
Evaluation Report: {self.exp_folder}
Experiment Path: {self.exp_path}
{'='*70}

Evaluation Mode: White-box
Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
"""
            self._add_report_line(header)

        evaluator = AttackEvaluator(self.config.num_classes, self.device)
        results = evaluator.evaluate_whitebox(self.student, self.test_loader, model_name="Student")

        if save_report and results:
            self._add_report_line("\nWhite-box Attack Evaluation Results:")
            self._add_report_line("-"*70)
            # for key, val in results.items():
            #     self._add_report_line(f"{key}: {val:.4f}")
            # 先打印原始指标
            for k in ['natural', 'pgd_trades', 'pgd_standard', 'fgsm', 'cw_linf']:
                if k in results:
                    self._add_report_line(f"{k}: {results[k]:.4f}")

            # 再打印 avg 和 nrr
            self._add_report_line("\n--- WR ---")
            for k in ['pgd_trades', 'pgd_standard', 'fgsm', 'cw_linf']:
                if f'{k}_wr' in results:
                    self._add_report_line(f"{k}_avg: {results[f'{k}_wr']:.4f}")
            self._add_report_line("\n--- NRR ---")
            for k in ['pgd_trades', 'pgd_standard', 'fgsm', 'cw_linf']:
                if f'{k}_nrr' in results:
                    self._add_report_line(f"{k}_nrr: {results[f'{k}_nrr']:.4f}")
            self._add_report_line(f"\n{'='*70}")
            self._add_report_line("Report generated successfully")
            self._add_report_line(f"{'='*70}\n")
            self._save_report('whitebox')

        return results

    def evaluate_blackbox(self, save_report=False):
        """
        运行黑盒攻击评估

        Args:
            save_report: 是否保存报告
        """
        if not self.student or not self.teacher_nat:
            logger.error("Student or teacher models not loaded")
            return None

        if save_report:
            self.report_lines = []
            header = f"""
{'='*70}
Evaluation Report: {self.exp_folder}
Experiment Path: {self.exp_path}
{'='*70}

Evaluation Mode: Black-box
Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
Teacher Model: ADV Teacher
"""
            self._add_report_line(header)

        evaluator = AttackEvaluator( self.config.num_classes, self.device)
        results = evaluator.evaluate_blackbox(self.student, self.teacher_adv, self.test_loader,
                                            student_name="Student")

        if save_report and results:
            self._add_report_line("\nBlack-box Attack Evaluation Results:")
            self._add_report_line("-"*70)
            for k in ['natural', 'pgd_trades', 'cw_linf', 'square']:
                if k in results:
                    self._add_report_line(f"{k}: {results[k]:.4f}")

            # 再打印 avg 和 nrr
            self._add_report_line("\n--- WR ---")
            for k in ['pgd_trades', 'cw_linf', 'square']:
                if f'{k}_wr' in results:
                    self._add_report_line(f"{k}_avg: {results[f'{k}_wr']:.4f}")
            self._add_report_line("\n--- NRR ---")
            for k in ['pgd_trades', 'cw_linf', 'square']:
                if f'{k}_nrr' in results:
                    self._add_report_line(f"{k}_nrr: {results[f'{k}_nrr']:.4f}")
            self._add_report_line(f"\n{'='*70}")
            self._add_report_line("Report generated successfully")
            self._add_report_line(f"{'='*70}\n")
            self._save_report('blackbox')

        return results

    def evaluate_autoattack(self):
        """
        运行AutoAttack评估
        """
        report_path = self.exp_path / f"eval_aa.txt"
        if not self.student:
            logger.error("Student model not loaded")
            return None

        evaluator = AttackEvaluator( self.config.num_classes, self.device, aa_path=report_path)
        evaluator.evaluate_autoattack(self.student, self.test_loader)
        return None

    def _add_report_line(self, line):
        """添加报告行"""
        self.report_lines.append(line)

    def _save_report(self, mode):
        """
        保存评估报告到实验文件夹

        Args:
            mode: 评估模式（autoattack, whitebox, blackbox, all）
        """
        report_path = self.exp_path / f"eval_{mode}.txt"

        report_content = '\n'.join(self.report_lines)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"✓ Report saved to: {report_path}")
        return report_path

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate student models with multiple attack methods')
    parser.add_argument('--config', type=str, default='config/cifar10.yaml',
                        help='Path to config file')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id')

    # 新的参数：实验文件夹
    parser.add_argument('--exp', type=str, default=None,
                        help='Experiment folder name (e.g., "exp_1")')
    parser.add_argument('--exp-root', type=str, default='outputs',
                        choices=['outputs', 'checkpoints'],
                        help='Root directory for experiments (default: outputs)')
    parser.add_argument('--mode', type=str, choices=['autoattack', 'whitebox', 'blackbox', 'all'],
                        default='all',
                        help='Evaluation mode (default: all)')
    args = parser.parse_args()

    # 配置日志
    logger.remove()  # 移除默认处理器
    logger.add(sys.stderr, format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>")
    logger.add("eval.log", format="{time} | {level: <8} | {message}")

    # 新模式：基于实验文件夹的评估
    if args.exp:
        try:
            evaluator = ExperimentEvaluator(args.exp, args.config, exp_root=args.exp_root)
        except FileNotFoundError as e:
            logger.error(f"Error: {e}")
            logger.info(f"\nTip: Available root directories:")
            logger.info(f"  - outputs/ (default)")
            logger.info(f"  - checkpoints/")
            sys.exit(1)

        if args.mode == 'whitebox':
            evaluator.load_student_model()
            evaluator.evaluate_whitebox(save_report=True)
        elif args.mode == 'blackbox':
            evaluator.load_student_model()
            evaluator.load_teacher_models()
            evaluator.evaluate_blackbox(save_report=True)
        elif args.mode == 'autoattack':
            evaluator.load_student_model()
            evaluator.evaluate_autoattack()
        else:  # 'all'
            evaluator.load_student_model()
            evaluator.load_teacher_models()
            evaluator.evaluate_whitebox(save_report=True)
            evaluator.evaluate_blackbox(save_report=True)
            evaluator.evaluate_autoattack()


if __name__ == '__main__':
    main()
