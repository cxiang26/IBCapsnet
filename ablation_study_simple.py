"""
简化版 Ablation Study
专注于IBCapsNet-squash核心部件对抗噪能力的影响

实验设计：
1. Baseline: 简单classifier，直接从context encoder输出得到10个类别logits，使用BCE
2. Exp2: 每个类别一个独立classifier，每个输出1个logit，使用BCE
3. Exp3: 每个类别输出16维潜在向量，使用squash和margin loss，包含KL损失
4. Exp4: 在Exp3基础上增加重构网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
import numpy as np
import os
import json
import csv
import time
import logging
import argparse
import glob
from datetime import datetime
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 导入 IBCapsNet 相关类
from IBCapsnet import (
    PrimaryCapsules, EnhancedContextEncoder,
    Squash, margin_loss, ReconstructionDecoder
)
from data_loader import Dataset

USE_CUDA = torch.cuda.is_available()


# ==============================
# 1. 实验配置
# ==============================
class SimpleAblationConfig:
    """简化版 Ablation Study 实验配置"""
    def __init__(self, dataset='mnist', n_epochs=20, batch_size=128, 
                 learning_rate=0.001, seed=42):
        self.dataset = dataset
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        
        # 根据数据集确定输入尺寸
        if dataset in ['mnist', 'mnist-small', 'fashion-mnist', 'fashion-mnist-small']:
            self.input_width, self.input_height, self.input_channel = 28, 28, 1
        elif dataset in ['cifar10', 'cifar10-small', 'svhn']:
            self.input_width, self.input_height, self.input_channel = 32, 32, 3
        else:
            self.input_width, self.input_height, self.input_channel = 28, 28, 1


# ==============================
# 2. 模型变体
# ==============================

class BaselineModel(nn.Module):
    """
    实验1: Baseline
    简单的classifier，直接从context encoder输出得到10个类别logits，使用BCE
    """
    def __init__(self, input_width=28, input_height=28, input_channel=1, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        
        # Conv + Primary Capsules
        self.conv1 = nn.Conv2d(input_channel, 256, kernel_size=9, stride=1)
        self.primary_caps = PrimaryCapsules()
        
        # 计算primary capsules数量
        conv1_out_size = input_width - 8
        primary_caps_spatial = (conv1_out_size - 9 + 1) // 2
        primary_caps_num = 32 * primary_caps_spatial * primary_caps_spatial
        
        # Context Encoder
        self.context_encoder = EnhancedContextEncoder(
            context_dim=256,
            primary_caps_num=primary_caps_num
        )
        
        # 简单classifier: 直接从context得到10个类别logits
        # 结构：Linear(256, 128) -> ReLU -> Linear(128, 10)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = F.relu(self.conv1(x))  # [B, 256, H, W]
        primary = self.primary_caps(x)  # [B, primary_caps_num, 8]
        context = self.context_encoder(primary)  # [B, 256]
        logits = self.classifier(context)  # [B, 10]
        probs = torch.sigmoid(logits)  # [B, 10]
        return probs, torch.tensor(0.0, device=x.device)  # 返回probs和dummy kl
    
    def loss(self, probs, labels, kl, beta=0.0):
        # 使用BCE损失
        ce_loss = F.binary_cross_entropy(probs, labels.float())
        return ce_loss, ce_loss, kl


class MultiClassifierModel(nn.Module):
    """
    实验2: 每个类别一个独立classifier
    每个classifier输出1个logit，使用BCE
    """
    def __init__(self, input_width=28, input_height=28, input_channel=1, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        
        # Conv + Primary Capsules
        self.conv1 = nn.Conv2d(input_channel, 256, kernel_size=9, stride=1)
        self.primary_caps = PrimaryCapsules()
        
        # 计算primary capsules数量
        conv1_out_size = input_width - 8
        primary_caps_spatial = (conv1_out_size - 9 + 1) // 2
        primary_caps_num = 32 * primary_caps_spatial * primary_caps_spatial
        
        # Context Encoder
        self.context_encoder = EnhancedContextEncoder(
            context_dim=256,
            primary_caps_num=primary_caps_num
        )
        
        # 每个类别一个独立的classifier
        # 每个classifier结构：Linear(256, 128) -> ReLU -> Linear(128, 1)
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ) for _ in range(num_classes)
        ])
    
    def forward(self, x):
        x = F.relu(self.conv1(x))  # [B, 256, H, W]
        primary = self.primary_caps(x)  # [B, primary_caps_num, 8]
        context = self.context_encoder(primary)  # [B, 256]
        
        # 每个classifier输出1个logit
        logits = torch.cat([
            self.classifiers[j](context) for j in range(self.num_classes)
        ], dim=1)  # [B, 10]
        
        probs = torch.sigmoid(logits)  # [B, 10]
        return probs, torch.tensor(0.0, device=x.device)  # 返回probs和dummy kl
    
    def loss(self, probs, labels, kl, beta=0.0):
        # 使用BCE损失
        ce_loss = F.binary_cross_entropy(probs, labels.float())
        return ce_loss, ce_loss, kl


class SquashWithKLModel(nn.Module):
    """
    实验3: 每个类别输出16维潜在向量，使用squash和margin loss，包含KL损失
    """
    def __init__(self, input_width=28, input_height=28, input_channel=1, 
                 num_classes=10, latent_dim=16, beta=1e-3):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Conv + Primary Capsules
        self.conv1 = nn.Conv2d(input_channel, 256, kernel_size=9, stride=1)
        self.primary_caps = PrimaryCapsules()
        
        # 计算primary capsules数量
        conv1_out_size = input_width - 8
        primary_caps_spatial = (conv1_out_size - 9 + 1) // 2
        primary_caps_num = 32 * primary_caps_spatial * primary_caps_spatial
        
        # Context Encoder
        self.context_encoder = EnhancedContextEncoder(
            context_dim=256,
            primary_caps_num=primary_caps_num
        )
        
        # 每个类别的编码器（μ, logσ²）
        self.class_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, latent_dim * 2)  # μ and logσ²
            ) for _ in range(num_classes)
        ])
        
        # Squash激活
        self.squash = Squash()
        
        # 先验分布（标准正态）
        self.prior = Normal(0, 1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        B = x.shape[0]
        
        x = F.relu(self.conv1(x))  # [B, 256, H, W]
        primary = self.primary_caps(x)  # [B, primary_caps_num, 8]
        context = self.context_encoder(primary)  # [B, 256]
        
        # 对每个类别生成胶囊
        all_z = []
        total_kl = 0.0
        
        for j in range(self.num_classes):
            params = self.class_encoders[j](context)  # [B, 2*latent_dim]
            mu, logvar = torch.chunk(params, 2, dim=-1)
            z = self.reparameterize(mu, logvar)  # [B, latent_dim]
            all_z.append(z)
            
            # KL divergence to prior
            q_dist = Normal(mu, torch.exp(0.5 * logvar))
            kl = kl_divergence(q_dist, self.prior).sum(dim=-1).mean()
            total_kl += kl
        
        all_z = torch.stack(all_z, dim=1)  # [B, 10, latent_dim]
        
        # 应用squash并计算范数作为存在性概率
        all_z_squashed = self.squash(all_z, dim=-1)  # [B, 10, latent_dim]
        class_probs = torch.norm(all_z_squashed, dim=-1)  # [B, 10]
        
        return class_probs, total_kl / self.num_classes, all_z
    
    def loss(self, probs, labels, kl, beta=None):
        if beta is None:
            beta = self.beta
        # 使用Margin Loss
        ce_loss = margin_loss(probs, labels, m_plus=0.9, m_minus=0.1, lambda_val=0.5)
        total_loss = ce_loss + beta * kl
        return total_loss, ce_loss, kl


class SquashWithKLReconModel(nn.Module):
    """
    实验4: 在Exp3基础上增加重构网络
    """
    def __init__(self, input_width=28, input_height=28, input_channel=1,
                 num_classes=10, latent_dim=16, beta=1e-3, recon_alpha=0.0005):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.beta = beta
        self.recon_alpha = recon_alpha
        
        # 使用Exp3的模型作为基础
        self.base_model = SquashWithKLModel(
            input_width, input_height, input_channel,
            num_classes, latent_dim, beta
        )
        
        # 添加重构解码器
        self.decoder = ReconstructionDecoder(
            latent_dim=latent_dim,
            num_classes=num_classes,
            input_width=input_width,
            input_height=input_height,
            input_channel=input_channel
        )
        self.mse_loss = nn.MSELoss()
    
    def forward(self, x, target=None, do_recon=False):
        probs, kl, all_z = self.base_model(x)
        
        recon_img = None
        if do_recon:
            recon_img = self.decoder(all_z, target=target)
        
        return probs, kl, recon_img
    
    def loss(self, probs, labels, kl, recon_img=None, original_img=None, 
             beta=None, alpha=None):
        if beta is None:
            beta = self.beta
        if alpha is None:
            alpha = self.recon_alpha
        
        # 分类损失（Margin Loss）
        ce_loss = margin_loss(probs, labels, m_plus=0.9, m_minus=0.1, lambda_val=0.5)
        
        # KL损失
        kl_loss = kl
        
        # 重构损失
        recon_loss = torch.tensor(0.0, device=probs.device)
        if recon_img is not None and original_img is not None:
            recon_loss = self.mse_loss(
                recon_img.view(recon_img.size(0), -1),
                original_img.view(original_img.size(0), -1)
            ) * alpha
        
        total_loss = ce_loss + beta * kl_loss + recon_loss
        return total_loss, ce_loss, kl_loss, recon_loss


# ==============================
# 3. 训练和测试函数
# ==============================

def train_model(model, optimizer, train_loader, epoch, n_epochs, logger=None):
    """训练函数"""
    model.train()
    use_recon = hasattr(model, 'recon_alpha')
    
    total_loss = 0.0
    total_ce_loss = 0.0
    total_kl_loss = 0.0
    total_recon_loss = 0.0
    valid_batches = 0
    total_correct = 0
    total_samples = 0
    
    for batch_id, (data, target) in enumerate(train_loader):
        data = data.to(next(model.parameters()).device)
        target = target.to(next(model.parameters()).device)
        labels = F.one_hot(target, 10).float()
        
        optimizer.zero_grad()
        
        if use_recon:
            probs, kl, recon_img = model(data, target=target, do_recon=True)
            loss, ce, kl_val, recon_loss_val = model.loss(
                probs, labels, kl, recon_img, data,
                beta=model.beta if hasattr(model, 'beta') else model.base_model.beta,
                alpha=model.recon_alpha
            )
        else:
            if isinstance(model, SquashWithKLModel):
                probs, kl, _ = model(data)
            else:
                probs, kl = model(data)
            
            if isinstance(model, SquashWithKLModel):
                loss, ce, kl_val = model.loss(probs, labels, kl)
            else:
                loss, ce, kl_val = model.loss(probs, labels, kl)
            recon_loss_val = torch.tensor(0.0, device=data.device)
        
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        pred = probs.argmax(dim=1)
        correct = pred.eq(target).sum().item()
        total_correct += correct
        total_samples += target.size(0)
        
        loss_value = loss.item()
        ce_value = ce.item()
        kl_value = kl_val.item() if isinstance(kl_val, torch.Tensor) else kl_val
        recon_value = recon_loss_val.item() if isinstance(recon_loss_val, torch.Tensor) else 0.0
        
        if not (np.isnan(loss_value) or np.isinf(loss_value)):
            total_loss += loss_value
            total_ce_loss += ce_value
            total_kl_loss += kl_value
            total_recon_loss += recon_value
            valid_batches += 1
    
    avg_loss = total_loss / valid_batches if valid_batches > 0 else 0.0
    avg_ce_loss = total_ce_loss / valid_batches if valid_batches > 0 else 0.0
    avg_kl_loss = total_kl_loss / valid_batches if valid_batches > 0 else 0.0
    avg_recon_loss = total_recon_loss / valid_batches if valid_batches > 0 else 0.0
    avg_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    return avg_loss, avg_ce_loss, avg_kl_loss, avg_recon_loss, avg_accuracy


def test_model(model, test_loader, noise_level=0.0, use_clamp_noise=False):
    """
    测试函数
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        noise_level: 噪声水平（0.0表示无噪声）
        use_clamp_noise: 是否使用clamp noise（加性噪声后clamp到[0,1]）
    
    Returns:
        accuracy: 测试准确率
    """
    model.eval()
    use_recon = hasattr(model, 'recon_alpha')
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(next(model.parameters()).device)
            target = target.to(next(model.parameters()).device)
            
            # 添加噪声（如果需要）
            if noise_level > 0.0:
                if use_clamp_noise:
                    # Clamp noise: 加性噪声后clamp到[0,1]（产生饱和效应）
                    noise = torch.randn_like(data) * noise_level
                    data = torch.clamp(data + noise, 0, 1)
                else:
                    # 普通加性噪声
                    noise = torch.randn_like(data) * noise_level
                    data = data + noise
            
            if use_recon:
                probs, kl, recon_img = model(data, do_recon=False)
            elif isinstance(model, SquashWithKLModel):
                probs, kl, _ = model(data)
            else:
                probs, kl = model(data)
            
            pred = probs.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def test_clamp_noise_robustness(model, test_loader, noise_levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]):
    """
    使用clamp noise测试模型鲁棒性
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        noise_levels: 噪声水平列表
    
    Returns:
        results: 字典，包含每个噪声水平下的准确率
    """
    results = {}
    for noise_level in noise_levels:
        acc = test_model(model, test_loader, noise_level=noise_level, use_clamp_noise=True)
        results[noise_level] = acc
    return results


# ==============================
# 4. 实验运行函数
# ==============================

def run_experiment(experiment_name, model_class, config, train_loader, test_loader, 
                   result_dir, logger, **model_kwargs):
    """
    运行单个实验
    
    Returns:
        results: 包含训练历史的字典
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Running Experiment: {experiment_name}")
    logger.info(f"{'='*60}")
    
    # 创建模型
    model = model_class(
        input_width=config.input_width,
        input_height=config.input_height,
        input_channel=config.input_channel,
        **model_kwargs
    )
    if USE_CUDA:
        model = model.cuda()
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_ce_loss': [],
        'train_kl_loss': [],
        'train_recon_loss': [],
        'train_acc': [],
        'test_acc': [],
        'test_acc_clamp_noise': [],
        'epoch_time': []
    }
    
    best_acc = 0.0
    best_epoch = 0
    
    # 训练循环
    for epoch in range(1, config.n_epochs + 1):
        epoch_start = time.time()
        
        # 训练
        train_loss, train_ce, train_kl, train_recon, train_acc = train_model(
            model, optimizer, train_loader, epoch, config.n_epochs, logger
        )
        
        # 测试（标准测试集）
        test_acc = test_model(model, test_loader)
        
        # Clamp noise测试（使用固定噪声水平0.3进行训练过程中的监控）
        test_acc_clamp_noise = test_model(model, test_loader, noise_level=0.3, use_clamp_noise=True)
        
        epoch_time = time.time() - epoch_start
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_ce_loss'].append(train_ce)
        history['train_kl_loss'].append(train_kl)
        history['train_recon_loss'].append(train_recon)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['test_acc_clamp_noise'].append(test_acc_clamp_noise)
        history['epoch_time'].append(epoch_time)
        
        # 日志
        logger.info(f"Epoch {epoch}/{config.n_epochs}: "
                   f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                   f"Test Acc={test_acc:.4f}, Test Acc (Clamp Noise 0.3)={test_acc_clamp_noise:.4f}, "
                   f"Time={epoch_time:.2f}s")
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            model_path = os.path.join(result_dir, f'{experiment_name}_best.pth')
            torch.save(model.state_dict(), model_path)
    
    # 训练完成后，进行完整的clamp noise鲁棒性测试
    logger.info(f"Running clamp noise robustness test for {experiment_name}...")
    clamp_noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    clamp_noise_results = test_clamp_noise_robustness(model, test_loader, noise_levels=clamp_noise_levels)
    
    # 找到最佳clamp noise准确率（在噪声水平0.3下）
    best_clamp_noise_acc = clamp_noise_results.get(0.3, 0.0)
    final_clamp_noise_acc = history['test_acc_clamp_noise'][-1] if history['test_acc_clamp_noise'] else 0.0
    
    # 保存最终结果
    results = {
        'experiment_name': experiment_name,
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'final_test_acc': history['test_acc'][-1],
        'best_clamp_noise_acc': best_clamp_noise_acc,
        'final_clamp_noise_acc': final_clamp_noise_acc,
        'clamp_noise_results': clamp_noise_results,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_time': sum(history['epoch_time']),
        'avg_epoch_time': np.mean(history['epoch_time']),
        'history': history
    }
    
    logger.info(f"Experiment {experiment_name} completed!")
    logger.info(f"Best Accuracy: {best_acc:.4f} (Epoch {best_epoch})")
    logger.info(f"Clamp Noise (0.3) Accuracy: {best_clamp_noise_acc:.4f}")
    logger.info(f"Total Training Time: {results['total_time']:.2f}s")
    
    return results


# ==============================
# 5. 可视化函数
# ==============================

def visualize_results(all_results, result_dir, dataset_name):
    """可视化实验结果"""
    os.makedirs(os.path.join(result_dir, 'visualizations'), exist_ok=True)
    
    # 1. 训练曲线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    for exp_name, results in all_results.items():
        history = results['history']
        plt.plot(history['test_acc'], label=exp_name, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    for exp_name, results in all_results.items():
        history = results['history']
        plt.plot(history['test_acc_clamp_noise'], label=exp_name, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (Clamp Noise 0.3)')
    plt.title('Clamp Noise Robustness During Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    exp_names = list(all_results.keys())
    final_accs = [all_results[exp]['final_test_acc'] for exp in exp_names]
    clamp_accs = [all_results[exp]['best_clamp_noise_acc'] for exp in exp_names]
    x = np.arange(len(exp_names))
    width = 0.35
    plt.bar(x - width/2, final_accs, width, label='Clean', alpha=0.8)
    plt.bar(x + width/2, clamp_accs, width, label='Clamp Noise 0.3', alpha=0.8)
    plt.xlabel('Experiment')
    plt.ylabel('Accuracy')
    plt.title('Final Performance Comparison')
    plt.xticks(x, exp_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'visualizations', 'training_curves.png'), dpi=150)
    plt.close()
    
    # 2. Clamp Noise鲁棒性对比
    plt.figure(figsize=(10, 6))
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for exp_name, results in all_results.items():
        clamp_results = results['clamp_noise_results']
        accs = [clamp_results.get(nl, 0.0) for nl in noise_levels]
        plt.plot(noise_levels, accs, 'o-', label=exp_name, linewidth=2, markersize=6)
    plt.xlabel('Clamp Noise Level')
    plt.ylabel('Accuracy')
    plt.title('Clamp Noise Robustness Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'visualizations', 'clamp_noise_robustness.png'), dpi=150)
    plt.close()
    
    # 3. 组件贡献分析
    plt.figure(figsize=(12, 6))
    
    exp_names = list(all_results.keys())
    clean_accs = [all_results[exp]['final_test_acc'] for exp in exp_names]
    clamp_accs = [all_results[exp]['best_clamp_noise_acc'] for exp in exp_names]
    
    x = np.arange(len(exp_names))
    width = 0.35
    
    plt.bar(x - width/2, clean_accs, width, label='Clean Accuracy', alpha=0.8)
    plt.bar(x + width/2, clamp_accs, width, label='Clamp Noise (0.3) Accuracy', alpha=0.8)
    
    # 标注数值
    for i, (clean, clamp) in enumerate(zip(clean_accs, clamp_accs)):
        plt.text(i - width/2, clean + 0.01, f'{clean:.3f}', ha='center', fontsize=9)
        plt.text(i + width/2, clamp + 0.01, f'{clamp:.3f}', ha='center', fontsize=9)
    
    plt.xlabel('Experiment')
    plt.ylabel('Accuracy')
    plt.title('Component Contribution Analysis')
    plt.xticks(x, exp_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'visualizations', 'component_contribution.png'), dpi=150)
    plt.close()


# ==============================
# 6. 主函数
# ==============================

def main():
    parser = argparse.ArgumentParser(description='Simple Ablation Study for IBCapsNet-squash')
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'mnist-small', 'fashion-mnist', 'fashion-mnist-small',
                               'cifar10', 'cifar10-small', 'svhn'],
                       help='Dataset to use (default: mnist)')
    parser.add_argument('--n_epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--result_dir', type=str, default=None,
                       help='Result directory (default: auto-generated)')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if USE_CUDA:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建配置
    config = SimpleAblationConfig(
        dataset=args.dataset,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed
    )
    
    # 创建结果目录
    if args.result_dir:
        result_dir = args.result_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_dir = f'ablation_study_simple_{args.dataset}_{timestamp}'
    os.makedirs(result_dir, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(result_dir, 'ablation_study.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Simple Ablation Study for IBCapsNet-squash")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Epochs: {args.n_epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Result directory: {result_dir}")
    logger.info("=" * 60)
    
    # 加载数据集
    if args.dataset.endswith('-small'):
        dataset = Dataset(args.dataset, args.batch_size, samples_per_class=100)
    else:
        dataset = Dataset(args.dataset, args.batch_size)
    train_loader = dataset.train_loader
    test_loader = dataset.test_loader
    
    # 定义所有实验
    experiments = [
        {
            'name': 'exp1_baseline',
            'model_class': BaselineModel,
            'kwargs': {}
        },
        {
            'name': 'exp2_multi_classifier',
            'model_class': MultiClassifierModel,
            'kwargs': {}
        },
        {
            'name': 'exp3_squash_kl',
            'model_class': SquashWithKLModel,
            'kwargs': {
                'latent_dim': 16,
                'beta': 1e-3
            }
        },
        {
            'name': 'exp4_squash_kl_recon',
            'model_class': SquashWithKLReconModel,
            'kwargs': {
                'latent_dim': 16,
                'beta': 1e-3,
                'recon_alpha': 0.0005
            }
        }
    ]
    
    # 运行所有实验
    all_results = {}
    for exp in experiments:
        results = run_experiment(
            exp['name'],
            exp['model_class'],
            config,
            train_loader,
            test_loader,
            result_dir,
            logger,
            **exp['kwargs']
        )
        all_results[exp['name']] = results
    
    # 保存所有结果
    results_file = os.path.join(result_dir, 'all_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # 生成可视化
    logger.info("Generating visualizations...")
    visualize_results(all_results, result_dir, args.dataset)
    
    # 生成摘要报告
    logger.info("Generating summary report...")
    summary = {
        'dataset': args.dataset,
        'experiments': {}
    }
    for exp_name, results in all_results.items():
        summary['experiments'][exp_name] = {
            'best_acc': results['best_acc'],
            'final_test_acc': results['final_test_acc'],
            'best_clamp_noise_acc': results['best_clamp_noise_acc'],
            'clamp_noise_results': results['clamp_noise_results'],
            'total_params': results['total_params']
        }
    
    summary_file = os.path.join(result_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # 打印最终摘要
    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    for exp_name, results in all_results.items():
        logger.info(f"\n{exp_name}:")
        logger.info(f"  Best Accuracy: {results['best_acc']:.4f}")
        logger.info(f"  Final Test Accuracy: {results['final_test_acc']:.4f}")
        logger.info(f"  Clamp Noise (0.3) Accuracy: {results['best_clamp_noise_acc']:.4f}")
        logger.info(f"  Total Parameters: {results['total_params']:,}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"All results saved to: {result_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()


