"""
IBCapsNet vs CapsNet 对比实验
验证 IBCapsNet 的优势：
1. 理论基础：信息瓶颈原理 vs 动态路由
2. 计算效率：VAE一次前向 vs 迭代路由
3. 训练稳定性：KL散度正则化 vs Margin Loss
4. 小样本学习能力
5. 鲁棒性（噪声、旋转）
6. 参数效率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import time
import json
import os
from datetime import datetime
from tqdm import tqdm
import logging
import csv
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from capsnet import CapsNet
from data_loader import Dataset
from train_capsnet import Config

# 尝试导入 IBCapsnet（尝试多种可能的文件名）
IBCAPSNET_AVAILABLE = False
from IBCapsnet import IBCapsNetWithRecon
IBCAPSNET_AVAILABLE = True

# 导入 LeNet
from train_lenet import LeNet


USE_CUDA = True if torch.cuda.is_available() else False


# ==============================
# LeNet 支持多通道输入
# ==============================
class LeNetMultiChannel(nn.Module):
    """
    LeNet-5 architecture 支持多通道输入（1通道用于MNIST，3通道用于CIFAR-10/SVHN）
    支持不同输入尺寸：28x28 (MNIST) 和 32x32 (CIFAR-10/SVHN)
    """
    def __init__(self, num_classes=10, in_channels=1, input_size=28):
        super(LeNetMultiChannel, self).__init__()
        self.input_size = input_size
        # 第一个卷积层: in_channels -> 6 channels, 5x5 kernel
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, padding=2)
        # 第一个池化层: 2x2
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 第二个卷积层: 6 -> 16 channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # 第二个池化层: 2x2
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # 计算全连接层输入尺寸
        # 对于28x28输入（MNIST）: 28x28 -> 14x14 -> 5x5 (16*5*5=400)
        # 对于32x32输入（CIFAR-10/SVHN）: 32x32 -> 16x16 -> 6x6 (16*6*6=576)
        if input_size == 28:
            fc_input_size = 16 * 5 * 5
        elif input_size == 32:
            fc_input_size = 16 * 6 * 6
        else:
            # 使用自适应池化处理未知尺寸
            fc_input_size = 16 * 5 * 5  # 默认值
            self.adaptive_pool = nn.AdaptiveAvgPool2d(5)  # 统一到5x5
        # 全连接层
        self.fc1 = nn.Linear(fc_input_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        # Conv1 + ReLU + Pool1
        x = self.pool1(F.relu(self.conv1(x)))
        # Conv2 + ReLU + Pool2
        x = self.pool2(F.relu(self.conv2(x)))
        
        # Flatten - 使用自适应池化确保尺寸一致
        if hasattr(self, 'adaptive_pool'):
            x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # 如果尺寸不匹配，添加适配层
        if x.size(1) != self.fc1.in_features:
            if not hasattr(self, 'adapt_fc'):
                self.adapt_fc = nn.Linear(x.size(1), self.fc1.in_features).to(x.device)
            x = self.adapt_fc(x)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ==============================
# 实验配置
# ==============================
EXPERIMENTS = {
    'accuracy': {
        'name': '准确率对比',
        'epochs': 20,
        'description': '对比最终分类准确率'
    },
    'training_speed': {
        'name': '训练速度对比',
        'epochs': 5,
        'description': '对比每个epoch的训练时间'
    },
    'few_shot': {
        'name': '小样本学习',
        'epochs': 30,
        'samples_per_class': [100, 200, 500],
        'description': '对比小样本学习能力'
    },
    'robustness': {
        'name': '鲁棒性测试',
        'epochs': 10,
        'noise_levels': [0.1, 0.2, 0.3],
        'rotation_angles': [15, 30, 45],
        'description': '测试对噪声和旋转的鲁棒性'
    }
}


# ==============================
# 训练函数
# ==============================
def train_capsnet(model, optimizer, train_loader, epoch, n_epochs, logger=None):
    """训练传统 CapsNet"""
    model.train()
    total_loss = 0.0
    total_margin_loss = 0.0
    total_recon_loss = 0.0
    total_correct = 0
    total_samples = 0
    n_batch = len(train_loader)
    
    start_time = time.time()
    
    for batch_id, (data, target) in enumerate(tqdm(train_loader, desc=f'CapsNet Epoch {epoch}/{n_epochs}')):
        # 打印第一个batch的数据统计信息
        if batch_id == 0 and epoch == 1:
            print(f"\n[CapsNet] First batch data statistics:")
            print(f"  Data shape: {data.shape}")
            print(f"  Data min: {data.min().item():.6f}")
            print(f"  Data max: {data.max().item():.6f}")
            print(f"  Data mean: {data.mean().item():.6f}")
            print(f"  Data std: {data.std().item():.6f}")
            if logger:
                logger.info(f"[CapsNet] First batch - min: {data.min().item():.6f}, max: {data.max().item():.6f}, mean: {data.mean().item():.6f}, std: {data.std().item():.6f}")
        
        target_onehot = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
        data, target_onehot = Variable(data), Variable(target_onehot)
        
        if USE_CUDA:
            data, target_onehot = data.cuda(), target_onehot.cuda()
        
        optimizer.zero_grad()
        output, reconstructions, masked = model(data)
        
        # 分别计算 margin loss 和 reconstruction loss
        margin_loss = model.margin_loss(output, target_onehot)
        recon_loss = model.reconstruction_loss(data, reconstructions)
        loss = margin_loss + recon_loss
        
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        pred = np.argmax(masked.data.cpu().numpy(), 1)
        correct = np.sum(pred == target.numpy())
        total_correct += correct
        total_samples += target.size(0)
        
        total_loss += loss.item()
        total_margin_loss += margin_loss.item()
        total_recon_loss += recon_loss.item()
    
    avg_loss = total_loss / n_batch
    avg_margin_loss = total_margin_loss / n_batch
    avg_recon_loss = total_recon_loss / n_batch
    avg_acc = total_correct / total_samples
    elapsed_time = time.time() - start_time
    
    return avg_loss, avg_acc, elapsed_time, avg_margin_loss, avg_recon_loss


def train_ibcapsnet(model, optimizer, train_loader, epoch, n_epochs, logger=None):
    """训练 IBCapsNet"""
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0
    total_recon = 0.0
    total_correct = 0
    total_samples = 0
    n_batch = len(train_loader)
    
    start_time = time.time()
    
    for batch_id, (data, target) in enumerate(tqdm(train_loader, desc=f'IBCapsNet Epoch {epoch}/{n_epochs}')):
        # 打印第一个batch的数据统计信息
        if batch_id == 0 and epoch == 1:
            print(f"\n[IBCapsNet] First batch data statistics:")
            print(f"  Data shape: {data.shape}")
            print(f"  Data min: {data.min().item():.6f}")
            print(f"  Data max: {data.max().item():.6f}")
            print(f"  Data mean: {data.mean().item():.6f}")
            print(f"  Data std: {data.std().item():.6f}")
            if logger:
                logger.info(f"[IBCapsNet] First batch - min: {data.min().item():.6f}, max: {data.max().item():.6f}, mean: {data.mean().item():.6f}, std: {data.std().item():.6f}")
        
        labels = F.one_hot(target, 10).float()
        data, target, labels = data.to(next(model.parameters()).device), target.to(next(model.parameters()).device), labels.to(next(model.parameters()).device)
        
        optimizer.zero_grad()
        probs, kl, recon_img = model(data, target=target, do_recon=True)
        loss, ce, kl_loss, recon = model.loss(probs, labels, kl, recon_img, data)
        
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        pred = probs.argmax(dim=1)
        correct = pred.eq(target).sum().item()
        total_correct += correct
        total_samples += target.size(0)
        
        total_loss += loss.item()
        total_ce += ce.item()
        total_kl += kl_loss.item()
        total_recon += recon.item() if recon.item() > 0 else 0
    
    avg_loss = total_loss / n_batch
    avg_acc = total_correct / total_samples
    elapsed_time = time.time() - start_time
    
    return avg_loss, avg_acc, elapsed_time, total_ce/n_batch, total_kl/n_batch, total_recon/n_batch


def test_capsnet(model, test_loader):
    """测试传统 CapsNet"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            target_onehot = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
            data, target_onehot = Variable(data), Variable(target_onehot)
            
            if USE_CUDA:
                data, target_onehot = data.cuda(), target_onehot.cuda()
            
            output, reconstructions, masked = model(data)
            pred = np.argmax(masked.data.cpu().numpy(), 1)
            correct += np.sum(pred == target.numpy())
            total += target.size(0)
    
    return correct / total


def test_ibcapsnet(model, test_loader):
    """测试 IBCapsNet"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(next(model.parameters()).device), target.to(next(model.parameters()).device)
            probs, _, _ = model(data, do_recon=False)
            pred = probs.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return correct / total


def train_lenet(model, optimizer, train_loader, epoch, n_epochs, logger=None):
    """训练 LeNet"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    n_batch = len(train_loader)
    
    criterion = nn.CrossEntropyLoss()
    start_time = time.time()
    
    for batch_id, (data, target) in enumerate(tqdm(train_loader, desc=f'LeNet Epoch {epoch}/{n_epochs}')):
        # 打印第一个batch的数据统计信息
        if batch_id == 0 and epoch == 1:
            print(f"\n[LeNet] First batch data statistics:")
            print(f"  Data shape: {data.shape}")
            print(f"  Data min: {data.min().item():.6f}")
            print(f"  Data max: {data.max().item():.6f}")
            print(f"  Data mean: {data.mean().item():.6f}")
            print(f"  Data std: {data.std().item():.6f}")
            if logger:
                logger.info(f"[LeNet] First batch - min: {data.min().item():.6f}, max: {data.max().item():.6f}, mean: {data.mean().item():.6f}, std: {data.std().item():.6f}")
        
        data, target = data.to(next(model.parameters()).device), target.to(next(model.parameters()).device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum().item()
        total_correct += correct
        total_samples += target.size(0)
        
        total_loss += loss.item()
    
    avg_loss = total_loss / n_batch
    avg_acc = total_correct / total_samples
    elapsed_time = time.time() - start_time
    
    return avg_loss, avg_acc, elapsed_time


def test_lenet(model, test_loader):
    """测试 LeNet"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(next(model.parameters()).device), target.to(next(model.parameters()).device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return correct / total


# ==============================
# 重构可视化函数
# ==============================
def visualize_reconstruction_samples(model, test_loader, model_name, epoch, result_dir, 
                                     num_samples=5, dataset_name='mnist'):
    """
    可视化模型的重构结果（5个样本）
    
    Args:
        model: 模型（CapsNet 或 IBCapsNet）
        test_loader: 测试数据加载器
        model_name: 模型名称
        epoch: 当前epoch
        result_dir: 结果保存目录
        num_samples: 要可视化的样本数量（默认5个）
        dataset_name: 数据集名称
    """
    model.eval()
    
    # 创建可视化目录
    vis_dir = os.path.join(result_dir, 'reconstruction_visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 获取测试样本
    samples_data = []
    samples_target = []
    
    for data, target in test_loader:
        samples_data.append(data)
        samples_target.append(target)
        if len(samples_data) * data.size(0) >= num_samples:
            break
    
    # 合并所有batch
    all_data = torch.cat(samples_data, dim=0)[:num_samples]
    all_target = torch.cat(samples_target, dim=0)[:num_samples]
    
    # 获取重构结果
    with torch.no_grad():
        if isinstance(model, CapsNet):
            # CapsNet
            target_onehot = torch.sparse.torch.eye(10).index_select(dim=0, index=all_target)
            data_var, target_onehot_var = Variable(all_data), Variable(target_onehot)
            
            if USE_CUDA:
                data_var, target_onehot_var = data_var.cuda(), target_onehot_var.cuda()
            
            output, reconstructions, masked = model(data_var)
            recon_images = reconstructions.cpu()
        elif hasattr(model, 'recon_alpha'):
            # IBCapsNet
            data_tensor = all_data.to(next(model.parameters()).device)
            target_tensor = all_target.to(next(model.parameters()).device)
            
            probs, kl, recon_img = model(data_tensor, target=target_tensor, do_recon=True)
            recon_images = recon_img.cpu()
        else:
            # 不支持重构的模型（如LeNet）
            return
    
    # 创建可视化图像
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle(f'{model_name} - Reconstruction at Epoch {epoch}', 
                 fontsize=14, fontweight='bold')
    
    # 第一行：原始图像
    for col_idx in range(num_samples):
        ax = axes[0, col_idx]
        ax.axis('off')
        if dataset_name in ['mnist', 'mnist-small', 'fashion-mnist']:
            img_orig = all_data[col_idx].squeeze(0).numpy()
            ax.imshow(img_orig, cmap='gray')
        else:
            img_orig = all_data[col_idx].permute(1, 2, 0).numpy()
            img_orig = np.clip(img_orig, 0, 1)
            ax.imshow(img_orig)
        ax.set_title(f'Original\nLabel: {all_target[col_idx].item()}', fontsize=10)
    
    # 第二行：重构图像
    for col_idx in range(num_samples):
        ax = axes[1, col_idx]
        ax.axis('off')
        if dataset_name in ['mnist', 'mnist-small', 'fashion-mnist']:
            img_recon = recon_images[col_idx].squeeze(0).numpy()
            img_recon = np.clip(img_recon, 0, 1)
            ax.imshow(img_recon, cmap='gray')
        else:
            img_recon = recon_images[col_idx].permute(1, 2, 0).numpy()
            img_recon = np.clip(img_recon, 0, 1)
            ax.imshow(img_recon)
        ax.set_title('Reconstruction', fontsize=10)
    
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(vis_dir, f'{model_name.lower()}_epoch_{epoch:03d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


# ==============================
# 辅助函数：训练单个模型
# ==============================
def train_single_model(model, optimizer, train_loader, test_loader, 
                       model_name, n_epochs, result_dir, logger,
                       train_func, test_func, dataset_name='mnist'):
    """
    训练单个模型的通用函数
    
    Args:
        model: 要训练的模型
        optimizer: 优化器
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        model_name: 模型名称（用于日志和保存）
        n_epochs: 训练轮数
        result_dir: 结果保存目录
        logger: 日志记录器
        train_func: 训练函数
        test_func: 测试函数
        dataset_name: 数据集名称（用于重构可视化）
    Returns:
        history: 训练历史字典
        best_acc: 最佳准确率
        best_epoch: 最佳准确率对应的epoch
    """
    history = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'time': []}
    if hasattr(model, 'recon_alpha'):  # IBCapsNet 有额外的损失项
        history.update({'ce_loss': [], 'kl_loss': [], 'recon_loss': []})
    elif isinstance(model, CapsNet):  # CapsNet 有 margin loss 和 reconstruction loss
        history.update({'margin_loss': [], 'recon_loss': []})
    
    best_acc = 0.0
    best_epoch = 0
    
    # 检查模型是否支持重构任务
    has_reconstruction = isinstance(model, CapsNet) or hasattr(model, 'recon_alpha')
    
    for epoch in range(1, n_epochs + 1):
        # 训练
        if hasattr(model, 'recon_alpha'):
            # IBCapsNet
            train_loss, train_acc, elapsed, ce, kl, recon = train_func(
                model, optimizer, train_loader, epoch, n_epochs, logger
            )
            history['ce_loss'].append(ce)
            history['kl_loss'].append(kl)
            history['recon_loss'].append(recon)
        elif isinstance(model, CapsNet):
            # CapsNet
            train_loss, train_acc, elapsed, margin_loss, recon_loss = train_func(
                model, optimizer, train_loader, epoch, n_epochs, logger
            )
            history['margin_loss'].append(margin_loss)
            history['recon_loss'].append(recon_loss)
        else:
            # LeNet 等其他模型
            train_loss, train_acc, elapsed = train_func(
                model, optimizer, train_loader, epoch, n_epochs, logger
            )
        
        # 测试
        test_acc = test_func(model, test_loader)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['time'].append(elapsed)
        
        # 日志
        if hasattr(model, 'recon_alpha'):
            logger.info(f"{model_name} Epoch {epoch}: Train Loss={train_loss:.4f}, "
                       f"Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}, "
                       f"Time={elapsed:.2f}s, CE={ce:.4f}, KL={kl:.4f}, Recon={recon:.6f}")
        elif isinstance(model, CapsNet):
            logger.info(f"{model_name} Epoch {epoch}: Train Loss={train_loss:.4f}, "
                       f"Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}, "
                       f"Time={elapsed:.2f}s, Margin={margin_loss:.4f}, Recon={recon_loss:.6f}")
        else:
            logger.info(f"{model_name} Epoch {epoch}: Train Loss={train_loss:.4f}, "
                       f"Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}, Time={elapsed:.2f}s")
        
        # 每隔10个epoch保存重构可视化（如果有重构任务）
        if has_reconstruction and epoch % 10 == 0:
            try:
                vis_path = visualize_reconstruction_samples(
                    model, test_loader, model_name, epoch, result_dir,
                    num_samples=5, dataset_name=dataset_name
                )
                logger.info(f"  -> Reconstruction visualization saved: {vis_path}")
            except Exception as e:
                logger.warning(f"  -> Failed to save reconstruction visualization: {e}")
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            model_path = os.path.join(result_dir, f'{model_name.lower()}_best.pth')
            torch.save(model.state_dict(), model_path)
            logger.info(f"  -> New best {model_name} model saved! Accuracy: {best_acc:.4f} (Epoch {epoch})")
    
    return history, best_acc, best_epoch


# ==============================
# 实验1: 准确率对比（支持多个模型和数据集）
# ==============================
def experiment_accuracy(logger, result_dir, dataset_name='mnist', n_epochs=30, include_lenet=True,
                         context_encoder_type='default'):
    """
    实验1: 准确率对比
    对比 CapsNet, IBCapsNet (linear), IBCapsNet (squash), IBCapsNet (inverse_squash), LeNet
    
    Args:
        logger: 日志记录器
        result_dir: 结果保存目录
        dataset_name: 数据集名称 ('mnist', 'fashion-mnist', 'svhn', 'cifar10')
        n_epochs: 训练轮数
        include_lenet: 是否包含LeNet对比
        context_encoder_type: Context encoder类型，'default' 或 'enhanced'
    """
    logger.info("=" * 60)
    logger.info(f"Experiment 1: Accuracy Comparison - Dataset: {dataset_name}")
    logger.info(f"Context Encoder Type: {context_encoder_type}")
    logger.info("=" * 60)
    
    # 根据数据集确定配置
    if dataset_name == 'mnist':
        config = Config('mnist')
        dataset = Dataset('mnist', 128)
        input_channels = 1
        input_size = 28
        num_classes = 10
    elif dataset_name == 'fashion-mnist':
        # FashionMNIST使用MNIST配置（都是28x28灰度图像，10个类别）
        config = Config('mnist')
        dataset = Dataset('fashion-mnist', 128)
        input_channels = 1
        input_size = 28
        num_classes = 10
    elif dataset_name == 'svhn':
        # SVHN使用CIFAR-10配置（都是32x32 RGB图像）
        config = Config('cifar10')
        dataset = Dataset('svhn', 128)
        input_channels = 3
        input_size = 32
        num_classes = 10
    elif dataset_name == 'cifar10':
        config = Config('cifar10')
        dataset = Dataset('cifar10', 128)
        input_channels = 3
        input_size = 32
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported: 'mnist', 'fashion-mnist', 'svhn', 'cifar10'")
    
    logger.info(f"Dataset: {dataset_name}, Input channels: {input_channels}, Input size: {input_size}x{input_size}")
    
    results = {}
    all_models = {}
    all_histories = {}
    all_best_accs = {}
    all_best_epochs = {}
    
    # ==============================
    # 1. 训练 CapsNet
    # ==============================
    logger.info("\n" + "=" * 60)
    logger.info("Training CapsNet...")
    logger.info("=" * 60)
    capsnet = CapsNet(config)
    if USE_CUDA:
        capsnet = capsnet.cuda()
    optimizer_caps = torch.optim.Adam(capsnet.parameters(), lr=0.01)
    
    capsnet_history, best_capsnet_acc, best_capsnet_epoch = train_single_model(
        capsnet, optimizer_caps, dataset.train_loader, dataset.test_loader,
        'CapsNet', n_epochs, result_dir, logger, train_capsnet, test_capsnet,
        dataset_name=dataset_name
    )
    all_models['CapsNet'] = capsnet
    all_histories['CapsNet'] = capsnet_history
    all_best_accs['CapsNet'] = best_capsnet_acc
    all_best_epochs['CapsNet'] = best_capsnet_epoch
    
    # ==============================
    # 2. 训练 IBCapsNet (Linear Classifier)
    # ==============================
    logger.info("\n" + "=" * 60)
    logger.info("Training IBCapsNet (Linear Classifier)...")
    logger.info("=" * 60)
    ibcapsnet_linear = IBCapsNetWithRecon(
        latent_dim=16, beta=1e-3, recon_alpha=0.0005, classifier_type='linear',
        input_width=input_size, input_height=input_size, input_channel=input_channels,
        context_encoder_type=context_encoder_type
    )
    if USE_CUDA:
        ibcapsnet_linear = ibcapsnet_linear.cuda()
    optimizer_ib_linear = torch.optim.Adam(ibcapsnet_linear.parameters(), lr=0.001)
    
    ibcapsnet_linear_history, best_ibcapsnet_linear_acc, best_ibcapsnet_linear_epoch = train_single_model(
        ibcapsnet_linear, optimizer_ib_linear, dataset.train_loader, dataset.test_loader,
        'IBCapsNet-Linear', n_epochs, result_dir, logger, train_ibcapsnet, test_ibcapsnet,
        dataset_name=dataset_name
    )
    all_models['IBCapsNet-Linear'] = ibcapsnet_linear
    all_histories['IBCapsNet-Linear'] = ibcapsnet_linear_history
    all_best_accs['IBCapsNet-Linear'] = best_ibcapsnet_linear_acc
    all_best_epochs['IBCapsNet-Linear'] = best_ibcapsnet_linear_epoch

    # ==============================
    # 3. 训练 IBCapsNet (Squash Classifier)
    # ==============================
    logger.info("\n" + "=" * 60)
    logger.info("Training IBCapsNet (Squash Classifier)...")
    logger.info("=" * 60)
    ibcapsnet_squash = IBCapsNetWithRecon(
        latent_dim=16, beta=1e-3, recon_alpha=0.0005, classifier_type='squash',
        input_width=input_size, input_height=input_size, input_channel=input_channels,
        context_encoder_type=context_encoder_type
    )
    if USE_CUDA:
        ibcapsnet_squash = ibcapsnet_squash.cuda()
    optimizer_ib_squash = torch.optim.Adam(ibcapsnet_squash.parameters(), lr=0.001)
    
    ibcapsnet_squash_history, best_ibcapsnet_squash_acc, best_ibcapsnet_squash_epoch = train_single_model(
        ibcapsnet_squash, optimizer_ib_squash, dataset.train_loader, dataset.test_loader,
        'IBCapsNet-Squash', n_epochs, result_dir, logger, train_ibcapsnet, test_ibcapsnet,
        dataset_name=dataset_name
    )
    all_models['IBCapsNet-Squash'] = ibcapsnet_squash
    all_histories['IBCapsNet-Squash'] = ibcapsnet_squash_history
    all_best_accs['IBCapsNet-Squash'] = best_ibcapsnet_squash_acc
    all_best_epochs['IBCapsNet-Squash'] = best_ibcapsnet_squash_epoch

    # ==============================
    # 4. 训练 IBCapsNet (Inverse Squash Classifier)
    # ==============================
    # logger.info("\n" + "=" * 60)
    # logger.info("Training IBCapsNet (Inverse Squash Classifier)...")
    # logger.info("=" * 60)
    # ibcapsnet_inverse_squash = IBCapsNetWithRecon(
    #     latent_dim=16, beta=1e-3, recon_alpha=0.0005, classifier_type='inverse_squash',
    #     input_width=input_size, input_height=input_size, input_channel=input_channels,
    #     context_encoder_type=context_encoder_type
    # )
    # if USE_CUDA:
    #     ibcapsnet_inverse_squash = ibcapsnet_inverse_squash.cuda()
    # optimizer_ib_inverse_squash = torch.optim.Adam(ibcapsnet_inverse_squash.parameters(), lr=0.001)
    
    # ibcapsnet_inverse_squash_history, best_ibcapsnet_inverse_squash_acc, best_ibcapsnet_inverse_squash_epoch = train_single_model(
    #     ibcapsnet_inverse_squash, optimizer_ib_inverse_squash, dataset.train_loader, dataset.test_loader,
    #     'IBCapsNet-Inverse_Squash', n_epochs, result_dir, logger, train_ibcapsnet, test_ibcapsnet,
    #     dataset_name=dataset_name
    # )
    # all_models['IBCapsNet-Inverse_Squash'] = ibcapsnet_inverse_squash
    # all_histories['IBCapsNet-Inverse_Squash'] = ibcapsnet_inverse_squash_history
    # all_best_accs['IBCapsNet-Inverse_Squash'] = best_ibcapsnet_inverse_squash_acc
    # all_best_epochs['IBCapsNet-Inverse_Squash'] = best_ibcapsnet_inverse_squash_epoch

    # ==============================
    # 5. 训练 LeNet（如果启用）
    # ==============================
    if include_lenet:
        logger.info("\n" + "=" * 60)
        logger.info("Training LeNet...")
        logger.info("=" * 60)
        lenet = LeNetMultiChannel(num_classes=num_classes, in_channels=input_channels, input_size=input_size)
        if USE_CUDA:
            lenet = lenet.cuda()
        optimizer_lenet = torch.optim.Adam(lenet.parameters(), lr=0.01)
        
        lenet_history, best_lenet_acc, best_lenet_epoch = train_single_model(
            lenet, optimizer_lenet, dataset.train_loader, dataset.test_loader,
            'LeNet', n_epochs, result_dir, logger, train_lenet, test_lenet,
            dataset_name=dataset_name
        )
        all_models['LeNet'] = lenet
        all_histories['LeNet'] = lenet_history
        all_best_accs['LeNet'] = best_lenet_acc
        all_best_epochs['LeNet'] = best_lenet_epoch
    
    # ==============================
    # 6. 对比结果并保存
    # ==============================
    logger.info("\n" + "=" * 60)
    logger.info("Best Model Comparison")
    logger.info("=" * 60)
    for model_name in all_best_accs.keys():
        logger.info(f"{model_name} Best Accuracy: {all_best_accs[model_name]:.4f} "
                   f"(Epoch {all_best_epochs[model_name]})")
    
    # 确定最佳模型
    best_model_name = max(all_best_accs, key=all_best_accs.get)
    best_model_acc = all_best_accs[best_model_name]
    best_model_epoch = all_best_epochs[best_model_name]
    
    # 保存最佳模型
    overall_best_path = os.path.join(result_dir, 'overall_best_model.pth')
    torch.save(all_models[best_model_name].state_dict(), overall_best_path)
    logger.info(f"\nOverall Best Model: {best_model_name} with accuracy {best_model_acc:.4f} "
               f"(Epoch {best_model_epoch})")
    logger.info(f"Best model saved to: {overall_best_path}")
    
    # 保存结果
    results = {
        'dataset': dataset_name,
        'capsnet': capsnet_history,
        'ibcapsnet_linear': ibcapsnet_linear_history,
        'ibcapsnet_squash': ibcapsnet_squash_history,
        # 'ibcapsnet_inverse_squash': ibcapsnet_inverse_squash_history,
        'final_accuracies': {
            'CapsNet': capsnet_history['test_acc'][-1],
            'IBCapsNet-Linear': ibcapsnet_linear_history['test_acc'][-1],
            'IBCapsNet-Squash': ibcapsnet_squash_history['test_acc'][-1],
            # 'IBCapsNet-Inverse_Squash': ibcapsnet_inverse_squash_history['test_acc'][-1]
        },
        'best_accuracies': all_best_accs,
        'best_epochs': all_best_epochs,
        'best_model_name': best_model_name,
        'best_model_accuracy': best_model_acc,
        'best_model_epoch': best_model_epoch,
        'improvements': {
            'IBCapsNet-Linear vs CapsNet': best_ibcapsnet_linear_acc - best_capsnet_acc,
            'IBCapsNet-Squash vs CapsNet': best_ibcapsnet_squash_acc - best_capsnet_acc,
            # 'IBCapsNet-Inverse_Squash vs CapsNet': best_ibcapsnet_inverse_squash_acc - best_capsnet_acc,
            'IBCapsNet-Squash vs IBCapsNet-Linear': best_ibcapsnet_squash_acc - best_ibcapsnet_linear_acc,
            # 'IBCapsNet-Inverse_Squash vs IBCapsNet-Linear': best_ibcapsnet_inverse_squash_acc - best_ibcapsnet_linear_acc,
            # 'IBCapsNet-Inverse_Squash vs IBCapsNet-Squash': best_ibcapsnet_inverse_squash_acc - best_ibcapsnet_squash_acc
        },
        'model_files': {
            'capsnet_best': 'capsnet_best.pth',
            'ibcapsnet_linear_best': 'ibcapsnet-linear_best.pth',
            'ibcapsnet_squash_best': 'ibcapsnet-squash_best.pth',
            # 'ibcapsnet_inverse_squash_best': 'ibcapsnet-inverse_squash_best.pth',
            'overall_best': 'overall_best_model.pth'
        }
    }
    
    # 如果包含LeNet，添加LeNet的结果
    if include_lenet:
        results['lenet'] = lenet_history
        results['final_accuracies']['LeNet'] = lenet_history['test_acc'][-1]
        results['improvements']['LeNet vs CapsNet'] = best_lenet_acc - best_capsnet_acc
        results['improvements']['IBCapsNet-Linear vs LeNet'] = best_ibcapsnet_linear_acc - best_lenet_acc
        results['improvements']['IBCapsNet-Squash vs LeNet'] = best_ibcapsnet_squash_acc - best_lenet_acc
        # results['improvements']['IBCapsNet-Inverse_Squash vs LeNet'] = best_ibcapsnet_inverse_squash_acc - best_lenet_acc
        results['model_files']['lenet_best'] = 'lenet_best.pth'
    
    with open(os.path.join(result_dir, 'experiment_accuracy.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # 打印摘要
    logger.info("\n" + "=" * 60)
    logger.info("Summary:")
    logger.info("=" * 60)
    logger.info(f"  Final Accuracies:")
    for model_name, acc in results['final_accuracies'].items():
        logger.info(f"    {model_name}: {acc:.4f}")
    logger.info(f"\n  Best Accuracies:")
    for model_name, acc in results['best_accuracies'].items():
        logger.info(f"    {model_name}: {acc:.4f} (Epoch {all_best_epochs[model_name]})")
    logger.info(f"\n  Improvements:")
    for comparison, improvement in results['improvements'].items():
        logger.info(f"    {comparison}: {improvement:+.4f}")
    logger.info(f"\n  Best Overall Model: {best_model_name} with accuracy {best_model_acc:.4f}")
    logger.info("=" * 60)
    
    return results


# ==============================
# 实验2: 训练速度对比
# ==============================
def experiment_training_speed(logger, result_dir):
    """实验2: 训练速度对比"""
    logger.info("=" * 60)
    logger.info("Experiment 2: Training Speed Comparison")
    logger.info("=" * 60)
    
    config = Config('mnist')
    dataset = Dataset('mnist', 128)
    
    # CapsNet
    logger.info("Measuring CapsNet training speed...")
    capsnet = CapsNet(config)
    if USE_CUDA:
        capsnet = capsnet.cuda()
    optimizer_caps = torch.optim.Adam(capsnet.parameters(), lr=0.01)
    
    capsnet_times = []
    for epoch in range(1, 6):
        _, _, elapsed, _, _ = train_capsnet(capsnet, optimizer_caps, dataset.train_loader, epoch, 5, logger)
        capsnet_times.append(elapsed)
        logger.info(f"CapsNet Epoch {epoch}: {elapsed:.2f}s")
    
    # IBCapsNet
    logger.info("Measuring IBCapsNet training speed...")
    ibcapsnet = IBCapsNetWithRecon(latent_dim=16, beta=1e-3, recon_alpha=0.0005)
    if USE_CUDA:
        ibcapsnet = ibcapsnet.cuda()
    optimizer_ib = torch.optim.Adam(ibcapsnet.parameters(), lr=0.001)
    
    ibcapsnet_times = []
    for epoch in range(1, 6):
        _, _, elapsed, _, _, _ = train_ibcapsnet(ibcapsnet, optimizer_ib, dataset.train_loader, epoch, 5, logger)
        ibcapsnet_times.append(elapsed)
        logger.info(f"IBCapsNet Epoch {epoch}: {elapsed:.2f}s")
    
    results = {
        'capsnet_avg_time': np.mean(capsnet_times),
        'ibcapsnet_avg_time': np.mean(ibcapsnet_times),
        'speedup': np.mean(capsnet_times) / np.mean(ibcapsnet_times),
        'capsnet_times': capsnet_times,
        'ibcapsnet_times': ibcapsnet_times
    }
    
    with open(os.path.join(result_dir, 'experiment_speed.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"CapsNet Average Time: {results['capsnet_avg_time']:.2f}s")
    logger.info(f"IBCapsNet Average Time: {results['ibcapsnet_avg_time']:.2f}s")
    logger.info(f"Speedup: {results['speedup']:.2f}x")
    
    return results


# ==============================
# 实验3: 小样本学习
# ==============================
def experiment_few_shot(logger, result_dir):
    """实验3: 小样本学习能力对比"""
    logger.info("=" * 60)
    logger.info("Experiment 3: Few-Shot Learning")
    logger.info("=" * 60)
    
    results = {}
    
    for samples_per_class in [100, 200, 500]:
        logger.info(f"Testing with {samples_per_class} samples per class...")
        config = Config('mnist-small')
        dataset = Dataset('mnist-small', 64, samples_per_class=samples_per_class)
        
        # CapsNet
        capsnet = CapsNet(config)
        if USE_CUDA:
            capsnet = capsnet.cuda()
        optimizer_caps = torch.optim.Adam(capsnet.parameters(), lr=0.01)
        
        for epoch in range(1, 31):
            train_capsnet(capsnet, optimizer_caps, dataset.train_loader, epoch, 30, logger)
        capsnet_acc = test_capsnet(capsnet, dataset.test_loader)
        
        # IBCapsNet
        ibcapsnet = IBCapsNetWithRecon(latent_dim=16, beta=1e-3, recon_alpha=0.0005)
        if USE_CUDA:
            ibcapsnet = ibcapsnet.cuda()
        optimizer_ib = torch.optim.Adam(ibcapsnet.parameters(), lr=0.001)
        
        for epoch in range(1, 31):
            train_ibcapsnet(ibcapsnet, optimizer_ib, dataset.train_loader, epoch, 30, logger)
        ibcapsnet_acc = test_ibcapsnet(ibcapsnet, dataset.test_loader)
        
        results[f'{samples_per_class}'] = {
            'capsnet': capsnet_acc,
            'ibcapsnet': ibcapsnet_acc,
            'improvement': ibcapsnet_acc - capsnet_acc
        }
        logger.info(f"Samples={samples_per_class}: CapsNet={capsnet_acc:.4f}, IBCapsNet={ibcapsnet_acc:.4f}, Improvement={ibcapsnet_acc - capsnet_acc:.4f}")
    
    with open(os.path.join(result_dir, 'experiment_few_shot.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


# ==============================
# 实验4: 参数效率对比
# ==============================
def experiment_parameter_efficiency(logger, result_dir):
    """实验4: 参数效率对比"""
    logger.info("=" * 60)
    logger.info("Experiment 4: Parameter Efficiency")
    logger.info("=" * 60)
    
    config = Config('mnist')
    
    # CapsNet
    capsnet = CapsNet(config)
    capsnet_params = sum(p.numel() for p in capsnet.parameters())
    capsnet_trainable = sum(p.numel() for p in capsnet.parameters() if p.requires_grad)
    
    # IBCapsNet
    ibcapsnet = IBCapsNetWithRecon(latent_dim=16, beta=1e-3, recon_alpha=0.0005)
    ibcapsnet_params = sum(p.numel() for p in ibcapsnet.parameters())
    ibcapsnet_trainable = sum(p.numel() for p in ibcapsnet.parameters() if p.requires_grad)
    
    results = {
        'capsnet': {
            'total_params': capsnet_params,
            'trainable_params': capsnet_trainable
        },
        'ibcapsnet': {
            'total_params': ibcapsnet_params,
            'trainable_params': ibcapsnet_trainable
        },
        'ratio': ibcapsnet_params / capsnet_params
    }
    
    with open(os.path.join(result_dir, 'experiment_params.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"CapsNet Parameters: {capsnet_params:,}")
    logger.info(f"IBCapsNet Parameters: {ibcapsnet_params:,}")
    logger.info(f"Parameter Ratio: {results['ratio']:.2f}x")
    
    return results


# ==============================
# 主函数
# ==============================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='IBCapsNet vs CapsNet Comparison Experiments')
    parser.add_argument('--dataset', type=str, default='mnist', 
                       choices=['mnist', 'fashion-mnist', 'svhn', 'cifar10'],
                       help='Dataset to use (default: mnist)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs (default: 30)')
    parser.add_argument('--no-lenet', action='store_true',
                       help='Exclude LeNet from comparison')
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['all', 'accuracy', 'speed', 'few_shot', 'params'],
                       help='Which experiment to run (default: all)')
    parser.add_argument('--context-encoder-type', type=str, default='default',
                       choices=['default', 'enhanced'],
                       help='Context encoder type for IBCapsNet: "default" (simple) or "enhanced" (with attention) (default: default)')
    
    args = parser.parse_args()
    
    torch.manual_seed(42)
    if USE_CUDA:
        torch.cuda.manual_seed(42)
    
    # 创建结果目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = f'comparison_results_{args.dataset}_{timestamp}'
    os.makedirs(result_dir, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(result_dir, 'comparison_experiment.log')
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
    logger.info("IBCapsNet vs CapsNet vs LeNet Comparison Experiments")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Include LeNet: {not args.no_lenet}")
    logger.info(f"Context Encoder Type: {args.context_encoder_type}")
    logger.info(f"CUDA Available: {USE_CUDA}")
    logger.info(f"Result Directory: {result_dir}")
    logger.info("=" * 60)
    
    all_results = {}
    
    # 运行实验
    try:
        if args.experiment in ['all', 'accuracy']:
            logger.info("\nRunning Experiment 1: Accuracy Comparison...")
            all_results['accuracy'] = experiment_accuracy(
                logger, result_dir, dataset_name=args.dataset, 
                n_epochs=args.epochs, include_lenet=not args.no_lenet,
                context_encoder_type=args.context_encoder_type
            )
        
        if args.experiment in ['all', 'speed']:
            logger.info("\nRunning Experiment 2: Training Speed Comparison...")
            all_results['speed'] = experiment_training_speed(logger, result_dir)
        
        if args.experiment in ['all', 'few_shot']:
            logger.info("\nRunning Experiment 3: Few-Shot Learning...")
            all_results['few_shot'] = experiment_few_shot(logger, result_dir)
        
        if args.experiment in ['all', 'params']:
            logger.info("\nRunning Experiment 4: Parameter Efficiency...")
            all_results['params'] = experiment_parameter_efficiency(logger, result_dir)
        
    except Exception as e:
        logger.error(f"Error during experiments: {e}", exc_info=True)
    
    # 保存所有结果
    summary = {
        'timestamp': timestamp,
        'dataset': args.dataset,
        'epochs': args.epochs,
        'include_lenet': not args.no_lenet,
        'context_encoder_type': args.context_encoder_type,
        'experiments': all_results,
        'summary': {
            'best_accuracies': all_results.get('accuracy', {}).get('best_accuracies', {}),
            'best_model': all_results.get('accuracy', {}).get('best_model_name', 'N/A'),
            'speedup': all_results.get('speed', {}).get('speedup', 0),
            'parameter_ratio': all_results.get('params', {}).get('ratio', 0)
        }
    }
    
    with open(os.path.join(result_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("All experiments completed!")
    logger.info(f"Results saved to: {result_dir}")
    logger.info("=" * 60)

