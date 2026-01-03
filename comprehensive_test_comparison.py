"""
模型训练完成后的全面测试对比脚本
对比 CapsNet 和 IBCapsNet 在多个维度的性能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import time
import json
import os
import argparse
import glob
from datetime import datetime
from tqdm import tqdm
import logging
from capsnet import CapsNet
from data_loader import Dataset
from test_capsnet import Config
from comparison_experiment import IBCapsNetWithRecon, LeNetMultiChannel
from train_lenet import LeNet

USE_CUDA = True if torch.cuda.is_available() else False


# ==============================
# 辅助函数：高斯模糊
# ==============================
def get_gaussian_kernel(kernel_size=5, sigma=1.0):
    """生成高斯核"""
    coords = torch.arange(kernel_size, dtype=torch.float32)
    coords -= kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.view(1, 1, -1, 1) * g.view(1, 1, 1, -1)


def apply_gaussian_blur(x, sigma=1.0, kernel_size=None):
    """
    对输入图像应用高斯模糊
    
    Args:
        x: [B, C, H, W] 输入图像
        sigma: 高斯核的标准差
        kernel_size: 核大小（如果为None，自动计算）
    Returns:
        blurred: [B, C, H, W] 模糊后的图像
    """
    if sigma <= 0:
        return x
    
    # 自动计算核大小（覆盖3个标准差）
    if kernel_size is None:
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
    
    # 生成高斯核
    kernel = get_gaussian_kernel(kernel_size, sigma)
    kernel = kernel.repeat(x.shape[1], 1, 1, 1)  # [C, 1, kernel_size, kernel_size]
    
    # 移动到正确的设备
    device = x.device
    kernel = kernel.to(device)
    
    # 应用卷积（使用padding保持尺寸）
    padding = kernel_size // 2
    blurred = F.conv2d(x, kernel, padding=padding, groups=x.shape[1])
    
    return blurred


# ==============================
# 辅助函数：模型前向传播
# ==============================
def forward_model(model, data, target=None, do_recon=False):
    """
    统一的模型前向传播函数，支持 CapsNet、IBCapsNet 和 LeNet
    
    Returns:
        probs: 概率分布 [B, 10]
        pred: 预测类别 [B]
        loss: 损失值（标量）
    """
    if isinstance(model, CapsNet):
        # CapsNet
        data = Variable(data)
        if USE_CUDA:
            data = data.cuda()
        
        output, reconstructions, masked = model(data)
        pred = np.argmax(masked.data.cpu().numpy(), 1)
        masked_tensor = masked.data.cpu().float()
        probs = F.softmax(masked_tensor, dim=1).numpy()
        
        if target is not None:
            target_onehot = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
            target_onehot = Variable(target_onehot)
            if USE_CUDA:
                target_onehot = target_onehot.cuda()
            loss = model.loss(data, output, target_onehot, reconstructions)
        else:
            loss = torch.tensor(0.0)
        
        return probs, pred, loss
    
    elif isinstance(model, (LeNet, LeNetMultiChannel)):
        # LeNet or LeNetMultiChannel
        data = data.to(next(model.parameters()).device)
        target = target.to(next(model.parameters()).device) if target is not None else None
        
        output = model(data)  # [B, 10] logits
        probs = F.softmax(output, dim=1).cpu().numpy()
        pred = np.argmax(probs, axis=1)
        
        if target is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target)
        else:
            loss = torch.tensor(0.0)
        
        return probs, pred, loss
    
    else:
        # IBCapsNet
        data = data.to(next(model.parameters()).device)
        target = target.to(next(model.parameters()).device) if target is not None else None
        
        probs, kl, recon_img = model(data, do_recon=do_recon)
        pred = probs.argmax(dim=1).cpu().numpy()
        probs_np = probs.cpu().numpy()
        
        if target is not None:
            labels = F.one_hot(target, 10).float()
            loss, ce, kl_loss, recon = model.loss(probs, labels, kl, recon_img, data)
        else:
            loss = torch.tensor(0.0)
        
        return probs_np, pred, loss


# ==============================
# 1. 基础性能指标对比
# ==============================
def test_basic_metrics(model, test_loader, model_name="Model", logger=None):
    """测试基础性能指标：准确率、损失、各类别准确率"""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    if logger:
        logger.info(f"Starting basic metrics test for {model_name}...")
        logger.info(f"  Test dataset size: {len(test_loader.dataset)}")
        logger.info(f"  Number of batches: {len(test_loader)}")
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc=f'Testing {model_name}'):
            # 保存原始 target 用于后续统计（确保在 CPU 上）
            target_cpu = target.cpu() if isinstance(target, torch.Tensor) else target
            target_np = target_cpu.numpy() if isinstance(target_cpu, torch.Tensor) else target_cpu
        
            # 使用统一的模型前向传播函数
            probs, pred, loss = forward_model(model, data, target, do_recon=False)
            
            all_preds.extend(pred)
            all_targets.extend(target_np)
            all_probs.extend(probs)
            total_loss += loss.item()
            correct += np.sum(pred == target_np)
            total += len(target_np)
            
            # 各类别统计
            for i in range(len(target_np)):
                label = target[i].item()
                class_total[label] += 1
                if pred[i] == label:
                    class_correct[label] += 1
    
    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(test_loader)
    class_accuracies = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0 
                        for i in range(10)]
    
    # 详细记录到日志
    if logger:
        logger.info(f"\n{model_name} - Basic Metrics Results:")
        logger.info(f"  Overall Accuracy: {accuracy:.6f} ({correct}/{total})")
        logger.info(f"  Average Loss: {avg_loss:.6f}")
        logger.info(f"  Per-Class Accuracies:")
        for i, acc in enumerate(class_accuracies):
            logger.info(f"    Class {i}: {acc:.6f} ({class_correct[i]}/{class_total[i]})")
        logger.info(f"  Prediction confidence statistics:")
        all_probs_array = np.array(all_probs)
        logger.info(f"    Mean confidence: {all_probs_array.max(axis=1).mean():.6f}")
        logger.info(f"    Std confidence: {all_probs_array.max(axis=1).std():.6f}")
        logger.info(f"    Min confidence: {all_probs_array.max(axis=1).min():.6f}")
        logger.info(f"    Max confidence: {all_probs_array.max(axis=1).max():.6f}")
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'class_accuracies': class_accuracies,
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs
    }


# ==============================
# 2. 混淆矩阵和分类报告
# ==============================
def generate_confusion_matrix(predictions, targets, model_name, save_dir):
    """生成混淆矩阵"""
    cm = confusion_matrix(targets, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_{model_name}.png'))
    plt.close()
    
    return cm


def generate_classification_report(predictions, targets, model_name, save_dir):
    """生成分类报告"""
    report = classification_report(targets, predictions, 
                                   target_names=[str(i) for i in range(10)],
                                   output_dict=True)
    
    # 保存为JSON
    with open(os.path.join(save_dir, f'classification_report_{model_name}.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    # 保存为文本
    report_str = classification_report(targets, predictions, 
                                       target_names=[str(i) for i in range(10)])
    with open(os.path.join(save_dir, f'classification_report_{model_name}.txt'), 'w') as f:
        f.write(f"Classification Report - {model_name}\n")
        f.write("=" * 60 + "\n")
        f.write(report_str)
    
    return report


# ==============================
# 3. 鲁棒性测试
# ==============================
def test_robustness_noise(model, test_loader, noise_levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
                         model_name="Model", noise_type='additive', logger=None):
    """
    测试对噪声的鲁棒性
    
    Args:
        noise_type: 噪声类型
            - 'additive': 加性噪声，data + noise（可能超出[0,1]范围）
            - 'clamped': 加性噪声后clamp到[0,1]（产生饱和效应）
            - 'multiplicative': 乘性噪声，data * (1 + noise)
            - 'salt_pepper': 椒盐噪声
            - 'gaussian_blur': 高斯模糊（模拟低质量图像）
        logger: 日志记录器
    """
    model.eval()
    results = {}
    
    if logger:
        logger.info(f"Testing {model_name} robustness to {noise_type} noise...")
        logger.info(f"  Noise levels: {noise_levels}")
        logger.info(f"  Test dataset size: {len(test_loader.dataset)}")
        logger.info(f"  Number of batches: {len(test_loader)}")
    
    for noise_level in noise_levels:
        correct = 0
        total = 0
        batch_accuracies = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                target_cpu = target.cpu() if isinstance(target, torch.Tensor) else target
                target_np = target_cpu.numpy() if isinstance(target_cpu, torch.Tensor) else target_cpu
                
                # 根据噪声类型添加噪声
                if noise_level == 0.0:
                    noisy_data = data.clone()
                elif noise_type == 'additive':
                    # 方式1：加性噪声（不加clamp）
                    noise = torch.randn_like(data) * noise_level
                    noisy_data = data + noise
                elif noise_type == 'clamped':
                    # 方式2：加性噪声后clamp到[0,1]（产生饱和效应）
                    noise = torch.randn_like(data) * noise_level
                    noisy_data = torch.clamp(data + noise, 0, 1)
                elif noise_type == 'multiplicative':
                    # 方式3：乘性噪声（对高值区域影响更大）
                    noise = torch.randn_like(data) * noise_level
                    noisy_data = data * (1 + noise)
                    noisy_data = torch.clamp(noisy_data, 0, 1)
                elif noise_type == 'salt_pepper':
                    # 方式4：椒盐噪声（随机像素变为0或1）
                    noisy_data = data.clone()
                    salt_mask = torch.rand_like(data) < noise_level / 2
                    pepper_mask = torch.rand_like(data) < noise_level / 2
                    noisy_data[salt_mask] = 1.0
                    noisy_data[pepper_mask] = 0.0
                elif noise_type == 'gaussian_blur':
                    # 方式5：高斯模糊（模拟低质量图像）
                    noisy_data = apply_gaussian_blur(data, sigma=noise_level * 2)
                else:
                    raise ValueError(f"Unknown noise_type: {noise_type}")
                
                # 使用统一的模型前向传播函数
                _, pred, _ = forward_model(model, noisy_data, target, do_recon=False)
                
                batch_correct = np.sum(pred == target_np)
                batch_total = len(target_np)
                batch_acc = batch_correct / batch_total if batch_total > 0 else 0.0
                batch_accuracies.append(batch_acc)
                
                correct += batch_correct
                total += batch_total
        
        accuracy = correct / total if total > 0 else 0.0
        results[noise_level] = accuracy
        
        # 详细记录到日志
        if logger:
            logger.info(f"  {model_name} - {noise_type} Noise Level {noise_level}:")
            logger.info(f"    Overall Accuracy: {accuracy:.6f} ({correct}/{total})")
            logger.info(f"    Batch Accuracy - Mean: {np.mean(batch_accuracies):.6f}, "
                       f"Std: {np.std(batch_accuracies):.6f}, "
                       f"Min: {np.min(batch_accuracies):.6f}, "
                       f"Max: {np.max(batch_accuracies):.6f}")
        print(f"{model_name} - {noise_type} Noise Level {noise_level}: Accuracy = {accuracy:.4f}")
    
    # 记录总结
    if logger:
        logger.info(f"\n{model_name} - {noise_type} Noise Robustness Summary:")
        for noise_level in noise_levels:
            logger.info(f"  Noise Level {noise_level}: {results[noise_level]:.6f}")
        if len(noise_levels) > 1:
            accuracies = [results[nl] for nl in noise_levels if nl > 0]
            if accuracies:
                logger.info(f"  Average accuracy (noise > 0): {np.mean(accuracies):.6f}")
                logger.info(f"  Accuracy degradation: {results[0.0] - np.mean(accuracies):.6f}")
    
    return results


def test_robustness_rotation(model, test_loader, angles=[5, 10, 15, 20, 30], model_name="Model", logger=None):
    """测试对旋转的鲁棒性"""
    model.eval()
    results = {}
    
    if logger:
        logger.info(f"Testing {model_name} robustness to rotation...")
        logger.info(f"  Rotation angles: {angles}")
    
    for angle in angles:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=angle),
            transforms.ToTensor()
        ])
        
        correct = 0
        total = 0
        batch_accuracies = []
        
        with torch.no_grad():
            for data, target in test_loader:
                target_cpu = target.cpu() if isinstance(target, torch.Tensor) else target
                target_np = target_cpu.numpy() if isinstance(target_cpu, torch.Tensor) else target_cpu
                rotated_data = torch.stack([transform(data[i]) for i in range(data.size(0))])
                
                # 使用统一的模型前向传播函数
                _, pred, _ = forward_model(model, rotated_data, target, do_recon=False)
                
                batch_correct = np.sum(pred == target_np)
                batch_total = len(target_np)
                batch_acc = batch_correct / batch_total if batch_total > 0 else 0.0
                batch_accuracies.append(batch_acc)
                
                correct += batch_correct
                total += batch_total
                accuracy = correct / total if total > 0 else 0.0
        
        results[angle] = accuracy
        
        if logger:
            logger.info(f"  {model_name} - Rotation {angle}°: Accuracy = {accuracy:.6f} ({correct}/{total})")
            logger.info(f"    Batch Accuracy - Mean: {np.mean(batch_accuracies):.6f}, "
                       f"Std: {np.std(batch_accuracies):.6f}")
        print(f"{model_name} - Rotation {angle}°: Accuracy = {accuracy:.4f}")
    
    if logger:
        logger.info(f"\n{model_name} - Rotation Robustness Summary:")
        for angle in angles:
            logger.info(f"  Angle {angle}°: {results[angle]:.6f}")
    
    return results


def test_robustness_occlusion(model, test_loader, occlusion_sizes=[4, 6, 8, 10], model_name="Model", logger=None):
    """测试对遮挡的鲁棒性"""
    model.eval()
    results = {}
    
    if logger:
        logger.info(f"Testing {model_name} robustness to occlusion...")
        logger.info(f"  Occlusion sizes: {occlusion_sizes}")
    
    for size in occlusion_sizes:
        correct = 0
        total = 0
        batch_accuracies = []
        
        with torch.no_grad():
            for data, target in test_loader:
                target_cpu = target.cpu() if isinstance(target, torch.Tensor) else target
                target_np = target_cpu.numpy() if isinstance(target_cpu, torch.Tensor) else target_cpu
                occluded_data = data.clone()
                # 在图像中心添加遮挡
                h, w = data.shape[2], data.shape[3]
                start_h, start_w = h // 2 - size // 2, w // 2 - size // 2
                occluded_data[:, :, start_h:start_h+size, start_w:start_w+size] = 0
                
                # 使用统一的模型前向传播函数
                _, pred, _ = forward_model(model, occluded_data, target, do_recon=False)
                
                batch_correct = np.sum(pred == target_np)
                batch_total = len(target_np)
                batch_acc = batch_correct / batch_total if batch_total > 0 else 0.0
                batch_accuracies.append(batch_acc)
                
                correct += batch_correct
                total += batch_total
                accuracy = correct / total if total > 0 else 0.0
        
        results[size] = accuracy
        
        if logger:
            logger.info(f"  {model_name} - Occlusion {size}x{size}: Accuracy = {accuracy:.6f} ({correct}/{total})")
            logger.info(f"    Batch Accuracy - Mean: {np.mean(batch_accuracies):.6f}, "
                       f"Std: {np.std(batch_accuracies):.6f}")
        print(f"{model_name} - Occlusion {size}x{size}: Accuracy = {accuracy:.4f}")
    
    if logger:
        logger.info(f"\n{model_name} - Occlusion Robustness Summary:")
        for size in occlusion_sizes:
            logger.info(f"  Occlusion {size}x{size}: {results[size]:.6f}")
    
    return results


# ==============================
# 4. 推理速度对比
# ==============================
def test_inference_speed(model, test_loader, num_runs=100, model_name="Model", logger=None):
    """测试推理速度"""
    model.eval()
    
    if logger:
        logger.info(f"Testing {model_name} inference speed...")
        logger.info(f"  Number of runs: {num_runs}")
        logger.info(f"  Batch size: {test_loader.batch_size}")
    
    # 预热
    if logger:
        logger.info("  Warming up...")
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            if isinstance(model, CapsNet):
                data = Variable(data)
                if USE_CUDA:
                    data = data.cuda()
                _ = model(data)
            elif isinstance(model, (LeNet, LeNetMultiChannel)):
                data = data.to(next(model.parameters()).device)
                _ = model(data)
            else:
                data = data.to(next(model.parameters()).device)
                _ = model(data, do_recon=False)
            if i >= 10:
                break
    
    # 测试推理时间
    if logger:
        logger.info("  Measuring inference time...")
    times = []
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            if i >= num_runs:
                break
            
            if isinstance(model, CapsNet):
                data = Variable(data)
                if USE_CUDA:
                    data = data.cuda()
                torch.cuda.synchronize() if USE_CUDA else None
                start = time.time()
                _ = model(data)
                torch.cuda.synchronize() if USE_CUDA else None
            elif isinstance(model, (LeNet, LeNetMultiChannel)):
                data = data.to(next(model.parameters()).device)
                torch.cuda.synchronize() if USE_CUDA else None
                start = time.time()
                _ = model(data)
                torch.cuda.synchronize() if USE_CUDA else None
            else:
                data = data.to(next(model.parameters()).device)
                torch.cuda.synchronize() if USE_CUDA else None
                start = time.time()
                _ = model(data, do_recon=False)
                torch.cuda.synchronize() if USE_CUDA else None
            
            elapsed = time.time() - start
            times.append(elapsed)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    fps = 1.0 / avg_time
    
    if logger:
        logger.info(f"\n{model_name} - Inference Speed Results:")
        logger.info(f"  Average Time: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
        logger.info(f"  Min Time: {min_time*1000:.2f}ms")
        logger.info(f"  Max Time: {max_time*1000:.2f}ms")
        logger.info(f"  FPS: {fps:.2f}")
        logger.info(f"  Throughput: {fps * test_loader.batch_size:.2f} samples/second")
    
    print(f"{model_name} - Average Inference Time: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
    print(f"{model_name} - FPS: {fps:.2f}")
    
    return {
        'avg_time': avg_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'fps': fps,
        'times': times
    }


# ==============================
# 5. 重构质量对比（如果支持）
# ==============================
def test_reconstruction_quality(model, test_loader, num_samples=100, model_name="Model"):
    """测试重构质量（MSE, SSIM等）"""
    if isinstance(model, CapsNet):
        # CapsNet 支持重构
        model.eval()
        mse_scores = []
        
        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                if i * data.size(0) >= num_samples:
                    break
                
                target_onehot = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
                data, target_onehot = Variable(data), Variable(target_onehot)
                if USE_CUDA:
                    data, target_onehot = data.cuda(), target_onehot.cuda()
                
                output, reconstructions, masked = model(data)
                mse = F.mse_loss(reconstructions, data).item()
                mse_scores.append(mse)
        
        avg_mse = np.mean(mse_scores)
        print(f"{model_name} - Average Reconstruction MSE: {avg_mse:.6f}")
        return {'mse': avg_mse, 'scores': mse_scores}
    
    elif hasattr(model, 'recon_alpha'):
        # IBCapsNet with reconstruction
        model.eval()
        mse_scores = []
        
        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                if i * data.size(0) >= num_samples:
                    break
                
                data, target = data.to(next(model.parameters()).device), target.to(next(model.parameters()).device)
                probs, kl, recon_img = model(data, target=target, do_recon=True)
                
                if recon_img is not None:
                    # recon_img 和 data 都是 [B, C, H, W] 形状，直接比较
                    mse = F.mse_loss(recon_img, data).item()
                    mse_scores.append(mse)
        
        if mse_scores:
            avg_mse = np.mean(mse_scores)
            print(f"{model_name} - Average Reconstruction MSE: {avg_mse:.6f}")
            return {'mse': avg_mse, 'scores': mse_scores}
    
    print(f"{model_name} - Reconstruction not supported")
    return None


# ==============================
# 6. 错误分析
# ==============================
def analyze_errors(predictions, targets, probabilities, model_name, save_dir):
    """分析错误分类的样本"""
    errors = []
    for i, (pred, true) in enumerate(zip(predictions, targets)):
        if pred != true:
            errors.append({
                'index': i,
                'true_label': int(true),
                'predicted_label': int(pred),
                'confidence': float(probabilities[i][pred])
            })
    
    # 按置信度排序
    errors.sort(key=lambda x: x['confidence'], reverse=True)
    
    # 保存错误分析
    with open(os.path.join(save_dir, f'error_analysis_{model_name}.json'), 'w') as f:
        json.dump({
            'total_errors': len(errors),
            'error_rate': len(errors) / len(predictions),
            'top_errors': errors[:50]  # 前50个错误
        }, f, indent=2)
    
    # 统计最常见的错误类型
    error_pairs = {}
    for err in errors:
        pair = (int(err['true_label']), int(err['predicted_label']))
        error_pairs[pair] = error_pairs.get(pair, 0) + 1
    
    most_common_errors = sorted(error_pairs.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print(f"\n{model_name} - Most Common Error Patterns:")
    for (true, pred), count in most_common_errors:
        print(f"  {true} -> {pred}: {count} times")
    
    return errors, error_pairs


# ==============================
# 7. 可视化对比
# ==============================
def visualize_comparison(results_capsnet, results_ibcapsnet, results_lenet=None, save_dir=None):
    """生成对比可视化"""
    
    # 1. 准确率对比柱状图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    models = ['CapsNet', 'IBCapsNet']
    accuracies = [results_capsnet['basic']['accuracy'], results_ibcapsnet['basic']['accuracy']]
    if results_lenet is not None:
        models.append('LeNet')
        accuracies.append(results_lenet['basic']['accuracy'])
    
    colors = ['blue', 'green', 'orange'] if results_lenet is not None else ['blue', 'green']
    plt.bar(models, accuracies, color=colors[:len(models)])
    plt.ylabel('Accuracy')
    plt.title('Overall Accuracy Comparison')
    y_min = min(accuracies) - 0.05 if accuracies else 0.9
    y_max = max(accuracies) + 0.05 if accuracies else 1.0
    plt.ylim([max(0, y_min), min(1.0, y_max)])
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
    
    # 2. 各类别准确率对比
    plt.subplot(1, 2, 2)
    x = np.arange(10)
    if results_lenet is not None:
        width = 0.25
        plt.bar(x - width, results_capsnet['basic']['class_accuracies'], width, label='CapsNet', color='blue')
        plt.bar(x, results_ibcapsnet['basic']['class_accuracies'], width, label='IBCapsNet', color='green')
        plt.bar(x + width, results_lenet['basic']['class_accuracies'], width, label='LeNet', color='orange')
    else:
        width = 0.35
        plt.bar(x - width/2, results_capsnet['basic']['class_accuracies'], width, label='CapsNet', color='blue')
        plt.bar(x + width/2, results_ibcapsnet['basic']['class_accuracies'], width, label='IBCapsNet', color='green')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy Comparison')
    plt.xticks(x, range(10))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_comparison.png'))
    plt.close()
    
    # 3. 鲁棒性对比（多种噪声类型）
    noise_types = ['additive', 'clamped', 'multiplicative', 'salt_pepper', 'gaussian_blur']
    available_noise_types = [nt for nt in noise_types 
                            if f'robustness_{nt}' in results_capsnet and f'robustness_{nt}' in results_ibcapsnet]
    
    if available_noise_types:
        # 创建子图
        n_types = len(available_noise_types)
        cols = 3
        rows = (n_types + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if n_types == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, noise_type in enumerate(available_noise_types):
            ax = axes[idx]
            noise_levels = list(results_capsnet[f'robustness_{noise_type}'].keys())
            capsnet_accs = [results_capsnet[f'robustness_{noise_type}'][n] for n in noise_levels]
            ibcapsnet_accs = [results_ibcapsnet[f'robustness_{noise_type}'][n] for n in noise_levels]
            
            ax.plot(noise_levels, capsnet_accs, 'o-', label='CapsNet', linewidth=2, markersize=6, color='blue')
            ax.plot(noise_levels, ibcapsnet_accs, 's-', label='IBCapsNet', linewidth=2, markersize=6, color='green')
            if results_lenet is not None and f'robustness_{noise_type}' in results_lenet:
                lenet_accs = [results_lenet[f'robustness_{noise_type}'][n] for n in noise_levels]
                ax.plot(noise_levels, lenet_accs, '^-', label='LeNet', linewidth=2, markersize=6, color='orange')
            ax.set_xlabel('Noise Level')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'Robustness to {noise_type.replace("_", " ").title()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.0])
        
        # 隐藏多余的子图
        for idx in range(n_types, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'robustness_noise_comparison_all.png'), dpi=150)
        plt.close()
        
        # 单独绘制clamped噪声对比（IBCapsNet的优势场景）
        if 'robustness_clamped' in results_capsnet and 'robustness_clamped' in results_ibcapsnet:
            plt.figure(figsize=(10, 6))
            noise_levels = list(results_capsnet['robustness_clamped'].keys())
            capsnet_accs = [results_capsnet['robustness_clamped'][n] for n in noise_levels]
            ibcapsnet_accs = [results_ibcapsnet['robustness_clamped'][n] for n in noise_levels]
            
            plt.plot(noise_levels, capsnet_accs, 'o-', label='CapsNet', linewidth=2, markersize=8, color='blue')
            plt.plot(noise_levels, ibcapsnet_accs, 's-', label='IBCapsNet', linewidth=2, markersize=8, color='green')
            if results_lenet is not None and 'robustness_clamped' in results_lenet:
                lenet_accs = [results_lenet['robustness_clamped'][n] for n in noise_levels]
                plt.plot(noise_levels, lenet_accs, '^-', label='LeNet', linewidth=2, markersize=8, color='orange')
            plt.xlabel('Noise Level', fontsize=12)
            plt.ylabel('Accuracy', fontsize=12)
            plt.title('Robustness to Clamped Additive Noise (IBCapsNet Advantage)', fontsize=14, fontweight='bold')
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.ylim([0, 1.0])
            
            # 标注改进幅度
            for i, (cl, ib) in enumerate(zip(capsnet_accs, ibcapsnet_accs)):
                if ib > cl:
                    improvement = ib - cl
                    plt.annotate(f'+{improvement:.3f}', 
                               xy=(noise_levels[i], ib), 
                               xytext=(noise_levels[i], ib + 0.05),
                               ha='center', fontsize=9, color='green', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'robustness_clamped_comparison.png'), dpi=150)
            plt.close()
    
    # 4. 推理速度对比
    if 'inference_speed' in results_capsnet and 'inference_speed' in results_ibcapsnet:
        plt.figure(figsize=(8, 6))
        models = ['CapsNet', 'IBCapsNet']
        fps = [results_capsnet['inference_speed']['fps'], results_ibcapsnet['inference_speed']['fps']]
        if results_lenet is not None and 'inference_speed' in results_lenet:
            models.append('LeNet')
            fps.append(results_lenet['inference_speed']['fps'])
        colors = ['blue', 'green', 'orange'] if results_lenet is not None and 'inference_speed' in results_lenet else ['blue', 'green']
        plt.bar(models, fps, color=colors[:len(models)])
        plt.ylabel('FPS (Frames Per Second)')
        plt.title('Inference Speed Comparison')
        for i, f in enumerate(fps):
            plt.text(i, f + max(fps) * 0.05, f'{f:.2f} FPS', ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'inference_speed_comparison.png'))
        plt.close()


# ==============================
# 主函数
# ==============================
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Comprehensive Model Testing and Comparison')
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'mnist-small', 'fashion-mnist', 'fashion-mnist-small', 'cifar10', 'cifar10-small', 'svhn'],
                       help='Dataset to use (default: mnist)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for testing (default: 128)')
    parser.add_argument('--capsnet_path', type=str, default=None,
                       help='Path to CapsNet model weights (default: auto-detect)')
    parser.add_argument('--ibcapsnet_path', type=str, default=None,
                       help='Path to IBCapsNet model weights (default: auto-detect)')
    parser.add_argument('--lenet_path', type=str, default=None,
                       help='Path to LeNet model weights (default: auto-detect)')
    parser.add_argument('--result_dir', type=str, default=None,
                       help='Result directory (default: auto-generated)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Comprehensive Model Testing and Comparison")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    print(f"CapsNet path: {args.capsnet_path or 'auto-detect'}")
    print(f"IBCapsNet path: {args.ibcapsnet_path or 'auto-detect'}")
    print(f"LeNet path: {args.lenet_path or 'auto-detect'}")
    print("=" * 60)
    
    # 创建结果目录
    if args.result_dir:
        result_dir = args.result_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_dir = f'test_comparison_results_{args.dataset}_{timestamp}'
    os.makedirs(result_dir, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(result_dir, 'test_comparison.log')
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
    logger.info("Comprehensive Model Testing and Comparison")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Result directory: {result_dir}")
    
    # 根据数据集确定配置
    dataset_name = args.dataset
    if dataset_name in ['mnist', 'mnist-small', 'fashion-mnist', 'fashion-mnist-small']:
        config_name = 'mnist'  # FashionMNIST使用MNIST配置（都是28x28灰度图像）
        input_channels = 1
        input_size = 28
        num_classes = 10
    elif dataset_name in ['cifar10', 'cifar10-small']:
        config_name = 'cifar10'
        input_channels = 3
        input_size = 32
        num_classes = 10
    elif dataset_name == 'svhn':
        config_name = 'cifar10'  # SVHN使用CIFAR-10配置（都是32x32 RGB图像）
        input_channels = 3
        input_size = 32
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    logger.info(f"Input channels: {input_channels}, Input size: {input_size}x{input_size}, Num classes: {num_classes}")
    
    # 加载数据集
    config = Config(config_name)
    if dataset_name.endswith('-small'):
        dataset = Dataset(dataset_name, args.batch_size, samples_per_class=100)
    else:
        dataset = Dataset(dataset_name, args.batch_size)
    test_loader = dataset.test_loader
    
    # 加载模型（需要先训练或加载预训练模型）
    logger.info("Loading models...")
    
    # CapsNet
    capsnet = CapsNet(config)
    if USE_CUDA:
        capsnet = capsnet.cuda()
    
    # 确定CapsNet模型路径
    if args.capsnet_path:
        capsnet_path = args.capsnet_path
    else:
        # 自动检测：尝试多个可能的路径
        possible_paths = [
            f'comparison_results_{dataset_name}_*/capsnet_best.pth',
            f'comparison_results_{config_name}_*/capsnet_best.pth',
            'capsnet_best.pth',
            f'capsnet_{dataset_name}_best.pth'
        ]
        capsnet_path = None
        for pattern in possible_paths:
            matches = glob.glob(pattern)
            if matches:
                capsnet_path = matches[0]
                break
    
    if capsnet_path and os.path.exists(capsnet_path):
        capsnet.load_state_dict(torch.load(capsnet_path, map_location='cpu'))
        logger.info(f"Loaded CapsNet from {capsnet_path}")
    else:
        logger.warning(f"CapsNet model not found, using untrained model")
        capsnet_path = None
    
    # IBCapsNet (默认使用squash分类器)
    ibcapsnet = IBCapsNetWithRecon(
        latent_dim=16, 
        beta=1e-3, 
        recon_alpha=0.0005, 
        classifier_type='squash',
        input_width=input_size,
        input_height=input_size,
        input_channel=input_channels
    )
    if USE_CUDA:
        ibcapsnet = ibcapsnet.cuda()
    
    # 确定IBCapsNet模型路径
    if args.ibcapsnet_path:
        ibcapsnet_path = args.ibcapsnet_path
    else:
        # 自动检测：优先查找squash版本，如果没有则查找linear版本
        possible_patterns = [
            f'comparison_results_{dataset_name}_*/ibcapsnet-squash_best.pth',
            f'comparison_results_{config_name}_*/ibcapsnet-squash_best.pth',
            f'comparison_results_{dataset_name}_*/ibcapsnet-linear_best.pth',
            f'comparison_results_{config_name}_*/ibcapsnet-linear_best.pth',
            'ibcapsnet-squash_best.pth',
            'ibcapsnet-linear_best.pth',
            f'ibcapsnet_{dataset_name}_best.pth'
        ]
        ibcapsnet_path = None
        for pattern in possible_patterns:
            matches = glob.glob(pattern)
            if matches:
                ibcapsnet_path = matches[0]
                break
    
    if ibcapsnet_path and os.path.exists(ibcapsnet_path):
        ibcapsnet.load_state_dict(torch.load(ibcapsnet_path, map_location='cpu'))
        logger.info(f"Loaded IBCapsNet from {ibcapsnet_path}")
    else:
        logger.warning(f"IBCapsNet model not found, using untrained model with squash classifier")
        ibcapsnet_path = None
    
    # LeNet (使用支持多通道的版本)
    if input_channels == 1:
        lenet = LeNet(num_classes=num_classes)
    else:
        lenet = LeNetMultiChannel(num_classes=num_classes, in_channels=input_channels, input_size=input_size)
    if USE_CUDA:
        lenet = lenet.cuda()
    
    # 确定LeNet模型路径
    if args.lenet_path:
        lenet_path = args.lenet_path
    else:
        # 自动检测
        possible_patterns = [
            f'lenet_{dataset_name}_best.pth',
            f'lenet_{config_name}_best.pth',
            'lenet_best.pth',
            'lenet_mnist_best.pth'  # 默认MNIST路径
        ]
        lenet_path = None
        for pattern in possible_patterns:
            if os.path.exists(pattern):
                lenet_path = pattern
                break
    
    if lenet_path and os.path.exists(lenet_path):
        lenet.load_state_dict(torch.load(lenet_path, map_location='cpu'))
        logger.info(f"Loaded LeNet from {lenet_path}")
    else:
        logger.warning(f"LeNet model not found, using untrained model")
        lenet_path = None
    
    results_capsnet = {}
    results_ibcapsnet = {}
    results_lenet = {}
    
    # 1. 基础性能指标
    logger.info("\n" + "=" * 60)
    logger.info("1. Testing Basic Metrics")
    logger.info("=" * 60)
    results_capsnet['basic'] = test_basic_metrics(capsnet, test_loader, "CapsNet", logger)
    results_ibcapsnet['basic'] = test_basic_metrics(ibcapsnet, test_loader, "IBCapsNet", logger)
    results_lenet['basic'] = test_basic_metrics(lenet, test_loader, "LeNet", logger)
    
    # 2. 混淆矩阵和分类报告
    logger.info("\n2. Generating Confusion Matrices and Classification Reports...")
    generate_confusion_matrix(results_capsnet['basic']['predictions'], 
                            results_capsnet['basic']['targets'], 
                            "CapsNet", result_dir)
    generate_confusion_matrix(results_ibcapsnet['basic']['predictions'], 
                            results_ibcapsnet['basic']['targets'], 
                            "IBCapsNet", result_dir)
    generate_confusion_matrix(results_lenet['basic']['predictions'], 
                            results_lenet['basic']['targets'], 
                            "LeNet", result_dir)
    
    generate_classification_report(results_capsnet['basic']['predictions'], 
                                  results_capsnet['basic']['targets'], 
                                  "CapsNet", result_dir)
    generate_classification_report(results_ibcapsnet['basic']['predictions'], 
                                  results_ibcapsnet['basic']['targets'], 
                                  "IBCapsNet", result_dir)
    generate_classification_report(results_lenet['basic']['predictions'], 
                                  results_lenet['basic']['targets'], 
                                  "LeNet", result_dir)
    
    # 3. 鲁棒性测试（多种噪声类型）
    logger.info("\n3. Testing Robustness to Different Noise Types...")
    
    # 3.1 加性噪声（不加clamp）
    logger.info("\n" + "=" * 60)
    logger.info("3.1 Testing Additive Noise (no clamp)")
    logger.info("=" * 60)
    results_capsnet['robustness_additive'] = test_robustness_noise(
        capsnet, test_loader, model_name="CapsNet", noise_type='additive', logger=logger
    )
    results_ibcapsnet['robustness_additive'] = test_robustness_noise(
        ibcapsnet, test_loader, model_name="IBCapsNet", noise_type='additive', logger=logger
    )
    results_lenet['robustness_additive'] = test_robustness_noise(
        lenet, test_loader, model_name="LeNet", noise_type='additive', logger=logger
    )
    
    # 3.2 加性噪声（加clamp，产生饱和效应）- 这是IBCapsNet的优势场景
    logger.info("\n" + "=" * 60)
    logger.info("3.2 Testing Clamped Additive Noise (saturation effect) - IBCapsNet Advantage")
    logger.info("=" * 60)
    results_capsnet['robustness_clamped'] = test_robustness_noise(
        capsnet, test_loader, model_name="CapsNet", noise_type='clamped', logger=logger
    )
    results_ibcapsnet['robustness_clamped'] = test_robustness_noise(
        ibcapsnet, test_loader, model_name="IBCapsNet", noise_type='clamped', logger=logger
    )
    results_lenet['robustness_clamped'] = test_robustness_noise(
        lenet, test_loader, model_name="LeNet", noise_type='clamped', logger=logger
    )
    
    # 3.3 乘性噪声（对高值区域影响更大）
    logger.info("\n" + "=" * 60)
    logger.info("3.3 Testing Multiplicative Noise")
    logger.info("=" * 60)
    results_capsnet['robustness_multiplicative'] = test_robustness_noise(
        capsnet, test_loader, model_name="CapsNet", noise_type='multiplicative', logger=logger
    )
    results_ibcapsnet['robustness_multiplicative'] = test_robustness_noise(
        ibcapsnet, test_loader, model_name="IBCapsNet", noise_type='multiplicative', logger=logger
    )
    results_lenet['robustness_multiplicative'] = test_robustness_noise(
        lenet, test_loader, model_name="LeNet", noise_type='multiplicative', logger=logger
    )
    
    # 3.4 椒盐噪声（随机像素变为0或1）
    logger.info("\n" + "=" * 60)
    logger.info("3.4 Testing Salt-Pepper Noise")
    logger.info("=" * 60)
    results_capsnet['robustness_salt_pepper'] = test_robustness_noise(
        capsnet, test_loader, model_name="CapsNet", noise_type='salt_pepper', logger=logger
    )
    results_ibcapsnet['robustness_salt_pepper'] = test_robustness_noise(
        ibcapsnet, test_loader, model_name="IBCapsNet", noise_type='salt_pepper', logger=logger
    )
    results_lenet['robustness_salt_pepper'] = test_robustness_noise(
        lenet, test_loader, model_name="LeNet", noise_type='salt_pepper', logger=logger
    )
    
    # 3.5 高斯模糊（模拟低质量图像）
    logger.info("\n" + "=" * 60)
    logger.info("3.5 Testing Gaussian Blur")
    logger.info("=" * 60)
    results_capsnet['robustness_gaussian_blur'] = test_robustness_noise(
        capsnet, test_loader, model_name="CapsNet", noise_type='gaussian_blur', 
        noise_levels=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], logger=logger
    )
    results_ibcapsnet['robustness_gaussian_blur'] = test_robustness_noise(
        ibcapsnet, test_loader, model_name="IBCapsNet", noise_type='gaussian_blur',
        noise_levels=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], logger=logger
    )
    results_lenet['robustness_gaussian_blur'] = test_robustness_noise(
        lenet, test_loader, model_name="LeNet", noise_type='gaussian_blur',
        noise_levels=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], logger=logger
    )
    
    # 为了兼容性，保留原来的robustness_noise键
    results_capsnet['robustness_noise'] = results_capsnet['robustness_additive']
    results_ibcapsnet['robustness_noise'] = results_ibcapsnet['robustness_additive']
    results_lenet['robustness_noise'] = results_lenet['robustness_additive']
    
    logger.info("\n" + "=" * 60)
    logger.info("4. Testing Robustness to Rotation")
    logger.info("=" * 60)
    results_capsnet['robustness_rotation'] = test_robustness_rotation(capsnet, test_loader, model_name="CapsNet", logger=logger)
    results_ibcapsnet['robustness_rotation'] = test_robustness_rotation(ibcapsnet, test_loader, model_name="IBCapsNet", logger=logger)
    results_lenet['robustness_rotation'] = test_robustness_rotation(lenet, test_loader, model_name="LeNet", logger=logger)
    
    logger.info("\n" + "=" * 60)
    logger.info("5. Testing Robustness to Occlusion")
    logger.info("=" * 60)
    results_capsnet['robustness_occlusion'] = test_robustness_occlusion(capsnet, test_loader, model_name="CapsNet", logger=logger)
    results_ibcapsnet['robustness_occlusion'] = test_robustness_occlusion(ibcapsnet, test_loader, model_name="IBCapsNet", logger=logger)
    results_lenet['robustness_occlusion'] = test_robustness_occlusion(lenet, test_loader, model_name="LeNet", logger=logger)
    
    # 4. 推理速度
    logger.info("\n" + "=" * 60)
    logger.info("6. Testing Inference Speed")
    logger.info("=" * 60)
    results_capsnet['inference_speed'] = test_inference_speed(capsnet, test_loader, model_name="CapsNet", logger=logger)
    results_ibcapsnet['inference_speed'] = test_inference_speed(ibcapsnet, test_loader, model_name="IBCapsNet", logger=logger)
    results_lenet['inference_speed'] = test_inference_speed(lenet, test_loader, model_name="LeNet", logger=logger)
    
    # 5. 重构质量
    logger.info("\n7. Testing Reconstruction Quality...")
    results_capsnet['reconstruction'] = test_reconstruction_quality(capsnet, test_loader, model_name="CapsNet")
    results_ibcapsnet['reconstruction'] = test_reconstruction_quality(ibcapsnet, test_loader, model_name="IBCapsNet")
    
    # 6. 错误分析
    logger.info("\n8. Analyzing Errors...")
    analyze_errors(results_capsnet['basic']['predictions'], 
                  results_capsnet['basic']['targets'],
                  results_capsnet['basic']['probabilities'],
                  "CapsNet", result_dir)
    analyze_errors(results_ibcapsnet['basic']['predictions'], 
                  results_ibcapsnet['basic']['targets'],
                  results_ibcapsnet['basic']['probabilities'],
                  "IBCapsNet", result_dir)
    analyze_errors(results_lenet['basic']['predictions'], 
                  results_lenet['basic']['targets'],
                  results_lenet['basic']['probabilities'],
                  "LeNet", result_dir)
    
    # 7. 可视化对比
    logger.info("\n9. Generating Comparison Visualizations...")
    visualize_comparison(results_capsnet, results_ibcapsnet, results_lenet, result_dir)
    
    # 分析IBCapsNet的优势
    logger.info("\n" + "=" * 60)
    logger.info("IBCapsNet Advantage Analysis")
    logger.info("=" * 60)
    
    advantage_analysis = {}
    
    # 分析clamped噪声下的优势（核心发现）
    if 'robustness_clamped' in results_capsnet and 'robustness_clamped' in results_ibcapsnet:
        capsnet_clamped = results_capsnet['robustness_clamped']
        ibcapsnet_clamped = results_ibcapsnet['robustness_clamped']
        
        improvements = {}
        for noise_level in capsnet_clamped.keys():
            if noise_level > 0:  # 排除noise_level=0
                improvement = ibcapsnet_clamped[noise_level] - capsnet_clamped[noise_level]
                improvements[noise_level] = improvement
        
        max_improvement = max(improvements.values()) if improvements else 0
        max_improvement_level = max(improvements, key=improvements.get) if improvements else None
        
        advantage_analysis['clamped_noise'] = {
            'improvements': improvements,
            'max_improvement': max_improvement,
            'max_improvement_level': max_improvement_level,
            'average_improvement': np.mean(list(improvements.values())) if improvements else 0
        }
        
        logger.info(f"\nClamped Noise Analysis (IBCapsNet Core Advantage):")
        logger.info(f"  Maximum improvement: {max_improvement:.6f} at noise level {max_improvement_level}")
        logger.info(f"  Average improvement: {advantage_analysis['clamped_noise']['average_improvement']:.6f}")
        logger.info(f"  Detailed improvements by noise level:")
        for level, imp in sorted(improvements.items()):
            capsnet_acc = capsnet_clamped[level]
            ibcapsnet_acc = ibcapsnet_clamped[level]
            improvement_pct = (imp / capsnet_acc * 100) if capsnet_acc > 0 else 0
            logger.info(f"    Noise level {level}:")
            logger.info(f"      CapsNet: {capsnet_acc:.6f}, IBCapsNet: {ibcapsnet_acc:.6f}")
            logger.info(f"      Absolute improvement: {imp:+.6f}")
            logger.info(f"      Relative improvement: {improvement_pct:+.2f}%")
    
    # 分析其他噪声类型的优势
    noise_types_to_analyze = ['additive', 'multiplicative', 'salt_pepper', 'gaussian_blur']
    for noise_type in noise_types_to_analyze:
        key = f'robustness_{noise_type}'
        if key in results_capsnet and key in results_ibcapsnet:
            capsnet_results = results_capsnet[key]
            ibcapsnet_results = results_ibcapsnet[key]
            
            improvements = {}
            for noise_level in capsnet_results.keys():
                if noise_level > 0:
                    improvement = ibcapsnet_results[noise_level] - capsnet_results[noise_level]
                    improvements[noise_level] = improvement
            
            if improvements:
                advantage_analysis[noise_type] = {
                    'average_improvement': np.mean(list(improvements.values())),
                    'max_improvement': max(improvements.values()),
                    'min_improvement': min(improvements.values()),
                    'improvements': improvements
                }
                logger.info(f"\n{noise_type.replace('_', ' ').title()} Noise Analysis:")
                logger.info(f"  Average improvement: {advantage_analysis[noise_type]['average_improvement']:.6f}")
                logger.info(f"  Maximum improvement: {advantage_analysis[noise_type]['max_improvement']:.6f}")
                logger.info(f"  Minimum improvement: {advantage_analysis[noise_type]['min_improvement']:.6f}")
                logger.info(f"  Detailed improvements:")
                for level, imp in sorted(improvements.items()):
                    capsnet_acc = capsnet_results[level]
                    ibcapsnet_acc = ibcapsnet_results[level]
                    logger.info(f"    Noise level {level}: {imp:+.6f} "
                               f"(CapsNet: {capsnet_acc:.6f}, IBCapsNet: {ibcapsnet_acc:.6f})")
    
    # 保存所有结果
    all_results = {
        'dataset': args.dataset,
        'model_paths': {
            'capsnet': capsnet_path,
            'ibcapsnet': ibcapsnet_path,
            'lenet': lenet_path
        },
        'capsnet': results_capsnet,
        'ibcapsnet': results_ibcapsnet,
        'lenet': results_lenet,
        'advantage_analysis': advantage_analysis,
        'summary': {
            'capsnet_accuracy': results_capsnet['basic']['accuracy'],
            'ibcapsnet_accuracy': results_ibcapsnet['basic']['accuracy'],
            'lenet_accuracy': results_lenet['basic']['accuracy'],
            'accuracy_improvement': results_ibcapsnet['basic']['accuracy'] - results_capsnet['basic']['accuracy'],
            'capsnet_fps': results_capsnet['inference_speed']['fps'],
            'ibcapsnet_fps': results_ibcapsnet['inference_speed']['fps'],
            'lenet_fps': results_lenet['inference_speed']['fps'],
            'speedup': results_ibcapsnet['inference_speed']['fps'] / results_capsnet['inference_speed']['fps']
        }
    }
    
    with open(os.path.join(result_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info("\n" + "=" * 60)
    logger.info("Testing Complete!")
    logger.info(f"Results saved to: {result_dir}")
    logger.info("=" * 60)
    
    # 打印摘要
    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Basic Performance:")
    logger.info(f"  CapsNet Accuracy: {results_capsnet['basic']['accuracy']:.6f}")
    logger.info(f"  IBCapsNet Accuracy: {results_ibcapsnet['basic']['accuracy']:.6f}")
    logger.info(f"  LeNet Accuracy: {results_lenet['basic']['accuracy']:.6f}")
    logger.info(f"  Accuracy Difference (IBCapsNet vs CapsNet): {results_ibcapsnet['basic']['accuracy'] - results_capsnet['basic']['accuracy']:+.6f}")
    logger.info(f"\nInference Speed:")
    logger.info(f"  CapsNet FPS: {results_capsnet['inference_speed']['fps']:.2f}")
    logger.info(f"  IBCapsNet FPS: {results_ibcapsnet['inference_speed']['fps']:.2f}")
    logger.info(f"  LeNet FPS: {results_lenet['inference_speed']['fps']:.2f}")
    logger.info(f"  Speedup (IBCapsNet vs CapsNet): {results_ibcapsnet['inference_speed']['fps'] / results_capsnet['inference_speed']['fps']:.2f}x")
    
    # 总结鲁棒性优势
    if 'robustness_clamped' in advantage_analysis:
        logger.info(f"\nRobustness Advantage (Clamped Noise):")
        logger.info(f"  Average improvement: {advantage_analysis['clamped_noise']['average_improvement']:.6f}")
        logger.info(f"  Maximum improvement: {advantage_analysis['clamped_noise']['max_improvement']:.6f} "
                   f"at noise level {advantage_analysis['clamped_noise']['max_improvement_level']}")
    
    logger.info("=" * 60)
    
    # 同时打印到控制台
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"CapsNet Accuracy: {results_capsnet['basic']['accuracy']:.4f}")
    print(f"IBCapsNet Accuracy: {results_ibcapsnet['basic']['accuracy']:.4f}")
    print(f"LeNet Accuracy: {results_lenet['basic']['accuracy']:.4f}")
    print(f"Accuracy Difference (IBCapsNet vs CapsNet): {results_ibcapsnet['basic']['accuracy'] - results_capsnet['basic']['accuracy']:.4f}")
    print(f"\nCapsNet FPS: {results_capsnet['inference_speed']['fps']:.2f}")
    print(f"IBCapsNet FPS: {results_ibcapsnet['inference_speed']['fps']:.2f}")
    print(f"LeNet FPS: {results_lenet['inference_speed']['fps']:.2f}")
    print(f"Speedup (IBCapsNet vs CapsNet): {results_ibcapsnet['inference_speed']['fps'] / results_capsnet['inference_speed']['fps']:.2f}x")
    if 'robustness_clamped' in advantage_analysis:
        print(f"\nIBCapsNet Clamped Noise Advantage: {advantage_analysis['clamped_noise']['average_improvement']:.4f} (avg)")
    print("=" * 60)


if __name__ == '__main__':
    main()

