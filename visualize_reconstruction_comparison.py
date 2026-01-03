"""
可视化对比CapsNet和IBCapsnet-squash在不同噪声情况下的重构结果
"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import argparse
from datetime import datetime
import logging

from capsnet import CapsNet
from IBCapsnet import IBCapsNetWithRecon
from data_loader import Dataset
from test_capsnet import Config

USE_CUDA = True if torch.cuda.is_available() else False


def add_clamp_noise(data, noise_level=0.3):
    """
    添加clamp噪声（加性噪声后clamp到[0,1]）
    
    Args:
        data: 输入图像 [B, C, H, W]
        noise_level: 噪声强度
    Returns:
        noisy_data: 添加噪声后的图像
    """
    noise = torch.randn_like(data) * noise_level
    noisy_data = torch.clamp(data + noise, 0, 1)
    return noisy_data


def add_gaussian_noise(data, noise_level=0.2):
    """
    添加高斯噪声（加性噪声，不clamp）
    
    Args:
        data: 输入图像 [B, C, H, W]
        noise_level: 噪声强度
    Returns:
        noisy_data: 添加噪声后的图像
    """
    noise = torch.randn_like(data) * noise_level
    noisy_data = data + noise
    return noisy_data


def add_salt_pepper_noise(data, salt_prob=0.1, pepper_prob=0.1):
    """
    添加椒盐噪声（突出IBCapsnet的优势）
    
    Args:
        data: 输入图像 [B, C, H, W]
        salt_prob: 盐噪声概率（白点）
        pepper_prob: 椒噪声概率（黑点）
    Returns:
        noisy_data: 添加噪声后的图像
    """
    noisy_data = data.clone()
    batch_size, channels, height, width = data.shape
    
    # 生成随机掩码
    random_mask = torch.rand(batch_size, channels, height, width, device=data.device)
    
    # 盐噪声（设置为1）
    salt_mask = random_mask < salt_prob
    noisy_data[salt_mask] = 1.0
    
    # 椒噪声（设置为0）
    pepper_mask = (random_mask >= salt_prob) & (random_mask < salt_prob + pepper_prob)
    noisy_data[pepper_mask] = 0.0
    
    return noisy_data


def get_capsnet_reconstruction(model, data, target):
    """
    获取CapsNet的重构结果
    
    Args:
        model: CapsNet模型
        data: 输入图像 [B, C, H, W]
        target: 真实标签 [B]
    Returns:
        reconstructions: 重构图像 [B, C, H, W]
    """
    model.eval()
    with torch.no_grad():
        target_onehot = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
        data_var, target_onehot_var = Variable(data), Variable(target_onehot)
        
        if USE_CUDA:
            data_var, target_onehot_var = data_var.cuda(), target_onehot_var.cuda()
        
        output, reconstructions, masked = model(data_var)
        return reconstructions.cpu()


def get_ibcapsnet_reconstruction(model, data, target):
    """
    获取IBCapsnet的重构结果
    
    Args:
        model: IBCapsnet模型
        data: 输入图像 [B, C, H, W]
        target: 真实标签 [B]
    Returns:
        recon_img: 重构图像 [B, C, H, W]
    """
    model.eval()
    with torch.no_grad():
        data_tensor = data.to(next(model.parameters()).device)
        target_tensor = target.to(next(model.parameters()).device)
        
        probs, kl, recon_img = model(data_tensor, target=target_tensor, do_recon=True)
        return recon_img.cpu()


def visualize_reconstruction_comparison(
    capsnet_model, ibcapsnet_model, test_loader, 
    num_samples=10, noise_levels={'clamp': 0.3, 'gaussian': 0.2, 'salt_pepper': 0.15},
    save_path='reconstruction_comparison.png', dataset_name='mnist',
    target_class=None
):
    """
    可视化对比CapsNet和IBCapsnet的重构结果
    
    Args:
        capsnet_model: CapsNet模型
        ibcapsnet_model: IBCapsnet模型
        test_loader: 测试数据加载器
        num_samples: 要可视化的样本数量（默认10个）
        noise_levels: 不同噪声类型的强度字典
        save_path: 保存路径
        dataset_name: 数据集名称
        target_class: 如果指定，则只选择该类别的前N个样本（None表示选择前N个任意样本）
    """
    # 获取测试集样本
    if target_class is not None:
        # 按类别选择前N个样本
        samples_data = []
        samples_target = []
        
        for data, target in test_loader:
            # 找到属于目标类别的样本
            mask = (target == target_class)
            if mask.any():
                samples_data.append(data[mask])
                samples_target.append(target[mask])
            
            # 检查是否已经收集到足够的样本
            total_collected = sum(len(batch) for batch in samples_data)
            if total_collected >= num_samples:
                break
        
        # 合并所有batch并取前N个
        if samples_data:
            all_data = torch.cat(samples_data, dim=0)
            all_target = torch.cat(samples_target, dim=0)
            
            # 检查是否收集到足够的样本
            actual_num_samples = min(len(all_data), num_samples)
            if len(all_data) < num_samples:
                print(f"Warning: Only found {len(all_data)} samples of class {target_class}, "
                      f"requested {num_samples}. Using all available samples.")
            
            all_data = all_data[:actual_num_samples]
            all_target = all_target[:actual_num_samples]
        else:
            raise ValueError(f"No samples found for class {target_class} in test dataset")
        
        print(f"Selected {len(all_data)} samples of class {target_class}")
    else:
        # 获取测试集前N个样本（原有逻辑）
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
    
    # 使用实际收集到的样本数量
    actual_num_samples = len(all_data)
    
    # 准备可视化
    # 结构：5行（原图、无噪声重构、clamp噪声重构、高斯噪声重构、椒盐噪声重构）x 2栏（CapsNet、IBCapsnet）
    # 每栏显示actual_num_samples个样本
    
    num_rows = 5  # 原图 + 无噪声重构 + 3种噪声重构
    num_cols_per_panel = actual_num_samples  # 每栏的列数（样本数）
    
    fig, axes = plt.subplots(num_rows, num_cols_per_panel * 2, 
                             figsize=(num_cols_per_panel * 2 * 1.2, num_rows * 1.5))
    
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    if num_cols_per_panel == 1:
        axes = axes.reshape(-1, 2)
    
    # 减小行间距和列间距，使排版更紧凑
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    
    # 设置标题
    if target_class is not None:
        title = f'Reconstruction Comparison: CapsNet vs IBCapsnet-Squash (Class {target_class})'
    else:
        title = 'Reconstruction Comparison: CapsNet vs IBCapsnet-Squash'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    # 行标签（简写，不超过4个字符）
    row_labels = [
        'Orig',
        'Recon',
        'Clmp',
        'Gaus',
        'S&P'
    ]
    
    original_data = all_data
    
    # 第一行：显示原图
    row_idx = 0
    for col_idx in range(actual_num_samples):
        # CapsNet栏（左栏）
        ax_caps = axes[row_idx, col_idx]
        ax_caps.axis('off')
        if dataset_name in ['mnist', 'mnist-small', 'fashion-mnist', 'fashion-mnist-small']:
            img_orig = original_data[col_idx].squeeze(0).numpy()
        else:
            img_orig = original_data[col_idx].permute(1, 2, 0).numpy()
            img_orig = np.clip(img_orig, 0, 1)
        ax_caps.imshow(img_orig, cmap='gray' if dataset_name in ['mnist', 'mnist-small', 'fashion-mnist', 'fashion-mnist-small'] else None)
        
        # IBCapsnet栏（右栏）
        ax_ib = axes[row_idx, num_cols_per_panel + col_idx]
        ax_ib.axis('off')
        ax_ib.imshow(img_orig, cmap='gray' if dataset_name in ['mnist', 'mnist-small', 'fashion-mnist', 'fashion-mnist-small'] else None)
    
    # 第二行：无噪声重构
    row_idx = 1
    capsnet_recon = get_capsnet_reconstruction(capsnet_model, original_data, all_target)
    ibcapsnet_recon = get_ibcapsnet_reconstruction(ibcapsnet_model, original_data, all_target)
    
    for col_idx in range(actual_num_samples):
        # CapsNet栏（左栏）
        ax_caps = axes[row_idx, col_idx]
        ax_caps.axis('off')
        if dataset_name in ['mnist', 'mnist-small', 'fashion-mnist', 'fashion-mnist-small']:
            img_caps = capsnet_recon[col_idx].squeeze(0).numpy()
        else:
            img_caps = capsnet_recon[col_idx].permute(1, 2, 0).numpy()
            img_caps = np.clip(img_caps, 0, 1)
        ax_caps.imshow(img_caps, cmap='gray' if dataset_name in ['mnist', 'mnist-small', 'fashion-mnist', 'fashion-mnist-small'] else None)
        
        # IBCapsnet栏（右栏）
        ax_ib = axes[row_idx, num_cols_per_panel + col_idx]
        ax_ib.axis('off')
        if dataset_name in ['mnist', 'mnist-small', 'fashion-mnist', 'fashion-mnist-small']:
            img_ib = ibcapsnet_recon[col_idx].squeeze(0).numpy()
        else:
            img_ib = ibcapsnet_recon[col_idx].permute(1, 2, 0).numpy()
            img_ib = np.clip(img_ib, 0, 1)
        ax_ib.imshow(img_ib, cmap='gray' if dataset_name in ['mnist', 'mnist-small', 'fashion-mnist', 'fashion-mnist-small'] else None)
    
    # 第三行：clamp噪声重构
    row_idx = 2
    noisy_data = add_clamp_noise(original_data, noise_levels['clamp'])
    capsnet_recon = get_capsnet_reconstruction(capsnet_model, noisy_data, all_target)
    ibcapsnet_recon = get_ibcapsnet_reconstruction(ibcapsnet_model, noisy_data, all_target)
    
    for col_idx in range(actual_num_samples):
        # CapsNet栏（左栏）
        ax_caps = axes[row_idx, col_idx]
        ax_caps.axis('off')
        if dataset_name in ['mnist', 'mnist-small', 'fashion-mnist', 'fashion-mnist-small']:
            img_caps = capsnet_recon[col_idx].squeeze(0).numpy()
        else:
            img_caps = capsnet_recon[col_idx].permute(1, 2, 0).numpy()
            img_caps = np.clip(img_caps, 0, 1)
        ax_caps.imshow(img_caps, cmap='gray' if dataset_name in ['mnist', 'mnist-small', 'fashion-mnist', 'fashion-mnist-small'] else None)
        
        # IBCapsnet栏（右栏）
        ax_ib = axes[row_idx, num_cols_per_panel + col_idx]
        ax_ib.axis('off')
        if dataset_name in ['mnist', 'mnist-small', 'fashion-mnist', 'fashion-mnist-small']:
            img_ib = ibcapsnet_recon[col_idx].squeeze(0).numpy()
        else:
            img_ib = ibcapsnet_recon[col_idx].permute(1, 2, 0).numpy()
            img_ib = np.clip(img_ib, 0, 1)
        ax_ib.imshow(img_ib, cmap='gray' if dataset_name in ['mnist', 'mnist-small', 'fashion-mnist', 'fashion-mnist-small'] else None)
    
    # 第四行：高斯噪声重构
    row_idx = 3
    noisy_data = add_gaussian_noise(original_data, noise_levels['gaussian'])
    # 将噪声数据clamp到[0,1]以便可视化
    noisy_data = torch.clamp(noisy_data, 0, 1)
    capsnet_recon = get_capsnet_reconstruction(capsnet_model, noisy_data, all_target)
    ibcapsnet_recon = get_ibcapsnet_reconstruction(ibcapsnet_model, noisy_data, all_target)
    
    for col_idx in range(actual_num_samples):
        # CapsNet栏（左栏）
        ax_caps = axes[row_idx, col_idx]
        ax_caps.axis('off')
        if dataset_name in ['mnist', 'mnist-small', 'fashion-mnist', 'fashion-mnist-small']:
            img_caps = capsnet_recon[col_idx].squeeze(0).numpy()
        else:
            img_caps = capsnet_recon[col_idx].permute(1, 2, 0).numpy()
            img_caps = np.clip(img_caps, 0, 1)
        ax_caps.imshow(img_caps, cmap='gray' if dataset_name in ['mnist', 'mnist-small', 'fashion-mnist', 'fashion-mnist-small'] else None)
        
        # IBCapsnet栏（右栏）
        ax_ib = axes[row_idx, num_cols_per_panel + col_idx]
        ax_ib.axis('off')
        if dataset_name in ['mnist', 'mnist-small', 'fashion-mnist', 'fashion-mnist-small']:
            img_ib = ibcapsnet_recon[col_idx].squeeze(0).numpy()
        else:
            img_ib = ibcapsnet_recon[col_idx].permute(1, 2, 0).numpy()
            img_ib = np.clip(img_ib, 0, 1)
        ax_ib.imshow(img_ib, cmap='gray' if dataset_name in ['mnist', 'mnist-small', 'fashion-mnist', 'fashion-mnist-small'] else None)
    
    # 第五行：椒盐噪声重构（突出IBCapsnet的优势）
    row_idx = 4
    salt_prob = noise_levels.get('salt_pepper', 0.15)
    noisy_data = add_salt_pepper_noise(original_data, salt_prob=salt_prob, pepper_prob=salt_prob)
    capsnet_recon = get_capsnet_reconstruction(capsnet_model, noisy_data, all_target)
    ibcapsnet_recon = get_ibcapsnet_reconstruction(ibcapsnet_model, noisy_data, all_target)
    
    for col_idx in range(actual_num_samples):
        # CapsNet栏（左栏）
        ax_caps = axes[row_idx, col_idx]
        ax_caps.axis('off')
        if dataset_name in ['mnist', 'mnist-small', 'fashion-mnist', 'fashion-mnist-small']:
            img_caps = capsnet_recon[col_idx].squeeze(0).numpy()
        else:
            img_caps = capsnet_recon[col_idx].permute(1, 2, 0).numpy()
            img_caps = np.clip(img_caps, 0, 1)
        ax_caps.imshow(img_caps, cmap='gray' if dataset_name in ['mnist', 'mnist-small', 'fashion-mnist', 'fashion-mnist-small'] else None)
        
        # IBCapsnet栏（右栏）
        ax_ib = axes[row_idx, num_cols_per_panel + col_idx]
        ax_ib.axis('off')
        if dataset_name in ['mnist', 'mnist-small', 'fashion-mnist', 'fashion-mnist-small']:
            img_ib = ibcapsnet_recon[col_idx].squeeze(0).numpy()
        else:
            img_ib = ibcapsnet_recon[col_idx].permute(1, 2, 0).numpy()
            img_ib = np.clip(img_ib, 0, 1)
        ax_ib.imshow(img_ib, cmap='gray' if dataset_name in ['mnist', 'mnist-small', 'fashion-mnist', 'fashion-mnist-small'] else None)
    
    # 在左侧添加行标签（调整位置以适应紧凑排版）
    # 计算每行的y位置（从下往上，考虑紧凑的间距）
    total_height = 0.95  # 总可用高度
    label_start_y = 0.97  # 起始y位置
    row_height = total_height / num_rows  # 每行的高度
    
    for row_idx, label in enumerate(row_labels):
        # 计算该行的中心y位置（从顶部开始）
        y_pos = label_start_y - (row_idx + 0.5) * row_height
        fig.text(0.01, y_pos, label, 
                fontsize=11, fontweight='bold', 
                rotation=90, va='center', ha='center')
    
    # 在顶部添加栏标签（简写）
    # 左栏：列0到num_cols_per_panel-1，中心位置约为0.25
    # 右栏：列num_cols_per_panel到2*num_cols_per_panel-1，中心位置约为0.75
    fig.text(0.25, 0.98, 'Caps', fontsize=14, fontweight='bold', ha='center')
    fig.text(0.75, 0.98, 'IBC', fontsize=14, fontweight='bold', ha='center')
    
    # 使用tight_layout但保持紧凑间距
    plt.tight_layout(rect=[0.05, 0, 1, 0.97], h_pad=0.1, w_pad=0.1)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    
    print(f"Visualization saved to: {save_path}")


def load_models(dataset_name='mnist', capsnet_path=None, ibcapsnet_path=None):
    """
    加载CapsNet和IBCapsnet模型
    
    Args:
        dataset_name: 数据集名称
        capsnet_path: CapsNet模型路径（如果为None则自动查找）
        ibcapsnet_path: IBCapsnet模型路径（如果为None则自动查找）
    Returns:
        capsnet_model: CapsNet模型
        ibcapsnet_model: IBCapsnet模型
        config: 配置对象
    """
    # 根据数据集确定配置
    if dataset_name in ['mnist', 'mnist-small']:
        config = Config('mnist')
        input_channels = 1
        input_size = 28
    elif dataset_name in ['fashion-mnist', 'fashion-mnist-small']:
        # FashionMNIST使用MNIST配置（都是28x28灰度图像，10个类别）
        config = Config('mnist')
        input_channels = 1
        input_size = 28
    elif dataset_name == 'svhn':
        config = Config('cifar10')
        input_channels = 3
        input_size = 32
    elif dataset_name == 'cifar10':
        config = Config('cifar10')
        input_channels = 3
        input_size = 32
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # 加载CapsNet
    capsnet = CapsNet(config)
    if USE_CUDA:
        capsnet = capsnet.cuda()
    
    if capsnet_path is None:
        # 自动查找CapsNet模型
        possible_paths = glob.glob(f'comparison_results_{dataset_name}_*/capsnet_best.pth')
        if not possible_paths:
            possible_paths = glob.glob('comparison_results_*/capsnet_best.pth')
        if possible_paths:
            capsnet_path = sorted(possible_paths)[-1]  # 使用最新的
    
    if capsnet_path and os.path.exists(capsnet_path):
        capsnet.load_state_dict(torch.load(capsnet_path, map_location='cpu'))
        print(f"Loaded CapsNet from: {capsnet_path}")
    else:
        print(f"Warning: CapsNet model not found at {capsnet_path}, using untrained model")
    
    # 加载IBCapsnet-squash
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
    
    if ibcapsnet_path is None:
        # 自动查找IBCapsnet模型
        possible_paths = glob.glob(f'comparison_results_{dataset_name}_*/ibcapsnet-squash_best.pth')
        if not possible_paths:
            possible_paths = glob.glob('comparison_results_*/ibcapsnet-squash_best.pth')
        if possible_paths:
            ibcapsnet_path = sorted(possible_paths)[-1]  # 使用最新的
    
    if ibcapsnet_path and os.path.exists(ibcapsnet_path):
        ibcapsnet.load_state_dict(torch.load(ibcapsnet_path, map_location='cpu'))
        print(f"Loaded IBCapsnet from: {ibcapsnet_path}")
    else:
        print(f"Warning: IBCapsnet model not found at {ibcapsnet_path}, using untrained model")
    
    return capsnet, ibcapsnet, config


def main():
    parser = argparse.ArgumentParser(description='Visualize reconstruction comparison between CapsNet and IBCapsnet')
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'fashion-mnist', 'svhn', 'cifar10'],
                       help='Dataset to use (default: mnist)')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to visualize (default: 10)')
    parser.add_argument('--capsnet_path', type=str, default=None,
                       help='Path to CapsNet model (default: auto-detect)')
    parser.add_argument('--ibcapsnet_path', type=str, default=None,
                       help='Path to IBCapsnet model (default: auto-detect)')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for visualization (default: current directory)')
    parser.add_argument('--clamp_noise', type=float, default=0.3,
                       help='Clamp noise level (default: 0.3)')
    parser.add_argument('--gaussian_noise', type=float, default=0.2,
                       help='Gaussian noise level (default: 0.2)')
    parser.add_argument('--salt_pepper_noise', type=float, default=0.15,
                       help='Salt-pepper noise probability (default: 0.15)')
    parser.add_argument('--target_class', type=int, default=None,
                       help='Target class to visualize (0-9). If specified, will select first N samples of this class (default: None, random samples)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    print("Loading models...")
    capsnet_model, ibcapsnet_model, config = load_models(
        dataset_name=args.dataset,
        capsnet_path=args.capsnet_path,
        ibcapsnet_path=args.ibcapsnet_path
    )
    
    # 加载测试数据
    print(f"Loading {args.dataset} test dataset...")
    dataset = Dataset(args.dataset, 128)  # 第二个参数是位置参数 _batch_size
    test_loader = dataset.test_loader
    
    # 准备噪声级别
    noise_levels = {
        'clamp': args.clamp_noise,
        'gaussian': args.gaussian_noise,
        'salt_pepper': args.salt_pepper_noise
    }
    
    # 生成保存路径
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.target_class is not None:
        filename = f'reconstruction_comparison_{args.dataset}_class{args.target_class}_{timestamp}.png'
    else:
        filename = f'reconstruction_comparison_{args.dataset}_{timestamp}.png'
    save_path = os.path.join(args.output_dir, filename)
    
    # 执行可视化
    if args.target_class is not None:
        print(f"Generating visualization with {args.num_samples} samples of class {args.target_class}...")
    else:
        print(f"Generating visualization with {args.num_samples} samples...")
    
    visualize_reconstruction_comparison(
        capsnet_model=capsnet_model,
        ibcapsnet_model=ibcapsnet_model,
        test_loader=test_loader,
        num_samples=args.num_samples,
        noise_levels=noise_levels,
        save_path=save_path,
        dataset_name=args.dataset,
        target_class=args.target_class
    )
    
    print("Visualization completed!")


if __name__ == '__main__':
    main()