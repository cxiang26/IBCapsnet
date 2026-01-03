"""
IBCapsNet: Information Bottleneck Capsule Network

网络结构：
1. Conv Layer: Conv2d(1, 256, kernel=9) -> ReLU -> [B, 256, 20, 20]
2. Primary Capsules: Conv2d(256, 32*8, kernel=9, stride=2) -> Squash -> [B, 1152, 8]
3. IBCapsules (核心创新):
   - Context Encoder: 全局上下文编码 [B, 1152, 8] -> [B, 256]
   - Class Encoders: 每个类别独立的VAE编码器，生成 (μ, logσ²)
   - Reparameterization: z = μ + ε·σ (ε ~ N(0,1))
   - Classifier: 三种类型可选
     * linear: Linear(latent_dim, 1) -> sigmoid
     * squash: Squash(z) -> norm (保留长向量，压缩短向量)
     * inverse_squash: InverseSquash(z) -> norm (压缩长向量，保留短向量)
4. Decoder (可选): Linear(latent_dim -> 512 -> 1024 -> 784) -> Sigmoid

核心创新点：
1. 【信息瓶颈原理】用VAE替代动态路由：每个类别胶囊通过变分编码器从全局上下文生成，
   通过KL散度正则化实现信息压缩，避免迭代路由的计算开销
2. 【一次前向传播】相比CapsNet的3次迭代路由，IBCapsNet只需一次前向传播，计算效率更高
3. 【灵活的分类器设计】支持三种分类器类型：
   - linear: 传统线性分类，使用 Binary Cross Entropy 损失
   - squash: 保留CapsNet风格，长向量激活强，使用 Margin Loss（与CapsNet一致）
   - inverse_squash: 反向激活，短向量激活强（创新点），使用 Margin Loss（与CapsNet一致）
4. 【KL散度正则化】通过KL散度约束潜在空间，提高泛化能力和训练稳定性
5. 【可重构设计】支持图像重构，通过masked胶囊重构原始图像，增强特征表达能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
import torch.optim as optim
from torchvision import datasets, transforms
from data_loader import Dataset
import logging
import csv
import numpy as np
import os
import json
from datetime import datetime
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ==============================
# 1. Squash 函数（保留 CapsNet 风格）
# ==============================
class Squash(nn.Module):
    def forward(self, x, dim=-1):
        norm_sq = torch.sum(x ** 2, dim=dim, keepdim=True)
        norm = torch.sqrt(norm_sq + 1e-8)
        scale = norm_sq / (1 + norm_sq)
        return scale * x / norm


class InverseSquash(nn.Module):
    """
    反向 Squash 函数
    使用 scale = 1 / (1 + norm_sq) 而不是 norm_sq / (1 + norm_sq)
    这样 scale_inverse = 1 - scale_original
    
    数学关系：
    - 原始 squash: scale = norm_sq / (1 + norm_sq)
    - 反向 squash: scale = 1 / (1 + norm_sq) = 1 - norm_sq / (1 + norm_sq)
    
    特性：
    - 当向量范数很小时，scale 接近 1（与原始 squash 相反）
    - 当向量范数很大时，scale 接近 0（与原始 squash 相反）
    """
    def forward(self, x, dim=-1):
        norm_sq = torch.sum(x ** 2, dim=dim, keepdim=True)
        norm = torch.sqrt(norm_sq + 1e-8)
        # 反向 scale: 1 / (1 + norm_sq) = 1 - norm_sq / (1 + norm_sq)
        scale = 1.0 / (1.0 + norm_sq)
        return scale * x / norm

# ==============================
# 1.5 Margin Loss（CapsNet 风格）
# ==============================
def margin_loss(v_length, labels, m_plus=0.9, m_minus=0.1, lambda_val=0.5):
    """
    CapsNet 的 Margin Loss
    
    Args:
        v_length: [B, num_classes] - 胶囊向量的长度（范数）
        labels: [B, num_classes] - one-hot 编码的标签
        m_plus: 正类边界（默认 0.9）
        m_minus: 负类边界（默认 0.1）
        lambda_val: 负类权重（默认 0.5）
    
    Returns:
        loss: scalar - margin loss
    """
    # T_k: 真实标签（1 表示存在，0 表示不存在）
    T_k = labels.float()  # [B, num_classes]
    
    # 正类损失：max(0, m+ - ||v_k||)^2
    positive_loss = T_k * torch.clamp(m_plus - v_length, min=0.0) ** 2
    
    # 负类损失：max(0, ||v_k|| - m-)^2
    negative_loss = lambda_val * (1.0 - T_k) * torch.clamp(v_length - m_minus, min=0.0) ** 2
    
    # 总损失：对所有类别和样本求和
    loss = (positive_loss + negative_loss).sum(dim=1).mean()
    
    return loss

# ==============================
# 2. 初级胶囊层（确定性）
# ==============================
class PrimaryCapsules(nn.Module):
    def __init__(self, in_channels=256, out_caps=32, caps_dim=8, kernel_size=9):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_caps * caps_dim, kernel_size, stride=2)
        self.out_caps = out_caps
        self.caps_dim = caps_dim
        self.squash = Squash()

    def forward(self, x):
        batch = x.shape[0]
        u = self.conv(x)  # [B, 256, 6, 6]
        u = u.view(batch, self.out_caps, self.caps_dim, -1)  # [B, 32, 8, 36]
        u = u.permute(0, 1, 3, 2).contiguous()  # [B, 32, 36, 8]
        u = self.squash(u)
        return u.view(batch, -1, self.caps_dim)  # [B, 1152, 8]


# ==============================
# 2.5 改进的 Context Encoder（考虑空间和通道信息）
# ==============================
class ChannelAttention(nn.Module):
    """通道注意力模块 - 关注重要的通道特征"""
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return x * out.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """空间注意力模块 - 关注重要的空间位置"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)


class EnhancedContextEncoder(nn.Module):
    """
    增强版 Context Encoder
    考虑空间信息和通道信息，逐步压缩
    
    输入: [B, primary_caps_num, 8] (primary_caps_num = 32个胶囊类型 × 空间位置数)
    输出: [B, 256] (上下文特征)
    
    设计思路：
    1. 通道扩展：从8维扩展到64维，增强表达能力
    2. 空间压缩：使用卷积逐步压缩空间维度，保留重要空间信息
    3. 注意力机制：使用通道注意力和空间注意力增强重要特征
    4. 全局池化：最终压缩为一维特征向量
    """
    def __init__(self, context_dim=256, primary_caps_num=1152):
        super().__init__()
        self.context_dim = context_dim
        self.primary_caps_num = primary_caps_num
        
        # 第一步：通道扩展
        self.channel_expand = nn.Sequential(
            nn.Conv1d(8, 64, kernel_size=1),  # [B, 64, primary_caps_num]
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # 第二步：空间压缩（逐步压缩，保留空间关系）
        # 动态计算压缩后的尺寸
        # 第一次压缩：stride=4, kernel=9, padding=4
        # output_size = (input_size + 2*padding - kernel) // stride + 1
        # = (primary_caps_num + 8 - 9) // 4 + 1 = (primary_caps_num - 1) // 4 + 1
        first_compress_size = (primary_caps_num - 1) // 4 + 1
        
        # 第二次压缩：stride=3, kernel=7, padding=3
        # output_size = (first_compress_size + 6 - 7) // 3 + 1 = (first_compress_size - 1) // 3 + 1
        second_compress_size = (first_compress_size - 1) // 3 + 1
        
        self.spatial_compress = nn.Sequential(
            # 第一次压缩
            nn.Conv1d(64, 128, kernel_size=9, stride=4, padding=4),  # [B, 128, first_compress_size]
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # 第二次压缩
            nn.Conv1d(128, 256, kernel_size=7, stride=3, padding=3),  # [B, 256, second_compress_size]
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # 第三步：转换为2D以使用空间和通道注意力
        # 将 second_compress_size 重塑为 2D，找到合适的因子分解
        # 例如：96 = 8 * 12, 48 = 8 * 6, 64 = 8 * 8
        # 我们尝试找到一个接近平方数的分解
        self.second_compress_size = second_compress_size
        # 找到最接近的因子分解
        sqrt_size = int(np.sqrt(second_compress_size))
        h_dim = sqrt_size
        w_dim = (second_compress_size + h_dim - 1) // h_dim  # 向上取整
        
        self.channel_attention = ChannelAttention(256, reduction=8)
        self.spatial_attention = SpatialAttention(kernel_size=7)
        self.h_dim = h_dim
        self.w_dim = w_dim
        
        # 第四步：最终压缩
        # 使用自适应池化确保输出尺寸一致
        self.final_compress = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=5, stride=2, padding=2),  # 进一步压缩
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        # 第五步：全局池化和最终投影
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # [B, 256, 1]
        self.final_proj = nn.Sequential(
            nn.Linear(256, context_dim),
            nn.LayerNorm(context_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, primary_caps_num, 8] - primary capsules
        Returns:
            context: [B, context_dim] - context features
        """
        B = x.shape[0]
        # [B, primary_caps_num, 8] -> [B, 8, primary_caps_num]
        x = x.transpose(1, 2)  # [B, 8, primary_caps_num]
        
        # 通道扩展
        x = self.channel_expand(x)  # [B, 64, primary_caps_num]
        
        # 空间压缩
        x = self.spatial_compress(x)  # [B, 256, second_compress_size]
        
        # 转换为2D以使用注意力机制
        # [B, 256, second_compress_size] -> [B, 256, h_dim, w_dim]
        # 如果尺寸不匹配，使用padding或裁剪
        current_size = x.shape[2]
        target_size = self.h_dim * self.w_dim
        if current_size < target_size:
            # 需要padding
            padding_size = target_size - current_size
            x = F.pad(x, (0, padding_size), mode='constant', value=0)
        elif current_size > target_size:
            # 需要裁剪
            x = x[:, :, :target_size]
        
        x_2d = x.view(B, 256, self.h_dim, self.w_dim)  # [B, 256, h_dim, w_dim]
        
        # 通道注意力：关注重要的通道特征
        x_2d = self.channel_attention(x_2d)  # [B, 256, h_dim, w_dim]
        
        # 空间注意力：关注重要的空间位置
        x_2d = self.spatial_attention(x_2d)  # [B, 256, h_dim, w_dim]
        
        # 转换回1D
        x = x_2d.view(B, 256, -1)  # [B, 256, target_size]
        
        # 最终压缩
        x = self.final_compress(x)  # [B, 256, 48]
        
        # 全局池化
        x = self.global_pool(x)  # [B, 256, 1]
        x = x.squeeze(-1)  # [B, 256]
        
        # 最终投影
        context = self.final_proj(x)  # [B, context_dim]
        
        return context

# ==============================
# 3. 信息瓶颈胶囊层（IBCaps）
# ==============================
class IBCapsules(nn.Module):
    def __init__(self, num_classes=10, in_caps_dim=8, latent_dim=16, beta=1e-3, classifier_type='linear', 
                 primary_caps_num=1152, context_encoder_type='default'):
        """
        Args:
            num_classes: 类别数量
            in_caps_dim: 输入胶囊维度
            latent_dim: 潜在空间维度
            beta: KL散度权重
            classifier_type: 分类器类型，'linear'、'squash' 或 'inverse_squash'，默认为 'linear'
            primary_caps_num: Primary capsules的数量（MNIST=1152, CIFAR-10/SVHN=2048）
            context_encoder_type: Context encoder类型，'default' 或 'enhanced'，默认为 'default'
        """
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.beta = beta
        self.classifier_type = classifier_type
        self.primary_caps_num = primary_caps_num

        # 全局上下文编码器（替代路由）
        if context_encoder_type == 'enhanced':
            # 使用增强版 Context Encoder
            self.context_encoder = EnhancedContextEncoder(
                context_dim=256, 
                primary_caps_num=primary_caps_num
            )
        else:
            # 使用默认的简单 Context Encoder
            self.context_encoder = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),  # [B, primary_caps_num, 8] -> [B, primary_caps_num, 1]
                nn.Flatten(),             # [B, primary_caps_num]
                nn.Linear(primary_caps_num, 512),
                nn.ReLU(),
                nn.Linear(512, 256)
            )

        # 每个类别的编码器（μ, logσ²）
        self.class_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, latent_dim * 2)  # μ and logσ²
            ) for _ in range(num_classes)
        ])

        # 分类头（从 z_j 预测类别）
        if classifier_type == 'linear':
            # 默认方式：使用线性层
            self.classifier = nn.Linear(latent_dim, 1)
            self.squash = None
            self.inverse_squash = None
        elif classifier_type == 'squash':
            # 使用squash方式：对胶囊向量应用squash，然后计算范数
            self.classifier = None  # squash方式不需要线性层
            self.squash = Squash()
            self.inverse_squash = None
        elif classifier_type == 'inverse_squash':
            # 使用inverse_squash方式：对胶囊向量应用inverse_squash，然后计算范数
            self.classifier = None  # inverse_squash方式不需要线性层
            self.squash = None
            self.inverse_squash = InverseSquash()
        else:
            raise ValueError(f"Unsupported classifier_type: {classifier_type}. Must be 'linear', 'squash', or 'inverse_squash'.")

        # 先验分布（标准正态）
        self.prior = Normal(0, 1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, primary_caps):
        """
        primary_caps: [B, 1152, 8]
        returns: 
            class_probs: [B, 10] —— 存在性概率（长度）
            kl_loss: scalar
        """
        B = primary_caps.shape[0]

        # Step 1: 提取全局上下文
        context = self.context_encoder(primary_caps)  # [B, 256]

        # Step 2: 对每个类别生成胶囊
        all_z = []
        total_kl = 0.0

        for j in range(self.num_classes):
            params = self.class_encoders[j](context)  # [B, 2*latent_dim]
            mu, logvar = torch.chunk(params, 2, dim=-1)
            z = self.reparameterize(mu, logvar)       # [B, latent_dim]
            all_z.append(z)

            # KL divergence to prior
            q_dist = Normal(mu, torch.exp(0.5 * logvar))
            kl = kl_divergence(q_dist, self.prior).sum(dim=-1).mean()
            total_kl += kl

        all_z = torch.stack(all_z, dim=1)  # [B, 10, latent_dim]

        # Step 3: 计算每个胶囊的存在性（通过分类置信度）
        if self.classifier_type == 'linear':
            # 线性方式：将每个 z_j 送入线性分类头
            logits = torch.cat([
                self.classifier(all_z[:, j, :]) for j in range(self.num_classes)
            ], dim=1)  # [B, 10]
            class_probs = torch.sigmoid(logits)  # ∈ [0,1]，作为存在性概率
        elif self.classifier_type == 'squash':
            # Squash方式：对每个胶囊向量应用squash，然后计算其范数作为存在性概率
            # all_z: [B, 10, latent_dim]
            # 对每个胶囊应用squash
            all_z_squashed = self.squash(all_z, dim=-1)  # [B, 10, latent_dim]
            # 计算每个胶囊的范数（长度）作为存在性概率
            # 范数已经在[0,1]范围内（因为squash的特性）
            class_probs = torch.norm(all_z_squashed, dim=-1)  # [B, 10]
            # 确保概率在[0,1]范围内（虽然squash已经保证，但为了安全可以加sigmoid）
            # class_probs = torch.sigmoid(class_probs)  # 可选：进一步归一化
        elif self.classifier_type == 'inverse_squash':
            # InverseSquash方式：对每个胶囊向量应用inverse_squash，然后计算其范数作为存在性概率
            # all_z: [B, 10, latent_dim]
            # 对每个胶囊应用inverse_squash
            all_z_inverse_squashed = self.inverse_squash(all_z, dim=-1)  # [B, 10, latent_dim]
            # 计算每个胶囊的范数（长度）作为存在性概率
            # inverse_squash的特性：短向量scale接近1，长向量scale接近0
            class_probs = torch.norm(all_z_inverse_squashed, dim=-1)  # [B, 10]
            # 由于inverse_squash的输出范数可能较小，可以应用sigmoid进行归一化
            class_probs = torch.sigmoid(class_probs * 10.0)  # 乘以10来增强信号，然后sigmoid归一化
        else:
            raise ValueError(f"Unsupported classifier_type: {self.classifier_type}")

        return all_z, class_probs, total_kl / self.num_classes

# ==============================
# 4. 重构解码器（参考 CapsNet 的 Decoder）
# ==============================
class ReconstructionDecoder(nn.Module):
    """
    重构解码器，参考 CapsNet 的 Decoder
    从潜在向量重构原始图像
    """
    def __init__(self, latent_dim=16, num_classes=10, input_width=28, input_height=28, input_channel=1):
        """
        Args:
            latent_dim: 潜在向量维度（每个胶囊的维度）
            num_classes: 类别数量
            input_width: 输入图像宽度
            input_height: 输入图像高度
            input_channel: 输入图像通道数
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.input_width = input_width
        self.input_height = input_height
        self.input_channel = input_channel
        
        # 重构网络：从 latent_dim * num_classes 维重构到图像（参考 CapsNet）
        self.reconstraction_layers = nn.Sequential(
            nn.Linear(latent_dim * num_classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, input_height * input_width * input_channel),
            nn.Sigmoid()  # 输出范围 [0, 1]
        )
    
    def forward(self, all_z, target=None):
        """
        Args:
            all_z: [B, num_classes, latent_dim] - 所有胶囊的潜在向量
            target: [B] - 真实标签（用于masking，如果为None则使用预测标签）
        Returns:
            recon: [B, input_channel, input_height, input_width] - 重构图像
        """
        # 计算每个胶囊的长度（magnitude）
        classes = torch.sqrt((all_z ** 2).sum(2))  # [B, num_classes]
        
        if target is not None:
            # 使用真实标签进行masking
            max_length_indices = target  # [B]
        else:
            # 使用预测标签进行masking（选择长度最大的胶囊）
            _, max_length_indices = classes.max(dim=1)  # [B]
        
        # 创建mask：one-hot向量
        device = all_z.device
        masked = torch.eye(self.num_classes, device=device)[max_length_indices]  # [B, num_classes]
        
        # 应用mask并reshape：all_z [B, num_classes, latent_dim] -> [B, latent_dim*num_classes]
        masked_z = (all_z * masked.unsqueeze(-1)).view(all_z.size(0), -1)  # [B, latent_dim*num_classes]
        
        # 通过重构网络
        reconstructions = self.reconstraction_layers(masked_z)  # [B, H*W*C]
        reconstructions = reconstructions.view(-1, self.input_channel, self.input_height, self.input_width)
        return reconstructions


# ==============================
# 5. 完整模型（无重构）
# ==============================
class IBCapsNet(nn.Module):
    def __init__(self, beta=1e-3, classifier_type='linear', input_width=28, input_height=28, input_channel=1,
                 context_encoder_type='default'):
        """
        Args:
            beta: KL散度权重
            classifier_type: 分类器类型，'linear' 或 'squash'，默认为 'linear'
            input_width: 输入图像宽度（默认28，用于MNIST）
            input_height: 输入图像高度（默认28，用于MNIST）
            input_channel: 输入图像通道数（默认1，用于MNIST）
            context_encoder_type: Context encoder类型，'default' 或 'enhanced'，默认为 'default'
        """
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel, 256, kernel_size=9, stride=1)
        self.primary_caps = PrimaryCapsules()
        
        # 计算primary capsules数量
        conv1_out_size = input_width - 8
        primary_caps_spatial = (conv1_out_size - 9 + 1) // 2
        primary_caps_num = 32 * primary_caps_spatial * primary_caps_spatial
        
        self.ib_caps = IBCapsules(
            beta=beta, 
            classifier_type=classifier_type,
            primary_caps_num=primary_caps_num,
            context_encoder_type=context_encoder_type
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))  # [B, 256, 20, 20]
        primary = self.primary_caps(x)  # [B, 1152, 8]
        z, probs, kl = self.ib_caps(primary)
        return probs, kl

    def loss(self, probs, labels, kl, beta=1e-3):
        """
        计算损失函数
        
        Args:
            probs: [B, num_classes] - 类别概率（对于squash/inverse_squash是范数，对于linear是sigmoid输出）
            labels: [B, num_classes] - one-hot标签
            kl: scalar - KL散度
            beta: KL散度权重
        
        Returns:
            total_loss: 总损失
            ce_loss: 分类损失（CE或Margin Loss）
            kl: KL散度损失
        """
        # 根据 classifier_type 选择不同的损失函数
        if self.ib_caps.classifier_type in ['squash', 'inverse_squash']:
            # 使用 Margin Loss（与 CapsNet 保持一致）
            # probs 是胶囊向量的范数（长度）
            ce_loss = margin_loss(probs, labels, m_plus=0.9, m_minus=0.1, lambda_val=0.5)
        else:
            # 使用标准交叉熵（linear 分类器）
            ce_loss = F.binary_cross_entropy(probs, labels.float())
        
        total_loss = ce_loss + beta * kl
        return total_loss, ce_loss, kl


# ==============================
# 6. 带重构的完整模型（参考 CapsNet）
# ==============================
class IBCapsNetWithRecon(nn.Module):
    """
    带重构功能的 IBCapsNet，参考 CapsNet 的设计
    """
    def __init__(self, latent_dim=16, beta=1e-3, recon_alpha=0.0005, 
                 classifier_type='linear', input_width=28, input_height=28, input_channel=1,
                 context_encoder_type='default'):
        """
        Args:
            latent_dim: 潜在向量维度
            beta: KL散度权重
            recon_alpha: 重构损失权重（参考 CapsNet 的 0.0005）
            classifier_type: 分类器类型，'linear' 或 'squash'
            input_width: 输入图像宽度
            input_height: 输入图像高度
            input_channel: 输入图像通道数
            context_encoder_type: Context encoder类型，'default' 或 'enhanced'，默认为 'default'
        """
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel, 256, kernel_size=9, stride=1)
        self.primary_caps = PrimaryCapsules()
        
        # 计算primary capsules数量
        # conv1: (input_size - 9 + 1) = input_size - 8
        # primary_caps (stride=2, kernel=9): ((input_size - 8) - 9 + 1) // 2 = (input_size - 16) // 2
        # 对于MNIST (28x28): (28-16)//2 = 6 -> 32*6*6 = 1152
        # 对于CIFAR-10/SVHN (32x32): (32-16)//2 = 8 -> 32*8*8 = 2048
        conv1_out_size = input_width - 8  # 假设没有padding
        primary_caps_spatial = (conv1_out_size - 9 + 1) // 2  # primary_caps stride=2, kernel=9
        primary_caps_num = 32 * primary_caps_spatial * primary_caps_spatial
        
        self.ib_caps = IBCapsules(
            latent_dim=latent_dim, 
            beta=beta, 
            classifier_type=classifier_type,
            primary_caps_num=primary_caps_num,
            context_encoder_type=context_encoder_type
        )
        self.decoder = ReconstructionDecoder(
            latent_dim=latent_dim,
            num_classes=self.ib_caps.num_classes,
            input_width=input_width,
            input_height=input_height,
            input_channel=input_channel
        )
        self.recon_alpha = recon_alpha
        self.mse_loss = nn.MSELoss()
    
    def forward(self, x, target=None, do_recon=False):
        """
        Args:
            x: [B, C, H, W] - 输入图像
            target: [B] - 真实标签（用于masking，如果为None则使用预测标签）
            do_recon: 是否进行重构
        Returns:
            probs: [B, num_classes] - 类别概率
            kl: scalar - KL散度损失
            recon_img: [B, C, H, W] 或 None - 重构图像
        """
        x = F.relu(self.conv1(x))  # [B, 256, 20, 20]
        primary = self.primary_caps(x)  # [B, 1152, 8]
        all_z, probs, kl = self.ib_caps(primary)  # all_z: [B, 10, latent_dim]
        
        recon_img = None
        if do_recon:
            # 使用masked方式选择正确的胶囊（参考 CapsNet）
            # 传入所有胶囊 all_z [B, num_classes, latent_dim]，decoder内部会进行masking
            recon_img = self.decoder(all_z, target=target)  # [B, C, H, W]
        
        return probs, kl, recon_img
    
    def loss(self, probs, labels, kl, recon_img=None, original_img=None, 
             beta=None, alpha=None):
        """
        计算总损失：分类损失 + KL损失 + 重构损失
        
        Args:
            probs: [B, num_classes] - 类别概率
            labels: [B, num_classes] - one-hot标签
            kl: scalar - KL散度
            recon_img: [B, C, H, W] 或 None - 重构图像
            original_img: [B, C, H, W] - 原始图像
            beta: KL散度权重（如果为None，使用self.ib_caps.beta）
            alpha: 重构损失权重（如果为None，使用self.recon_alpha）
        Returns:
            total_loss: 总损失
            ce_loss: 分类损失
            kl_loss: KL散度损失
            recon_loss: 重构损失
        """
        # 分类损失：根据 classifier_type 选择不同的损失函数
        if self.ib_caps.classifier_type in ['squash', 'inverse_squash']:
            # 使用 Margin Loss（与 CapsNet 保持一致）
            # probs 是胶囊向量的范数（长度）
            ce_loss = margin_loss(probs, labels, m_plus=0.9, m_minus=0.1, lambda_val=0.5)
        else:
            # 使用标准交叉熵（linear 分类器）
            ce_loss = F.binary_cross_entropy(probs, labels.float())
        
        # KL散度损失
        if beta is None:
            beta = self.ib_caps.beta
        kl_loss = kl
        
        # 重构损失（参考 CapsNet）
        recon_loss = torch.tensor(0.0, device=probs.device)
        if recon_img is not None and original_img is not None:
            if alpha is None:
                alpha = self.recon_alpha
            # 使用MSE损失，并乘以权重（参考 CapsNet 的 0.0005）
            recon_loss = self.mse_loss(
                recon_img.view(recon_img.size(0), -1),
                original_img.view(original_img.size(0), -1)
            ) * alpha
        
        # 总损失
        total_loss = ce_loss + beta * kl_loss + recon_loss
        
        return total_loss, ce_loss, kl_loss, recon_loss
    
def train(model, optimizer, train_loader, epoch, n_epochs, logger=None):
    """训练函数（支持重构模型）"""
    model.train()
    # 检测是否使用重构模型
    use_recon = hasattr(model, 'recon_alpha')
    
    total_loss = 0.0
    total_ce_loss = 0.0
    total_kl_loss = 0.0
    total_recon_loss = 0.0
    valid_batches = 0
    total_correct = 0
    total_samples = 0
    
    for batch_id, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}/{n_epochs}')):
        data, target = data.to(next(model.parameters()).device), target.to(next(model.parameters()).device)
        labels = F.one_hot(target, 10).float()

        optimizer.zero_grad()
        
        # 根据模型类型调用不同的forward和loss
        if use_recon:
            # 训练时启用重构
            probs, kl, recon_img = model(data, target=target, do_recon=True)
            loss, ce, kl_val, recon_loss_val = model.loss(
                probs, labels, kl, recon_img, data, 
                beta=model.ib_caps.beta, 
                alpha=model.recon_alpha
            )
        else:
            # 不使用重构
            probs, kl = model(data)
            loss, ce, kl_val = model.loss(probs, labels, kl, beta=model.ib_caps.beta)
            recon_loss_val = torch.tensor(0.0, device=data.device)
        
        # 检查loss是否为NaN或Inf
        if torch.isnan(loss) or torch.isinf(loss):
            warning_msg = f"Warning: NaN/Inf loss detected at batch {batch_id}, skipping..."
            tqdm.write(warning_msg)
            if logger:
                logger.warning(warning_msg)
            continue
        
        loss.backward()
        
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 计算准确率
        pred = probs.argmax(dim=1)
        correct = pred.eq(target).sum().item()
        total_correct += correct
        total_samples += target.size(0)
        
        # 累加损失
        loss_value = loss.item()
        ce_value = ce.item()
        kl_value = kl_val.item()
        recon_value = recon_loss_val.item() if recon_loss_val > 0 else 0.0
        
        if not (np.isnan(loss_value) or np.isinf(loss_value)):
            total_loss += loss_value
            total_ce_loss += ce_value
            total_kl_loss += kl_value
            total_recon_loss += recon_value
            valid_batches += 1
        
        # 每100个batch记录一次
        if batch_id % 100 == 0:
            msg = "Epoch: [{}/{}], Batch: [{}/{}], train accuracy: {:.6f}, loss: {:.6f}, CE: {:.6f}, KL: {:.6f}, Recon: {:.6f}".format(
                epoch, n_epochs, batch_id + 1, len(train_loader),
                correct / float(target.size(0)), loss_value, ce_value, kl_value, recon_value
            )
            tqdm.write(msg)
            if logger:
                logger.info(msg)
    
    # 计算平均loss和accuracy
    avg_loss = total_loss / valid_batches if valid_batches > 0 else 0.0
    avg_ce_loss = total_ce_loss / valid_batches if valid_batches > 0 else 0.0
    avg_kl_loss = total_kl_loss / valid_batches if valid_batches > 0 else 0.0
    avg_recon_loss = total_recon_loss / valid_batches if valid_batches > 0 else 0.0
    avg_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    msg = 'Epoch: [{}/{}], train loss: {:.6f}, train CE loss: {:.6f}, train KL loss: {:.6f}, train Recon loss: {:.6f}, train accuracy: {:.6f}'.format(
        epoch, n_epochs, avg_loss, avg_ce_loss, avg_kl_loss, avg_recon_loss, avg_accuracy
    )
    tqdm.write(msg)
    if logger:
        logger.info(msg)
    
    return avg_loss, avg_ce_loss, avg_kl_loss, avg_recon_loss, avg_accuracy


def test(model, test_loader, epoch, n_epochs, logger=None, do_recon=False):
    """测试函数（支持重构）"""
    model.eval()
    # 检测是否使用重构模型
    use_recon = hasattr(model, 'recon_alpha')
    
    test_loss = 0.0
    test_ce_loss = 0.0
    test_kl_loss = 0.0
    test_recon_loss = 0.0
    correct = 0
    valid_batches = 0
    
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(tqdm(test_loader, desc='Testing')):
            data, target = data.to(next(model.parameters()).device), target.to(next(model.parameters()).device)
            labels = F.one_hot(target, 10).float()

            # 根据模型类型调用不同的forward和loss
            if use_recon:
                probs, kl, recon_img = model(data, target=target, do_recon=do_recon)
                loss, ce, kl_val, recon_loss_val = model.loss(
                    probs, labels, kl, recon_img, data,
                    beta=model.ib_caps.beta,
                    alpha=model.recon_alpha
                )
            else:
                probs, kl = model(data)
                loss, ce, kl_val = model.loss(probs, labels, kl, beta=model.ib_caps.beta)
                recon_loss_val = torch.tensor(0.0, device=data.device)

            loss_value = loss.item()
            ce_value = ce.item()
            kl_value = kl_val.item()
            recon_value = recon_loss_val.item() if recon_loss_val > 0 else 0.0
            
            # 只累加有效的loss
            if not (np.isnan(loss_value) or np.isinf(loss_value)):
                test_loss += loss_value
                test_ce_loss += ce_value
                test_kl_loss += kl_value
                test_recon_loss += recon_value
                valid_batches += 1
            else:
                warning_msg = f"Warning: NaN/Inf loss detected at test batch {batch_id}"
                if logger:
                    logger.warning(warning_msg)
            
            pred = probs.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    accuracy = correct / len(test_loader.dataset)
    avg_loss = test_loss / valid_batches if valid_batches > 0 else 0.0
    avg_ce_loss = test_ce_loss / valid_batches if valid_batches > 0 else 0.0
    avg_kl_loss = test_kl_loss / valid_batches if valid_batches > 0 else 0.0
    avg_recon_loss = test_recon_loss / valid_batches if valid_batches > 0 else 0.0
    
    if use_recon:
        msg = "Epoch: [{}/{}], test accuracy: {:.6f}, test loss: {:.6f}, test CE loss: {:.6f}, test KL loss: {:.6f}, test Recon loss: {:.6f}".format(
            epoch, n_epochs, accuracy, avg_loss, avg_ce_loss, avg_kl_loss, avg_recon_loss
        )
    else:
        msg = "Epoch: [{}/{}], test accuracy: {:.6f}, test loss: {:.6f}, test CE loss: {:.6f}, test KL loss: {:.6f}".format(
            epoch, n_epochs, accuracy, avg_loss, avg_ce_loss, avg_kl_loss
        )
    tqdm.write(msg)
    if logger:
        logger.info(msg)
    
    if use_recon:
        return accuracy, avg_loss, avg_ce_loss, avg_kl_loss, avg_recon_loss
    else:
        return accuracy, avg_loss, avg_ce_loss, avg_kl_loss


def visualize_reconstructions(model, test_loader, epoch, save_dir, num_classes=10, num_samples_per_class=10, logger=None):
    """
    可视化重构结果：每个类别的前num_samples_per_class个样本
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        epoch: 当前epoch
        save_dir: 保存目录
        num_classes: 类别数
        num_samples_per_class: 每个类别的样本数
        logger: 日志记录器
    """
    model.eval()
    device = next(model.parameters()).device
    
    # 检查模型是否支持重构
    use_recon = hasattr(model, 'recon_alpha')
    if not use_recon:
        if logger:
            logger.warning("Model does not support reconstruction, skipping visualization")
        return
    
    # 收集每个类别的样本
    samples_by_class = {i: [] for i in range(num_classes)}
    
    with torch.no_grad():
        for data, target in test_loader:
            # 检查是否已经收集足够的样本
            if all(len(samples_by_class[i]) >= num_samples_per_class for i in range(num_classes)):
                break
            
            data = data.to(device)
            target = target.to(device)
            
            # 对每个样本进行处理
            for i in range(data.size(0)):
                label = target[i].item()
                
                # 如果该类别还没有收集足够的样本
                if len(samples_by_class[label]) < num_samples_per_class:
                    # 获取单个样本
                    single_data = data[i:i+1]  # [1, 1, 28, 28]
                    single_target = target[i:i+1]  # [1]
                    
                    # 进行重构
                    probs, kl, recon_img = model(single_data, target=single_target, do_recon=True)
                    
                    if recon_img is not None:
                        # 将重构结果转换为图像格式
                        original = single_data.cpu().squeeze(0).squeeze(0)  # [28, 28]
                        recon = recon_img.cpu().view(1, 28, 28).squeeze(0)  # [28, 28]
                        pred = probs.argmax(dim=1).cpu().item()
                        
                        samples_by_class[label].append({
                            'original': original,
                            'reconstructed': recon,
                            'true_label': label,
                            'predicted_label': pred,
                            'confidence': probs.max().item()
                        })
    
    # 创建可视化
    os.makedirs(save_dir, exist_ok=True)
    
    # 为每个类别创建可视化
    fig, axes = plt.subplots(num_classes, num_samples_per_class * 2, 
                             figsize=(num_samples_per_class * 2 * 2, num_classes * 2))
    
    if num_classes == 1:
        axes = axes.reshape(1, -1)
    
    for class_idx in range(num_classes):
        samples = samples_by_class[class_idx]
        
        for sample_idx in range(num_samples_per_class):
            col_idx = sample_idx * 2
            
            if sample_idx < len(samples):
                sample = samples[sample_idx]
                
                # 原始图像
                axes[class_idx, col_idx].imshow(sample['original'].numpy(), cmap='gray')
                axes[class_idx, col_idx].set_title(
                    f'Original\nTrue: {sample["true_label"]}\nPred: {sample["predicted_label"]}',
                    fontsize=8
                )
                axes[class_idx, col_idx].axis('off')
                
                # 重构图像
                axes[class_idx, col_idx + 1].imshow(sample['reconstructed'].numpy(), cmap='gray')
                axes[class_idx, col_idx + 1].set_title(
                    f'Reconstructed\nConf: {sample["confidence"]:.3f}',
                    fontsize=8
                )
                axes[class_idx, col_idx + 1].axis('off')
            else:
                # 如果没有足够的样本，显示空白
                axes[class_idx, col_idx].axis('off')
                axes[class_idx, col_idx + 1].axis('off')
        
        # 在每行的最左侧添加类别标签
        if num_samples_per_class > 0:
            axes[class_idx, 0].text(-0.1, 0.5, f'Class {class_idx}', 
                                   transform=axes[class_idx, 0].transAxes,
                                   rotation=90, va='center', ha='right',
                                   fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Reconstruction Visualization - Epoch {epoch}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(save_dir, f'reconstruction_epoch_{epoch}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.info(f"Reconstruction visualization saved to {save_path}")
    else:
        print(f"✓ Reconstruction visualization saved to {save_path}")


# ==============================
# 主函数
# ==============================
if __name__ == '__main__':
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='IBCapsNet Training')
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'mnist-small', 'cifar10', 'cifar10-small'],
                       help='Dataset to use (default: mnist)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs (default: 30)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--samples_per_class', type=int, default=100,
                       help='Samples per class for small datasets (default: 100)')
    parser.add_argument('--latent_dim', type=int, default=16,
                       help='Latent dimension (default: 16)')
    parser.add_argument('--beta', type=float, default=1e-3,
                       help='KL divergence weight (default: 1e-3)')
    parser.add_argument('--recon_alpha', type=float, default=0.0005,
                       help='Reconstruction loss weight (default: 0.0005)')
    parser.add_argument('--classifier_type', type=str, default='linear',
                       choices=['linear', 'squash', 'inverse_squash'],
                       help='Classifier type: linear, squash, or inverse_squash (default: linear)')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save results (default: auto-generated)')
    parser.add_argument('--visualize_recon', action='store_true',
                       help='Visualize reconstructions during training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    USE_CUDA = torch.cuda.is_available()
    
    # 创建保存目录
    if args.save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = f'ibcapsnet_results_{args.dataset}_{args.classifier_type}_{timestamp}'
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建日志目录
    log_dir = os.path.join(save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'ibcapsnet_{args.dataset}_{timestamp}.log')
    csv_file = os.path.join(save_dir, f'training_history_{args.dataset}_{timestamp}.csv')
    json_file = os.path.join(save_dir, f'training_history_{args.dataset}_{timestamp}.json')
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    logger = logging.getLogger(__name__)
    
    # 打印配置信息
    print("=" * 60)
    print("IBCapsNet Training")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    if args.dataset.endswith('-small'):
        print(f"Samples per class: {args.samples_per_class}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Latent dimension: {args.latent_dim}")
    print(f"Beta (KL weight): {args.beta}")
    print(f"Recon alpha: {args.recon_alpha}")
    print(f"Classifier type: {args.classifier_type}")
    print(f"CUDA available: {USE_CUDA}")
    print(f"Random seed: {args.seed}")
    print(f"Save directory: {save_dir}")
    print(f"Log file: {log_file}")
    print(f"CSV file: {csv_file}")
    print("=" * 60)
    
    logger.info("=" * 60)
    logger.info("IBCapsNet Training")
    logger.info("=" * 60)
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    logger.info(f"CUDA available: {USE_CUDA}")
    logger.info("=" * 60)
    
    # 加载数据集
    if args.dataset.endswith('-small'):
        dataset = Dataset(args.dataset, args.batch_size, samples_per_class=args.samples_per_class)
    else:
        dataset = Dataset(args.dataset, args.batch_size)
    
    train_loader = dataset.train_loader
    test_loader = dataset.test_loader
    
    # 确定输入尺寸（用于模型初始化）
    if args.dataset in ['mnist', 'mnist-small']:
        input_width, input_height, input_channel = 28, 28, 1
    elif args.dataset in ['cifar10', 'cifar10-small']:
        input_width, input_height, input_channel = 32, 32, 3
    else:
        input_width, input_height, input_channel = 28, 28, 1  # 默认
    
    # 创建模型
    model = IBCapsNetWithRecon(
        latent_dim=args.latent_dim,
        beta=args.beta,
        recon_alpha=args.recon_alpha,
        classifier_type=args.classifier_type,
        input_width=input_width,
        input_height=input_height,
        input_channel=input_channel
    )
    
    if USE_CUDA:
        model = model.cuda()
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print()
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 创建CSV文件并写入表头
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train Loss', 'Train CE Loss', 'Train KL Loss', 
                        'Train Recon Loss', 'Train Accuracy', 'Test Loss', 
                        'Test CE Loss', 'Test KL Loss', 'Test Recon Loss', 
                        'Test Accuracy', 'Best Accuracy'])
    
    # 训练和测试
    best_accuracy = 0.0
    best_epoch = 0
    training_history = []
    
    # 创建可视化目录
    if args.visualize_recon:
        vis_dir = os.path.join(save_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
    
    print("\nStarting training...")
    logger.info("Starting training...")
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss, train_ce, train_kl, train_recon, train_acc = train(
            model, optimizer, train_loader, epoch, args.epochs, logger
        )
        
        # 测试
        test_acc, test_loss, test_ce, test_kl, test_recon = test(
            model, test_loader, epoch, args.epochs, logger, do_recon=False
        )
        
        # 记录历史
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_ce_loss': train_ce,
            'train_kl_loss': train_kl,
            'train_recon_loss': train_recon,
            'train_accuracy': train_acc,
            'test_loss': test_loss,
            'test_ce_loss': test_ce,
            'test_kl_loss': test_kl,
            'test_recon_loss': test_recon,
            'test_accuracy': test_acc,
            'best_accuracy': best_accuracy
        })
        
        # 保存到CSV
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, train_loss, train_ce, train_kl, train_recon, train_acc,
                test_loss, test_ce, test_kl, test_recon, test_acc, best_accuracy
            ])
        
        # 更新最佳准确率
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_epoch = epoch
            # 保存最佳模型
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"  -> New best model saved! Accuracy: {best_accuracy:.4f} (Epoch {epoch})")
        
        # 可视化重构（如果启用）
        if args.visualize_recon and epoch % 5 == 0:  # 每5个epoch可视化一次
            visualize_reconstructions(
                model, test_loader, epoch, vis_dir, 
                num_classes=10, num_samples_per_class=5, logger=logger
            )
    
    # 保存训练历史到JSON
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=2, default=str)
    
    # 保存最终模型
    final_model_path = os.path.join(save_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    
    # 打印最终结果
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best Accuracy: {best_accuracy:.4f} (Epoch {best_epoch})")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"\nResults saved to: {save_dir}")
    print(f"  - Best model: {os.path.join(save_dir, 'best_model.pth')}")
    print(f"  - Final model: {final_model_path}")
    print(f"  - Training history (CSV): {csv_file}")
    print(f"  - Training history (JSON): {json_file}")
    print(f"  - Log file: {log_file}")
    if args.visualize_recon:
        print(f"  - Visualizations: {vis_dir}")
    print("=" * 60)
    
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Best Accuracy: {best_accuracy:.4f} (Epoch {best_epoch})")
    logger.info(f"Final Test Accuracy: {test_acc:.4f}")
    logger.info(f"Results saved to: {save_dir}")
    logger.info("=" * 60)
