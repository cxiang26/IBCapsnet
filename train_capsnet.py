import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from capsnet import CapsNet
from data_loader import Dataset
from tqdm import tqdm
import logging
import csv
import os
from datetime import datetime

USE_CUDA = True if torch.cuda.is_available() else False
BATCH_SIZE = 128
N_EPOCHS = 30
LEARNING_RATE = 0.01
MOMENTUM = 0.9

'''
Config class to determine the parameters for capsule net
'''


class Config:
    def __init__(self, dataset='mnist'):
        if dataset == 'mnist':
            # CNN (cnn)
            self.cnn_in_channels = 1
            self.cnn_out_channels = 256
            self.cnn_kernel_size = 9

            # Primary Capsule (pc)
            self.pc_num_capsules = 8
            self.pc_in_channels = 256
            self.pc_out_channels = 32
            self.pc_kernel_size = 9
            self.pc_num_routes = 32 * 6 * 6

            # Digit Capsule (dc)
            self.dc_num_capsules = 10
            self.dc_num_routes = 32 * 6 * 6
            self.dc_in_channels = 8
            self.dc_out_channels = 16

            # Decoder
            self.input_width = 28
            self.input_height = 28

        elif dataset == 'mnist-small':
            # MNIST小样本训练：配置与mnist相同，只是训练样本数不同
            # CNN (cnn)
            self.cnn_in_channels = 1
            self.cnn_out_channels = 256
            self.cnn_kernel_size = 9

            # Primary Capsule (pc)
            self.pc_num_capsules = 8
            self.pc_in_channels = 256
            self.pc_out_channels = 32
            self.pc_kernel_size = 9
            self.pc_num_routes = 32 * 6 * 6

            # Digit Capsule (dc)
            self.dc_num_capsules = 10
            self.dc_num_routes = 32 * 6 * 6
            self.dc_in_channels = 8
            self.dc_out_channels = 16

            # Decoder
            self.input_width = 28
            self.input_height = 28

        elif dataset == 'cifar10':
            # CNN (cnn)
            self.cnn_in_channels = 3
            self.cnn_out_channels = 256
            self.cnn_kernel_size = 9

            # Primary Capsule (pc)
            self.pc_num_capsules = 8
            self.pc_in_channels = 256
            self.pc_out_channels = 32
            self.pc_kernel_size = 9
            self.pc_num_routes = 32 * 8 * 8

            # Digit Capsule (dc)
            self.dc_num_capsules = 10
            self.dc_num_routes = 32 * 8 * 8
            self.dc_in_channels = 8
            self.dc_out_channels = 16

            # Decoder
            self.input_width = 32
            self.input_height = 32

        elif dataset == 'cifar10-small':
            # CIFAR-10小样本训练：配置与cifar10相同，只是训练样本数不同
            # CNN (cnn)
            self.cnn_in_channels = 3
            self.cnn_out_channels = 256
            self.cnn_kernel_size = 9

            # Primary Capsule (pc)
            self.pc_num_capsules = 8
            self.pc_in_channels = 256
            self.pc_out_channels = 32
            self.pc_kernel_size = 9
            self.pc_num_routes = 32 * 8 * 8

            # Digit Capsule (dc)
            self.dc_num_capsules = 10
            self.dc_num_routes = 32 * 8 * 8
            self.dc_in_channels = 8
            self.dc_out_channels = 16

            # Decoder
            self.input_width = 32
            self.input_height = 32

        elif dataset == 'office-caltech':
            # CNN (cnn)
            self.cnn_in_channels = 3
            self.cnn_out_channels = 256
            self.cnn_kernel_size = 9

            # Primary Capsule (pc)
            self.pc_num_capsules = 8
            self.pc_in_channels = 256
            self.pc_out_channels = 32
            self.pc_kernel_size = 9
            # 对于224x224输入，经过conv(9x9, stride=1) -> 216x216
            # 经过primary_caps(9x9, stride=2) -> 104x104
            self.pc_num_routes = 32 * 104 * 104

            # Digit Capsule (dc)
            self.dc_num_capsules = 10  # Office-Caltech有10个类别
            self.dc_num_routes = 32 * 104 * 104
            self.dc_in_channels = 8
            self.dc_out_channels = 16

            # Decoder
            self.input_width = 224
            self.input_height = 224

        elif dataset == 'office31':
            # CNN (cnn)
            self.cnn_in_channels = 3
            self.cnn_out_channels = 256
            self.cnn_kernel_size = 9

            # Primary Capsule (pc)
            self.pc_num_capsules = 8
            self.pc_in_channels = 256
            self.pc_out_channels = 32
            self.pc_kernel_size = 9
            # 对于224x224输入，经过conv(9x9, stride=1) -> 216x216
            # 经过primary_caps(9x9, stride=2) -> 104x104
            self.pc_num_routes = 32 * 104 * 104

            # Digit Capsule (dc)
            self.dc_num_capsules = 31  # Office-31有31个类别
            self.dc_num_routes = 32 * 104 * 104
            self.dc_in_channels = 8
            self.dc_out_channels = 16

            # Decoder
            self.input_width = 224
            self.input_height = 224

        elif dataset == 'svhn':
            # SVHN数据集：32x32 RGB图像，10个类别
            # CNN (cnn)
            self.cnn_in_channels = 3
            self.cnn_out_channels = 256
            self.cnn_kernel_size = 9

            # Primary Capsule (pc)
            self.pc_num_capsules = 8
            self.pc_in_channels = 256
            self.pc_out_channels = 32
            self.pc_kernel_size = 9
            self.pc_num_routes = 32 * 8 * 8

            # Digit Capsule (dc)
            self.dc_num_capsules = 10
            self.dc_num_routes = 32 * 8 * 8
            self.dc_in_channels = 8
            self.dc_out_channels = 16

            # Decoder
            self.input_width = 32
            self.input_height = 32

        elif dataset == 'svhn-small':
            # SVHN小样本训练：配置与svhn相同，只是训练样本数不同
            # CNN (cnn)
            self.cnn_in_channels = 3
            self.cnn_out_channels = 256
            self.cnn_kernel_size = 9

            # Primary Capsule (pc)
            self.pc_num_capsules = 8
            self.pc_in_channels = 256
            self.pc_out_channels = 32
            self.pc_kernel_size = 9
            self.pc_num_routes = 32 * 8 * 8

            # Digit Capsule (dc)
            self.dc_num_capsules = 10
            self.dc_num_routes = 32 * 8 * 8
            self.dc_in_channels = 8
            self.dc_out_channels = 16

            # Decoder
            self.input_width = 32
            self.input_height = 32

        elif dataset == 'your own dataset':
            pass


def train(model, optimizer, train_loader, epoch, logger=None):
    capsule_net = model
    capsule_net.train()
    n_batch = len(list(enumerate(train_loader)))
    total_loss = 0
    valid_batches = 0
    total_correct = 0
    total_samples = 0
    
    for batch_id, (data, target) in enumerate(tqdm(train_loader)):

        target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)

        if USE_CUDA:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output, reconstructions, masked = capsule_net(data)
        loss = capsule_net.loss(data, output, target, reconstructions)
        
        # 检查loss是否为NaN或Inf
        if torch.isnan(loss) or torch.isinf(loss):
            warning_msg = f"Warning: NaN/Inf loss detected at batch {batch_id}, skipping..."
            tqdm.write(warning_msg)
            if logger:
                logger.warning(warning_msg)
            continue
        
        loss.backward()
        
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(capsule_net.parameters(), max_norm=1.0)
        
        optimizer.step()
        correct = sum(np.argmax(masked.data.cpu().numpy(), 1) == np.argmax(target.data.cpu().numpy(), 1))
        total_correct += correct
        total_samples += BATCH_SIZE
        
        train_loss = loss.item()
        # 只累加有效的loss
        if not (np.isnan(train_loss) or np.isinf(train_loss)):
            total_loss += train_loss
            valid_batches += 1
        
        if batch_id % 100 == 0:
            msg = "Epoch: [{}/{}], Batch: [{}/{}], train accuracy: {:.6f}, loss: {:.6f}".format(
                epoch,
                N_EPOCHS,
                batch_id + 1,
                n_batch,
                correct / float(BATCH_SIZE),
                train_loss / float(BATCH_SIZE)
            )
            tqdm.write(msg)
            if logger:
                logger.info(msg)
    
    # 计算平均loss和accuracy
    avg_loss = total_loss / valid_batches if valid_batches > 0 else 0.0
    avg_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    msg = 'Epoch: [{}/{}], train loss: {:.6f}, train accuracy: {:.6f}'.format(
        epoch, N_EPOCHS, avg_loss, avg_accuracy
    )
    tqdm.write(msg)
    if logger:
        logger.info(msg)
    
    return avg_loss, avg_accuracy


def test(capsule_net, test_loader, epoch, logger=None):
    capsule_net.eval()
    test_loss = 0
    correct = 0
    valid_batches = 0
    
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(test_loader):

            target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
            data, target = Variable(data), Variable(target)

            if USE_CUDA:
                data, target = data.cuda(), target.cuda()

            output, reconstructions, masked = capsule_net(data)
            loss = capsule_net.loss(data, output, target, reconstructions)

            loss_value = loss.item()
            # 只累加有效的loss
            if not (np.isnan(loss_value) or np.isinf(loss_value)):
                test_loss += loss_value
                valid_batches += 1
            else:
                warning_msg = f"Warning: NaN/Inf loss detected at test batch {batch_id}"
                if logger:
                    logger.warning(warning_msg)
            
            correct += sum(np.argmax(masked.data.cpu().numpy(), 1) ==
                           np.argmax(target.data.cpu().numpy(), 1))

    accuracy = correct / len(test_loader.dataset)
    avg_loss = test_loss / valid_batches if valid_batches > 0 else 0.0
    msg = "Epoch: [{}/{}], test accuracy: {:.6f}, loss: {:.6f}".format(
        epoch, N_EPOCHS, accuracy, avg_loss
    )
    tqdm.write(msg)
    if logger:
        logger.info(msg)
    
    return accuracy, avg_loss


if __name__ == '__main__':
    torch.manual_seed(1)
    if USE_CUDA:
        torch.cuda.manual_seed(1)
    
    # 配置
    # 支持的数据集: 'mnist', 'mnist-small', 'cifar10', 'cifar10-small', 'office-caltech', 'office31'
    dataset = 'mnist'  # 小样本训练：每个类别100个样本
    samples_per_class = 100  # 小样本训练时每个类别的样本数（仅对-small数据集有效）
    
    # 创建日志目录
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'capsnet_{dataset}_{timestamp}.log')
    csv_file = os.path.join(log_dir, f'training_history_{dataset}_{timestamp}.csv')
    
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
    
    print("=" * 60)
    print("Capsule Network Training")
    print("=" * 60)
    print(f"Dataset: {dataset}")
    if dataset.endswith('-small'):
        print(f"Samples per class: {samples_per_class}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {N_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"CUDA available: {USE_CUDA}")
    print(f"Log file: {log_file}")
    print(f"CSV file: {csv_file}")
    print("=" * 60)
    
    logger.info("=" * 60)
    logger.info("Capsule Network Training")
    logger.info("=" * 60)
    logger.info(f"Dataset: {dataset}")
    if dataset.endswith('-small'):
        logger.info(f"Samples per class: {samples_per_class}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Epochs: {N_EPOCHS}")
    logger.info(f"Learning rate: {LEARNING_RATE}")
    logger.info(f"CUDA available: {USE_CUDA}")
    logger.info("=" * 60)
    
    config = Config(dataset)
    # 对于-small数据集，传递samples_per_class参数
    if dataset.endswith('-small'):
        mnist = Dataset(dataset, BATCH_SIZE, samples_per_class=samples_per_class)
    else:
        mnist = Dataset(dataset, BATCH_SIZE)

    capsule_net = CapsNet(config)
    capsule_net = torch.nn.DataParallel(capsule_net)
    if USE_CUDA:
        capsule_net = capsule_net.cuda()
    capsule_net = capsule_net.module

    # 打印模型参数数量
    total_params = sum(p.numel() for p in capsule_net.parameters())
    trainable_params = sum(p.numel() for p in capsule_net.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.Adam(capsule_net.parameters(), lr=LEARNING_RATE)
    
    # 创建CSV文件并写入表头
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy', 'Best Accuracy'])
    
    # 训练和测试
    best_accuracy = 0.0
    training_history = []
    
    for e in range(1, N_EPOCHS + 1):
        train_loss, train_acc = train(capsule_net, optimizer, mnist.train_loader, e, logger)
        test_acc, test_loss = test(capsule_net, mnist.test_loader, e, logger)
        
        # 记录训练历史
        training_history.append({
            'epoch': e,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'best_accuracy': best_accuracy
        })
        
        # 保存到CSV
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([e, train_loss, train_acc, test_loss, test_acc, best_accuracy])
        
        # 保存最佳模型
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            model_path = f'capsnet_{dataset}_best.pth'
            torch.save(capsule_net.state_dict(), model_path)
            msg = f"New best model saved! Accuracy: {best_accuracy:.6f}, Model: {model_path}"
            print(msg)
            logger.info(msg)
    
    # 保存最终模型
    final_model_path = f'capsnet_{dataset}_final.pth'
    torch.save(capsule_net.state_dict(), final_model_path)
    logger.info(f"Final model saved: {final_model_path}")
    
    print("=" * 60)
    print(f"Training completed! Best accuracy: {best_accuracy:.6f}")
    print(f"Log file saved: {log_file}")
    print(f"Training history saved: {csv_file}")
    print("=" * 60)
    
    logger.info("=" * 60)
    logger.info(f"Training completed! Best accuracy: {best_accuracy:.6f}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Training history: {csv_file}")
    logger.info("=" * 60)
