import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
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


class LeNet(nn.Module):
    """
    LeNet-5 architecture for MNIST classification
    """
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        # 第一个卷积层: 1 -> 6 channels, 5x5 kernel
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        # 第一个池化层: 2x2
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 第二个卷积层: 6 -> 16 channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # 第二个池化层: 2x2
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 全连接层
        # 经过两次池化后，28x28 -> 14x14 -> 5x5 (因为conv2没有padding)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        # Conv1 + ReLU + Pool1
        x = self.pool1(F.relu(self.conv1(x)))
        # Conv2 + ReLU + Pool2
        x = self.pool2(F.relu(self.conv2(x)))
        # Flatten
        x = x.view(-1, 16 * 5 * 5)
        # FC layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(model, optimizer, train_loader, epoch, logger=None):
    """
    训练函数
    """
    model.train()
    n_batch = len(list(enumerate(train_loader)))
    total_loss = 0
    valid_batches = 0
    total_correct = 0
    total_samples = 0
    
    criterion = nn.CrossEntropyLoss()
    
    for batch_id, (data, target) in enumerate(tqdm(train_loader)):
        if USE_CUDA:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
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
        pred = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).sum().item()
        total_correct += correct
        total_samples += data.size(0)
        
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
                correct / float(data.size(0)),
                train_loss
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


def test(model, test_loader, epoch, logger=None):
    """
    测试函数
    """
    model.eval()
    test_loss = 0
    correct = 0
    valid_batches = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(test_loader):
            if USE_CUDA:
                data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = criterion(output, target)

            loss_value = loss.item()
            # 只累加有效的loss
            if not (np.isnan(loss_value) or np.isinf(loss_value)):
                test_loss += loss_value
                valid_batches += 1
            else:
                warning_msg = f"Warning: NaN/Inf loss detected at test batch {batch_id}"
                if logger:
                    logger.warning(warning_msg)
            
            # 计算准确率
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()

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
    # 支持的数据集: 'mnist', 'mnist-small', 'cifar10', 'cifar10-small'
    dataset = 'mnist'  # 与capsnet保持一致
    samples_per_class = 100  # 小样本训练时每个类别的样本数（仅对-small数据集有效）
    
    # 创建日志目录
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'lenet_{dataset}_{timestamp}.log')
    csv_file = os.path.join(log_dir, f'training_history_lenet_{dataset}_{timestamp}.csv')
    
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
    print("LeNet Training")
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
    logger.info("LeNet Training")
    logger.info("=" * 60)
    logger.info(f"Dataset: {dataset}")
    if dataset.endswith('-small'):
        logger.info(f"Samples per class: {samples_per_class}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Epochs: {N_EPOCHS}")
    logger.info(f"Learning rate: {LEARNING_RATE}")
    logger.info(f"CUDA available: {USE_CUDA}")
    logger.info("=" * 60)
    
    # 加载数据集（与capsnet保持一致）
    if dataset.endswith('-small'):
        data_loader = Dataset(dataset, BATCH_SIZE, samples_per_class=samples_per_class)
    else:
        data_loader = Dataset(dataset, BATCH_SIZE)

    # 创建LeNet模型
    lenet = LeNet(num_classes=10)
    if USE_CUDA:
        lenet = lenet.cuda()

    # 打印模型参数数量
    total_params = sum(p.numel() for p in lenet.parameters())
    trainable_params = sum(p.numel() for p in lenet.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print()
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # 使用Adam优化器（与capsnet保持一致）
    optimizer = torch.optim.Adam(lenet.parameters(), lr=LEARNING_RATE)
    
    # 创建CSV文件并写入表头
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy', 'Best Accuracy'])
    
    # 训练和测试
    best_accuracy = 0.0
    training_history = []
    
    for e in range(1, N_EPOCHS + 1):
        train_loss, train_acc = train(lenet, optimizer, data_loader.train_loader, e, logger)
        test_acc, test_loss = test(lenet, data_loader.test_loader, e, logger)
        
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
            model_path = f'lenet_{dataset}_best.pth'
            torch.save(lenet.state_dict(), model_path)
            msg = f"New best model saved! Accuracy: {best_accuracy:.6f}, Model: {model_path}"
            print(msg)
            logger.info(msg)
    
    # 保存最终模型
    final_model_path = f'lenet_{dataset}_final.pth'
    torch.save(lenet.state_dict(), final_model_path)
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

