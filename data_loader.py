import torch
from torchvision import datasets, transforms
import os
import random
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision.datasets import ImageFolder


def create_small_dataset(full_dataset, samples_per_class=100, random_seed=42):
    """
    从完整数据集中采样每个类别的指定数量样本
    
    Args:
        full_dataset: 完整的数据集（torchvision dataset）
        samples_per_class: 每个类别采样的样本数，默认100
        random_seed: 随机种子，保证可复现
    
    Returns:
        Subset: 采样后的数据集子集
    """
    # 设置随机种子
    random.seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.manual_seed(random_seed)
    
    # 获取所有类别
    num_classes = len(full_dataset.classes) if hasattr(full_dataset, 'classes') else 10
    
    # 按类别组织索引
    indices_by_class = {i: [] for i in range(num_classes)}
    for idx in range(len(full_dataset)):
        _, label = full_dataset[idx]
        if isinstance(label, torch.Tensor):
            label = label.item()
        indices_by_class[label].append(idx)
    
    # 从每个类别中采样
    selected_indices = []
    for class_idx in range(num_classes):
        class_indices = indices_by_class[class_idx]
        if len(class_indices) > samples_per_class:
            # 随机采样
            sampled = random.sample(class_indices, samples_per_class)
        else:
            # 如果类别样本数不足，使用全部样本
            sampled = class_indices
            print(f"Warning: Class {class_idx} has only {len(class_indices)} samples, using all.")
        selected_indices.extend(sampled)
    
    # 打乱顺序
    random.shuffle(selected_indices)
    
    # 创建子集
    subset = Subset(full_dataset, selected_indices)
    
    print(f"Created small dataset: {len(selected_indices)} samples from {num_classes} classes "
          f"({samples_per_class} per class)")
    
    return subset


class Dataset:
    def __init__(self, dataset, _batch_size, samples_per_class=100):
        """
        Args:
            dataset: 数据集名称
            _batch_size: 批次大小
            samples_per_class: 小样本训练时每个类别的样本数（仅对-small数据集有效）
        """
        super(Dataset, self).__init__()
        if dataset == 'mnist':
            dataset_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            train_dataset = datasets.MNIST('data/mnist', train=True, download=True,
                                           transform=dataset_transform)
            test_dataset = datasets.MNIST('data/mnist', train=False, download=True,
                                          transform=dataset_transform)

            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=_batch_size, shuffle=True)
            self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=_batch_size, shuffle=False)

        elif dataset == 'mnist-small':
            # MNIST小样本训练：每个类别采样指定数量样本
            dataset_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            # 加载完整数据集
            full_train_dataset = datasets.MNIST(
                'data/mnist', 
                train=True, 
                download=True,
                transform=dataset_transform
            )
            
            # 采样小样本训练集
            train_dataset = create_small_dataset(
                full_train_dataset, 
                samples_per_class=samples_per_class,
                random_seed=42
            )
            
            # 测试集使用完整数据
            test_dataset = datasets.MNIST(
                'data/mnist', 
                train=False, 
                download=True,
                transform=dataset_transform
            )

            self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=_batch_size, shuffle=True
            )
            self.test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=_batch_size, shuffle=False
            )

        elif dataset == 'fashion-mnist':
            # FashionMNIST数据集：28x28灰度图像，10个类别（T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot）
            dataset_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))  # 使用与MNIST相同的标准化参数
            ])

            train_dataset = datasets.FashionMNIST('data/fashion-mnist', train=True, download=True,
                                                 transform=dataset_transform)
            test_dataset = datasets.FashionMNIST('data/fashion-mnist', train=False, download=True,
                                                transform=dataset_transform)

            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=_batch_size, shuffle=True)
            self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=_batch_size, shuffle=False)

        elif dataset == 'fashion-mnist-small':
            # FashionMNIST小样本训练：每个类别采样指定数量样本
            dataset_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            # 加载完整数据集
            full_train_dataset = datasets.FashionMNIST(
                'data/fashion-mnist', 
                train=True, 
                download=True,
                transform=dataset_transform
            )
            
            # 采样小样本训练集
            train_dataset = create_small_dataset(
                full_train_dataset, 
                samples_per_class=samples_per_class,
                random_seed=42
            )
            
            # 测试集使用完整数据
            test_dataset = datasets.FashionMNIST(
                'data/fashion-mnist', 
                train=False, 
                download=True,
                transform=dataset_transform
            )

            self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=_batch_size, shuffle=True
            )
            self.test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=_batch_size, shuffle=False
            )

        elif dataset == 'cifar10':
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            train_dataset = datasets.CIFAR10(
                'data/cifar', train=True, download=True, transform=data_transform)
            test_dataset = datasets.CIFAR10(
                'data/cifar', train=False, download=True, transform=data_transform)

            self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=_batch_size, shuffle=True)
            
            self.test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=_batch_size, shuffle=False)
        
        elif dataset == 'svhn':
            # SVHN数据集：Street View House Numbers，32x32 RGB图像，10个类别
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            train_dataset = datasets.SVHN(
                'data/svhn', split='train', download=True, transform=data_transform)
            test_dataset = datasets.SVHN(
                'data/svhn', split='test', download=True, transform=data_transform)

            self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=_batch_size, shuffle=True)
            
            self.test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=_batch_size, shuffle=False)
        
        elif dataset == 'svhn-small':
            # SVHN小样本训练：每个类别采样指定数量样本
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            # 加载完整训练集
            full_train_dataset = datasets.SVHN(
                'data/svhn', split='train', download=True, transform=data_transform
            )
            
            # 采样小样本训练集
            train_dataset = create_small_dataset(
                full_train_dataset,
                samples_per_class=samples_per_class,
                random_seed=42
            )
            
            # 测试集使用完整数据
            test_dataset = datasets.SVHN(
                'data/svhn', split='test', download=True, transform=data_transform
            )

            self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=_batch_size, shuffle=True
            )
            self.test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=_batch_size, shuffle=False
            )
        
        elif dataset == 'cifar10-small':
            # CIFAR-10小样本训练：每个类别采样指定数量样本
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            # 加载完整训练集
            full_train_dataset = datasets.CIFAR10(
                'data/cifar', train=True, download=True, transform=data_transform
            )
            
            # 采样小样本训练集
            train_dataset = create_small_dataset(
                full_train_dataset,
                samples_per_class=samples_per_class,
                random_seed=42
            )
            
            # 测试集使用完整数据
            test_dataset = datasets.CIFAR10(
                'data/cifar', train=False, download=True, transform=data_transform
            )

            self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=_batch_size, shuffle=True
            )
            self.test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=_batch_size, shuffle=False
            )
        
        elif dataset == 'office-caltech':
            # Office-Caltech数据集：4个域（amazon, caltech, dslr, webcam），10个类别
            # 数据集路径：data/office-caltech/
            # 默认使用amazon作为训练域，caltech作为测试域
            # 可以通过设置环境变量或修改代码来选择不同的域
            
            data_root = 'data/office-caltech'
            if not os.path.exists(data_root):
                raise FileNotFoundError(
                    f"Office-Caltech dataset not found at {data_root}. "
                    "Please download the dataset and organize it as:\n"
                    "data/office-caltech/\n"
                    "  amazon/\n"
                    "    backpack/\n"
                    "    bike/\n"
                    "    ...\n"
                    "  caltech/\n"
                    "  dslr/\n"
                    "  webcam/\n"
                )
            
            # 图像预处理：Office-Caltech图像通常是224x224或需要resize
            data_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])  # ImageNet标准化
            ])
            
            # 默认配置：使用amazon作为训练，caltech作为测试
            # 可以根据需要修改源域和目标域
            source_domain = os.getenv('OFFICE_CALTECH_SOURCE', 'amazon')
            target_domain = os.getenv('OFFICE_CALTECH_TARGET', 'caltech')
            
            train_domain_path = os.path.join(data_root, source_domain)
            test_domain_path = os.path.join(data_root, target_domain)
            
            if not os.path.exists(train_domain_path):
                # 如果指定域不存在，尝试使用所有域的组合
                print(f"Warning: {train_domain_path} not found. Using all domains for training.")
                train_datasets = []
                for domain in ['amazon', 'caltech', 'dslr', 'webcam']:
                    domain_path = os.path.join(data_root, domain)
                    if os.path.exists(domain_path):
                        train_datasets.append(ImageFolder(domain_path, transform=data_transform))
                if train_datasets:
                    train_dataset = ConcatDataset(train_datasets)
                else:
                    raise FileNotFoundError(f"No valid domain found in {data_root}")
            else:
                train_dataset = ImageFolder(train_domain_path, transform=data_transform)
            
            if not os.path.exists(test_domain_path):
                # 如果测试域不存在，使用训练域的一部分作为测试
                print(f"Warning: {test_domain_path} not found. Using train domain for testing.")
                test_dataset = train_dataset
            else:
                test_dataset = ImageFolder(test_domain_path, transform=data_transform)
            
            self.train_loader = DataLoader(
                train_dataset, batch_size=_batch_size, shuffle=True, num_workers=4, pin_memory=True
            )
            self.test_loader = DataLoader(
                test_dataset, batch_size=_batch_size, shuffle=False, num_workers=4, pin_memory=True
            )
            
            print(f"Office-Caltech dataset loaded:")
            print(f"  Training domain: {source_domain}, Classes: {len(train_dataset.classes)}")
            print(f"  Testing domain: {target_domain}, Classes: {len(test_dataset.classes)}")
            
        elif dataset == 'office31':
            # Office-31数据集：3个域（amazon, dslr, webcam），31个类别
            # 数据集路径：data/office31/
            # 默认使用amazon作为训练域，webcam作为测试域
            
            data_root = 'data/office31'
            if not os.path.exists(data_root):
                raise FileNotFoundError(
                    f"Office-31 dataset not found at {data_root}. "
                    "Please download the dataset and organize it as:\n"
                    "data/office31/\n"
                    "  amazon/\n"
                    "    backpack/\n"
                    "    bike/\n"
                    "    ...\n"
                    "  dslr/\n"
                    "  webcam/\n"
                )
            
            # 图像预处理：Office-31图像通常是224x224或需要resize
            data_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])  # ImageNet标准化
            ])
            
            # 默认配置：使用amazon作为训练，webcam作为测试
            # 可以根据需要修改源域和目标域
            source_domain = os.getenv('OFFICE31_SOURCE', 'amazon')
            target_domain = os.getenv('OFFICE31_TARGET', 'webcam')
            
            train_domain_path = os.path.join(data_root, source_domain)
            test_domain_path = os.path.join(data_root, target_domain)
            
            if not os.path.exists(train_domain_path):
                # 如果指定域不存在，尝试使用所有域的组合
                print(f"Warning: {train_domain_path} not found. Using all domains for training.")
                train_datasets = []
                for domain in ['amazon', 'dslr', 'webcam']:
                    domain_path = os.path.join(data_root, domain)
                    if os.path.exists(domain_path):
                        train_datasets.append(ImageFolder(domain_path, transform=data_transform))
                if train_datasets:
                    train_dataset = ConcatDataset(train_datasets)
                else:
                    raise FileNotFoundError(f"No valid domain found in {data_root}")
            else:
                train_dataset = ImageFolder(train_domain_path, transform=data_transform)
            
            if not os.path.exists(test_domain_path):
                # 如果测试域不存在，使用训练域的一部分作为测试
                print(f"Warning: {test_domain_path} not found. Using train domain for testing.")
                test_dataset = train_dataset
            else:
                test_dataset = ImageFolder(test_domain_path, transform=data_transform)
            
            self.train_loader = DataLoader(
                train_dataset, batch_size=_batch_size, shuffle=True, num_workers=4, pin_memory=True
            )
            self.test_loader = DataLoader(
                test_dataset, batch_size=_batch_size, shuffle=False, num_workers=4, pin_memory=True
            )
            
            # 获取类别数量
            train_classes = len(train_dataset.classes) if hasattr(train_dataset, 'classes') else (
                len(train_dataset.datasets[0].classes) if isinstance(train_dataset, ConcatDataset) else 'N/A'
            )
            
            print(f"Office-31 dataset loaded:")
            print(f"  Training domain: {source_domain}, Classes: {train_classes}")
            print(f"  Testing domain: {target_domain}, Classes: {len(test_dataset.classes)}")
