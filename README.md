# IBCapsNet: Information Bottleneck Capsule Network for Noise-Robust Representation Learning

A PyTorch implementation of **IBCapsNet**, an efficient and robust capsule network architecture that replaces dynamic routing with variational encoding based on the Information Bottleneck principle.

## 🎯 Overview

IBCapsNet introduces a novel approach to capsule networks by leveraging the Information Bottleneck (IB) principle and variational autoencoders (VAE) to replace the computationally expensive dynamic routing mechanism in traditional CapsNet. This results in **3.64× faster inference** while maintaining comparable accuracy and significantly improved robustness.

### Key Innovations

1. **Information Bottleneck Principle**: Replaces iterative dynamic routing with variational encoding, achieving information compression through KL divergence regularization
2. **Single Forward Pass**: Eliminates the need for 3 iterations of routing, reducing computational complexity from O(3×N×M) to O(N×M)
3. **Enhanced Robustness**: Superior performance under noise conditions, with up to **42.77% improvement** in high-noise scenarios
4. **Flexible Classifier Design**: Supports three classifier types (linear, squash, inverse_squash) for different application scenarios

## 📊 Experimental Results

### Accuracy Comparison

| Dataset | CapsNet | IBCapsNet-Linear | IBCapsNet-Squash | LeNet |
|---------|---------|-----------------|------------------|-------|
| MNIST | 99.46% | 99.39% | 99.41% | 98.99% |
| Fashion-MNIST | 90.83% | 90.72% | 90.78% | 90.17% |
| CIFAR-10 | ~72.30% | ~68.93% | ~70.58% | ~60.86% |
| SVHN | 92.12% | 91.31% | 92.01% | 85.75% |

### Performance Highlights

- **Inference Speed**: 3.64× faster (149.93 FPS vs 41.15 FPS on MNIST)
- **Robustness**: Up to 17.10% average improvement across datasets under clamped additive noise
- **Parameter Efficiency**: Comparable parameter count to CapsNet (~10M parameters)

## 📁 Project Structure

### Core Implementation Files

- **`IBCapsnet.py`**: Core IBCapsNet implementation
  - `IBCapsNet`: Base model without reconstruction
  - `IBCapsNetWithRecon`: Full model with reconstruction capability
  - `EnhancedContextEncoder`: Enhanced context encoder with attention mechanisms
  - `IBCapsules`: Information bottleneck capsule layer (core innovation)

- **`capsnet.py`**: Original CapsNet implementation (Hinton's architecture)

- **`train_lenet.py`**: LeNet implementation for baseline comparison

### Experimental Scripts

- **`comparison_experiment.py`**: Main comparison experiments
  - Accuracy comparison across multiple datasets
  - Training speed comparison
  - Few-shot learning experiments
  - Parameter efficiency analysis

- **`ablation_study_simple.py`**: Progressive ablation study
  - Component contribution analysis
  - Noise robustness testing

- **`comprehensive_test_comparison.py`**: Comprehensive testing and comparison
  - Multi-dataset evaluation
  - Robustness testing under various noise conditions

### Utility Files

- **`data_loader.py`**: Dataset loader supporting MNIST, Fashion-MNIST, CIFAR-10, and SVHN
- **`test_capsnet.py`**: Testing script for CapsNet
- **`visualize_reconstruction_comparison.py`**: Visualization tools for reconstruction comparison

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd IBCapsNet

# Install dependencies
pip install torch torchvision numpy matplotlib tqdm
```

### Requirements

- Python 3.6+
- PyTorch 1.0+ (tested with PyTorch 1.8+)
- NumPy
- Matplotlib
- tqdm

### Basic Usage

#### 1. Run Comparison Experiments

```bash
# Compare models on MNIST (30 epochs)
python comparison_experiment.py --dataset mnist --epochs 30

# Compare models on CIFAR-10
python comparison_experiment.py --dataset cifar10 --epochs 30

# Exclude LeNet from comparison
python comparison_experiment.py --dataset mnist --epochs 30 --no-lenet

# Use enhanced context encoder
python comparison_experiment.py --dataset cifar10 --epochs 30 --context-encoder-type enhanced
```

#### 2. Run Ablation Study

```bash
# Run progressive ablation study on Fashion-MNIST
python ablation_study_simple.py --dataset fashion-mnist --epochs 20
```

#### 3. Comprehensive Testing

```bash
# Run comprehensive comparison tests
python comprehensive_test_comparison.py --dataset mnist
```

#### 4. Test Individual Models

```bash
# Test CapsNet
python train_capsnet.py

# Train LeNet
python train_lenet.py
```

## 🔬 Experimental Details

### Supported Datasets

- **MNIST**: 28×28 grayscale images, 10 classes
- **Fashion-MNIST**: 28×28 grayscale images, 10 classes
- **CIFAR-10**: 32×32 RGB images, 10 classes
- **SVHN**: 32×32 RGB images, 10 classes

### Model Variants

1. **IBCapsNet-Linear**: Uses linear classifier with binary cross-entropy loss
2. **IBCapsNet-Squash**: Uses squash activation (CapsNet-style) with margin loss
3. **IBCapsNet-Inverse_Squash**: Uses inverse squash activation (novel design) with margin loss

### Key Hyperparameters

- **Learning Rate**: 0.001 (IBCapsNet), 0.01 (CapsNet, LeNet)
- **Batch Size**: 128
- **KL Divergence Weight (β)**: 1e-3
- **Reconstruction Weight (α)**: 0.0005
- **Latent Dimension**: 16 (default)

## 📈 Results and Analysis

### Experimental Results Location

All experimental results are saved in timestamped directories:

- **Comparison Results**: `comparison_results_{dataset}_{timestamp}/`
  - `summary.json`: Experiment summary
  - `experiment_accuracy.json`: Detailed accuracy results
  - `*_best.pth`: Best model checkpoints
  - `reconstruction_visualizations/`: Reconstruction visualizations

- **Ablation Study Results**: `ablation_study_simple_{dataset}_{timestamp}/`
  - `all_results.json`: All experimental results
  - `summary.json`: Experiment summary
  - `visualizations/`: Visualization charts

### Key Findings

1. **Efficiency**: IBCapsNet achieves 3.64× speedup while maintaining comparable accuracy
2. **Robustness**: Significant improvements under noise conditions, especially for clamped additive noise (17.10% average improvement)
3. **Component Analysis**: Enhanced context encoder contributes most (+1.26%), followed by KL regularization and VAE encoding
4. **Reconstruction Impact**: Reconstruction network is crucial for noise robustness (+10.16% improvement)

## 🔍 Architecture Details

### IBCapsNet Architecture

```
Input Image
    ↓
Conv Layer (256 channels)
    ↓
Primary Capsules (1152 capsules, 8-dim each)
    ↓
Context Encoder (Global context: 256-dim)
    ↓
Class Encoders (10 independent VAE encoders)
    ↓
Reparameterization (z = μ + ε·σ)
    ↓
Classifier (Linear/Squash/Inverse_Squash)
    ↓
Output Probabilities
```

### Key Components

1. **Context Encoder**: Encodes global context from primary capsules
   - Default: Simple average pooling + FC layers
   - Enhanced: Channel and spatial attention mechanisms

2. **Class Encoders**: Independent VAE encoders for each class
   - Input: Global context (256-dim)
   - Output: Latent distribution parameters (μ, logσ²)

3. **Reparameterization**: Samples latent vectors z ~ N(μ, σ²)

4. **Classifier**: Three types available
   - Linear: Standard linear classification
   - Squash: CapsNet-style activation (preserves long vectors)
   - Inverse Squash: Novel activation (preserves short vectors)

5. **Reconstruction Decoder** (optional): Reconstructs images from latent vectors

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{ibcapsnet2024,
  title={IBCapsNet: Information Bottleneck Capsule Network for Noise-Robust Representation Learning},
  author={Canqun Xiang, Chen Yang, Jiaoyan Zhao},
  journal={Journal/Conference Name},
  year={2024}
}
```

## 🙏 Acknowledgements

- Original CapsNet implementation: [Capsule-Network-Tutorial](https://github.com/higgsfield/Capsule-Network-Tutorial)
- Hinton's original paper: [Dynamic routing between capsules](http://papers.nips.cc/paper/6975-dynamic-routing-between-capsules)
- Information Bottleneck theory: Tishby et al. (2000)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This implementation extends the original PyTorch Capsule Network repository with IBCapsNet, a novel architecture that significantly improves efficiency and robustness while maintaining comparable accuracy.
