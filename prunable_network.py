"""
Self-Pruning Neural Network Implementation
Case Study: Tredence AI Engineering Internship
Author: [Your Name]

This module implements a neural network that learns to prune itself during training
by using learnable gate parameters on each weight in the network.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class PrunableLinear(nn.Module):
    """
    A custom linear layer with learnable gates for self-pruning.
    
    Each weight in this layer is multiplied by a gate value (between 0 and 1).
    The gates are learned during training, allowing the network to decide which
    weights are important and which should be pruned (set to ~0).
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
    """
    
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        
        # Standard weight and bias parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Gate scores - will be transformed to gates via sigmoid
        # Shape matches weight: (out_features, in_features)
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Initialize parameters using kaiming uniform distribution
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        
        # Initialize gate_scores to a value that produces ~0.5 probability
        # sigmoid(0) = 0.5, so initializing to 0 gives moderate sparsity
        nn.init.zeros_(self.gate_scores)
        
        # Initialize bias
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        """
        Forward pass with gated weights.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
            
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # Convert gate_scores to gates using sigmoid (produces values in [0, 1])
        gates = torch.sigmoid(self.gate_scores)
        
        # Element-wise multiplication: apply gates to weights
        pruned_weights = self.weight * gates
        
        # Standard linear transformation with pruned weights
        return torch.nn.functional.linear(x, pruned_weights, self.bias)
    
    def get_sparsity_loss(self):
        """
        Calculate L1 regularization loss on gate values.
        
        L1 penalty encourages sparsity by penalizing the sum of gate values.
        This drives gates toward 0 (pruning) or 1 (keeping), not intermediate values.
        
        Returns:
            Scalar tensor representing the L1 norm of all gate values
        """
        gates = torch.sigmoid(self.gate_scores)
        return gates.sum()
    
    def get_sparsity_level(self, threshold=1e-2):
        """
        Calculate the percentage of weights that have been pruned.
        
        Args:
            threshold: Gate values below this are considered pruned
            
        Returns:
            Sparsity level as a percentage (0-100)
        """
        gates = torch.sigmoid(self.gate_scores)
        pruned_count = (gates < threshold).sum().item()
        total_count = gates.numel()
        return (pruned_count / total_count) * 100
    
    def get_gates(self):
        """Get current gate values."""
        return torch.sigmoid(self.gate_scores).detach().cpu()


class PrunableNetwork(nn.Module):
    """
    A feed-forward neural network for CIFAR-10 using prunable layers.
    
    Architecture:
    - Input: 3072 features (32x32x3 images flattened)
    - Hidden layers: 512, 256, 128 units with ReLU activation
    - Output: 10 classes (CIFAR-10 classes)
    
    All layers use PrunableLinear for self-pruning capability.
    """
    
    def __init__(self):
        super(PrunableNetwork, self).__init__()
        
        # Prunable layers
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 128)
        self.fc4 = PrunableLinear(128, 10)
        
        # Activation function
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)
            
        Returns:
            Logits of shape (batch_size, 10)
        """
        # Flatten image to 3072D vector
        x = x.view(x.size(0), -1)
        
        # Pass through layers with ReLU activation
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x
    
    def get_total_sparsity_loss(self):
        """
        Calculate total sparsity loss across all prunable layers.
        
        Returns:
            Sum of L1 losses from all prunable linear layers
        """
        sparsity_loss = (
            self.fc1.get_sparsity_loss() +
            self.fc2.get_sparsity_loss() +
            self.fc3.get_sparsity_loss() +
            self.fc4.get_sparsity_loss()
        )
        return sparsity_loss
    
    def get_overall_sparsity_level(self, threshold=1e-2):
        """
        Calculate overall sparsity level across all prunable layers.
        
        Args:
            threshold: Gate values below this are considered pruned
            
        Returns:
            Overall sparsity as a percentage
        """
        sparsity_levels = [
            self.fc1.get_sparsity_level(threshold),
            self.fc2.get_sparsity_level(threshold),
            self.fc3.get_sparsity_level(threshold),
            self.fc4.get_sparsity_level(threshold),
        ]
        return np.mean(sparsity_levels)


def prepare_data(batch_size=128):
    """
    Prepare CIFAR-10 dataset with standard normalization.
    
    Args:
        batch_size: Batch size for data loaders
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Define transforms: normalize to ImageNet mean/std
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])
    
    # Download and load CIFAR-10
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer, criterion, device, lambda_sparse):
    """
    Train for one epoch with sparsity regularization.
    
    Args:
        model: PrunableNetwork instance
        train_loader: Training data loader
        optimizer: Optimizer (e.g., Adam)
        criterion: Classification loss (e.g., CrossEntropyLoss)
        device: Device to train on (cuda or cpu)
        lambda_sparse: Weight for sparsity loss
        
    Returns:
        Average loss and accuracy for the epoch
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Calculate classification loss
        classification_loss = criterion(outputs, targets)
        
        # Calculate sparsity loss
        sparsity_loss = model.get_total_sparsity_loss()
        
        # Combined loss
        total_loss_val = classification_loss + lambda_sparse * sparsity_loss
        
        # Backward pass and optimization
        total_loss_val.backward()
        optimizer.step()
        
        # Statistics
        total_loss += total_loss_val.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def test_epoch(model, test_loader, criterion, device):
    """
    Evaluate model on test set.
    
    Args:
        model: PrunableNetwork instance
        test_loader: Test data loader
        criterion: Classification loss
        device: Device to evaluate on
        
    Returns:
        Average loss and accuracy on test set
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def train_model(lambda_sparse, num_epochs=50, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Complete training pipeline for self-pruning network.
    
    Args:
        lambda_sparse: Weight for sparsity regularization
        num_epochs: Number of training epochs
        device: Device to train on
        
    Returns:
        Dictionary containing model, results, and metrics
    """
    print(f"\n{'='*60}")
    print(f"Training with λ (lambda) = {lambda_sparse}")
    print(f"{'='*60}")
    
    # Prepare data
    train_loader, test_loader = prepare_data(batch_size=128)
    
    # Initialize model
    model = PrunableNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training loop
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, lambda_sparse
        )
        test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
        scheduler.step()
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | "
                  f"Test Loss: {test_loss:.4f} | "
                  f"Test Acc: {test_acc:.2f}%")
    
    # Calculate final metrics
    final_sparsity = model.get_overall_sparsity_level(threshold=1e-2)
    final_test_acc = test_accs[-1]
    
    print(f"\nFinal Results (λ = {lambda_sparse}):")
    print(f"  Test Accuracy: {final_test_acc:.2f}%")
    print(f"  Sparsity Level: {final_sparsity:.2f}%")
    
    return {
        'model': model,
        'lambda': lambda_sparse,
        'test_accuracy': final_test_acc,
        'sparsity_level': final_sparsity,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
    }


def plot_gate_distribution(model, lambda_sparse, save_path='gate_distribution.png'):
    """
    Plot the distribution of final gate values for visualization.
    
    This plot shows the effectiveness of pruning:
    - A large spike near 0 indicates many pruned weights
    - A cluster away from 0 indicates important weights
    
    Args:
        model: Trained PrunableNetwork
        lambda_sparse: Lambda value used for training
        save_path: Path to save the plot
    """
    # Collect all gate values
    all_gates = []
    all_gates.append(model.fc1.get_gates().flatten().numpy())
    all_gates.append(model.fc2.get_gates().flatten().numpy())
    all_gates.append(model.fc3.get_gates().flatten().numpy())
    all_gates.append(model.fc4.get_gates().flatten().numpy())
    
    all_gates = np.concatenate(all_gates)
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_gates, bins=100, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=0.01, color='red', linestyle='--', linewidth=2, label='Pruning Threshold (0.01)')
    plt.xlabel('Gate Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Distribution of Gate Values (λ = {lambda_sparse})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Gate distribution plot saved to {save_path}")
    plt.close()


def main():
    """
    Main execution: train self-pruning network with different lambda values.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Lambda values to test (low, medium, high sparsity)
    lambda_values = [0.0001, 0.001, 0.01]
    
    results = []
    
    # Train with different lambda values
    for lambda_sparse in lambda_values:
        result = train_model(lambda_sparse, num_epochs=50, device=device)
        results.append(result)
        
        # Plot gate distribution for each model
        plot_name = f'gate_distribution_lambda_{lambda_sparse}.png'
        plot_gate_distribution(result['model'], lambda_sparse, plot_name)
    
    # Print results summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Lambda':<15} {'Test Accuracy (%)':<20} {'Sparsity Level (%)':<20}")
    print(f"{'-'*70}")
    
    for result in results:
        lambda_val = result['lambda']
        test_acc = result['test_accuracy']
        sparsity = result['sparsity_level']
        print(f"{lambda_val:<15.5f} {test_acc:<20.2f} {sparsity:<20.2f}")
    
    print(f"{'='*70}\n")
    
    # Save results to file
    with open('results_summary.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("SELF-PRUNING NEURAL NETWORK - RESULTS SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write("Explanation of Sparsity:\n")
        f.write("-" * 70 + "\n")
        f.write("L1 regularization on sigmoid gates encourages sparsity because:\n")
        f.write("1. L1 penalty (sum of absolute values) has a geometric property that\n")
        f.write("   naturally drives coefficients to exactly zero.\n")
        f.write("2. When combined with sigmoid gates, the L1 loss on gate values\n")
        f.write("   penalizes having gates at intermediate values (0.3-0.7).\n")
        f.write("3. The optimizer prefers to push gates to either 0 (fully pruned)\n")
        f.write("   or 1 (fully active) rather than maintain them in-between.\n")
        f.write("4. The trade-off is controlled by λ: higher λ = more aggressive pruning.\n\n")
        
        f.write("Results Table:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Lambda':<15} {'Test Accuracy (%)':<20} {'Sparsity Level (%)':<20}\n")
        f.write("-" * 70 + "\n")
        
        for result in results:
            lambda_val = result['lambda']
            test_acc = result['test_accuracy']
            sparsity = result['sparsity_level']
            f.write(f"{lambda_val:<15.5f} {test_acc:<20.2f} {sparsity:<20.2f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("Analysis:\n")
        f.write("-" * 70 + "\n")
        f.write("The results demonstrate the trade-off between model sparsity and accuracy:\n")
        f.write("- Lower λ values: Higher accuracy, lower sparsity (fewer weights pruned)\n")
        f.write("- Higher λ values: Lower accuracy, higher sparsity (more weights pruned)\n")
        f.write("- The optimal λ depends on your deployment constraints (speed vs accuracy)\n")
    
    print("Results saved to results_summary.txt")


if __name__ == '__main__':
    main()
