"""
ML Training Script with GPU Support
Trains ResNet18 on CIFAR-10 with real GPU workload
"""
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from model import create_model
from metrics import MetricsCollector

# CIFAR-10 classes
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def get_data_loaders(batch_size=128, data_dir='./data'):
    """Create CIFAR-10 data loaders"""
    print(f"Loading CIFAR-10 dataset to {data_dir}...")

    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # No augmentation for validation
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_val
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device, metrics_collector):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 50 == 0:
            print(f"  Batch [{batch_idx}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"Acc: {100. * correct / total:.2f}%")

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc


def train(epochs=10, batch_size=128, lr=0.1, save_path='model.pth', data_dir='./data'):
    """Main training function"""
    print("\n" + "="*60)
    print("Starting ML Training with GPU")
    print("="*60)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Metrics
    metrics_collector = MetricsCollector()
    print("\nInitial Metrics:")
    print(metrics_collector.get_all_metrics())

    # Model
    print("\nCreating model...")
    model = create_model(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Data
    train_loader, val_loader = get_data_loaders(batch_size, data_dir)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_acc = 0.0
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)

    for epoch in range(epochs):
        epoch_start = time.time()
        print(f"\nEpoch [{epoch+1}/{epochs}]")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, metrics_collector
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Print epoch summary
        print(f"\nEpoch [{epoch+1}/{epochs}] Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # GPU metrics
        gpu_metrics = metrics_collector.get_gpu_metrics()
        print(f"  GPU Memory: {gpu_metrics['gpu_memory_allocated_mb']:.0f} MB")
        print(f"  GPU Utilization: {gpu_metrics['gpu_utilization_percent']}%")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, save_path)
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {save_path}")

    # Final metrics
    print("\nFinal GPU Metrics:")
    final_metrics = metrics_collector.get_all_metrics()
    for key, value in final_metrics.items():
        if 'gpu' in key or 'memory' in key or 'cpu' in key:
            print(f"  {key}: {value}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ResNet18 on CIFAR-10')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--save-path', type=str, default='model.pth', help='path to save model')
    parser.add_argument('--data-dir', type=str, default='./data', help='data directory')

    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
        data_dir=args.data_dir
    )
