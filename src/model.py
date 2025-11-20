"""
ML Model Definition - ResNet18 for CIFAR-10
This model performs real GPU computations for training and inference.
"""
import torch
import torch.nn as nn
import torchvision.models as models


class CIFAR10ResNet(nn.Module):
    """ResNet18 adapted for CIFAR-10 (32x32 images, 10 classes)"""

    def __init__(self, num_classes=10):
        super(CIFAR10ResNet, self).__init__()
        # Load pretrained ResNet18 and modify for CIFAR-10
        self.model = models.resnet18(pretrained=False)

        # Modify first conv layer for 32x32 images (CIFAR-10)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # Remove maxpool for small images

        # Modify final layer for 10 classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


def create_model(device='cuda'):
    """Create model and move to device"""
    model = CIFAR10ResNet(num_classes=10)
    model = model.to(device)
    return model


def get_model_info():
    """Get model information"""
    model = CIFAR10ResNet()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'name': 'ResNet18-CIFAR10',
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'input_size': [3, 32, 32],
        'num_classes': 10
    }
