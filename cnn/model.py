# model.py
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

def create_model(num_classes=2):
    weights = ResNet18_Weights.IMAGENET1K_V1
    m = models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m