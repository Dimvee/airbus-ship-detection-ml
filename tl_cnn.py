import torch
import torch.nn as nn
from torchvision import models

class TLCNN(nn.Module):
    def __init__(self):
        super().__init__()

        weights = models.DenseNet169_Weights.DEFAULT
        self.backbone = models.densenet169(weights=weights)

        # Freeze pretrained DenseNet169 layers
        for param in self.backbone.parameters():
            param.requires_grad = False

        in_features = self.backbone.classifier.in_features

        self.backbone.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x)

print("TL-CNN source code saved successfully!")
