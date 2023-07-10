import torch.nn as nn
import torch
from torchvision.models import resnet18, resnet34, resnet50


class ResNet18YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.resnet = self.init_resnet()
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.Dropout(p=0.5, inplace=True),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, self.S**2 * (5 * self.B + self.C)),
            # nn.Sigmoid()
        )

    def init_resnet(self):
        resnet = resnet18(weights="IMAGENET1K_V1")
        
        # print(resnet)

        # replace relu with leaky relu
        resnet = self.replace_with_leaky_relu(resnet)

        # remove feedforward layer
        named_children = resnet.named_children()
        layers_to_remove = set(["fc", "avgpool"])
        layers = [
            module for name, module in named_children if name not in layers_to_remove
        ]

        # add a conv layer at the end to reduce feature map to (512, 7, 7)
        layers.append(nn.Conv2d(512, 512, kernel_size=2, stride=2))

        return nn.Sequential(*layers)

    def replace_with_leaky_relu(self, nn_module):
        named_children = nn_module.named_children()

        # loop over immediate children modules
        for name, module in named_children:
            is_relu = isinstance(module, nn.ReLU)

            if is_relu:
                leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
                setattr(nn_module, name, leaky_relu)
            else:
                self.replace_with_leaky_relu(module)

        return nn_module

    def forward(self, x):
        x = self.resnet(x)
        # print(x.shape)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(-1, self.S, self.S, 5 * self.B + self.C)
        # x = nn.ReLU()(x)

        return x
