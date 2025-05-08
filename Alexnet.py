import torch
from torch import nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=100):
        super(AlexNet, self).__init__()
        self.Conv1 = nn.Conv2d(3, 96, 11, 4, 2)
        self.Conv2 = nn.Conv2d(96, 256, 5, 1, 2)
        self.Conv3 = nn.Conv2d(256, 384, 3, 1, 1)
        self.Conv4 = nn.Conv2d(384, 384, 3, 1, 1)
        self.Conv5 = nn.Conv2d(384, 256, 3, 1, 1)

        self.LocalResponseNorm = nn.LocalResponseNorm(5, beta=0.75, alpha=0.0001, k=2)
        self.ReLu = nn.ReLU()
        self.MaxPool = nn.MaxPool2d(3,2)




        self.ConvolutionProcess = nn.Sequential(
            self.Conv1,
            self.ReLu,
            self.LocalResponseNorm,
            self.MaxPool,

            self.Conv2,
            self.ReLu,
            self.LocalResponseNorm,
            self.MaxPool,

            self.Conv3,
            self.ReLu,

            self.Conv4,
            self.ReLu,

            self.Conv5,
            self.ReLu,
            self.MaxPool
        )

        self.Fc1 = nn.Linear(9216, 4096)
        self.Fc2 = nn.Linear(4096, 4096)
        self.Fc3 = nn.Linear(4096, num_classes)
        self.Dropout = nn.Dropout(0.5)

        self.Classifier = nn.Sequential(
            self.Dropout,
            self.Fc1,
            self.ReLu,
            self.Dropout,
            self.Fc2,
            self.ReLu,
            self.Fc3,
        )


    def forward(self,x):
        ConvOut = self.ConvolutionProcess(x)
        FlattenX = torch.flatten(ConvOut, start_dim=1)
        ClassifierOut = self.Classifier(FlattenX)

        return ClassifierOut



