from torchvision import models
import torch.nn as nn
from torchinfo import summary


class Model(nn.Module):
    def __init__(self,n):
        super(Model, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048, n)

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model = Model(56)
    summary(model, input_size=(1, 3, 224, 224))
