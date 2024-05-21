from torch import nn
import torch.nn.functional as F


class CatVsDogModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = nn.Conv2d(3, 32, kernel_size=3)

    def forward(self, x):
        ...
