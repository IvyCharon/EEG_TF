from basemodel.DResNet import ResNet18
from basemodel.MLP import ReLULayers, MLP
from basemodel.CNN import CNN
from basemodel.SST_AGCN import SST_AGCN
import torch.nn as nn
from torch.autograd import Function

def get_base_model(model_type):
    if model_type not in globals():
        raise NotImplementedError(
            "Algorithm not found: {}".format(model_type))
    return globals()[model_type]


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=1) -> None:
        super().__init__()
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
