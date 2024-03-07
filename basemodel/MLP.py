import torch.nn as nn
import torch    

class ReLULayers(nn.Module):
    def __init__(self, layer_num, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        if layer_num > 2:
            for _ in range(layer_num - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, output_dim), nn.ReLU()]
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
    
class MLP(nn.Module):
    def __init__(self, input_dim = 310) -> None:
        super().__init__()
        self.net = ReLULayers(2, input_dim, 512, 512)
    
    def forward(self, x):
        out = self.net(x)
        return out
    
    def output_dim(self):
        return 512
