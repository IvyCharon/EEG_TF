import torch
import torch.nn as nn
from basemodel.MLP import ReLULayers
from basemodel.base_model import get_base_model

class Baseline(nn.Module):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, args):
        super(Baseline, self).__init__()
        self.featurizer = get_base_model(args.model)()
        self.classifier = nn.Linear(in_features=self.featurizer.output_dim(), out_features=args.num_classes)
        
        self.network = nn.Sequential(
            self.featurizer, self.classifier)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        pred = self.network(x).type(torch.float32)
        y = y.long()
        loss = self.criterion(pred, y)
        return loss

    def predict(self, x):
        return self.network(x)
    
    def get_parameters(self):
        return self.parameters()
