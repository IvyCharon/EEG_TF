import torch.nn as nn
import torch.nn.functional as F
import torch
from basemodel.base_model import get_base_model
from DA.loss_func import MMDLoss
import numpy as np

class L2B(nn.Module):
    def __init__(self, args):
        super(L2B, self).__init__()
        self.featurizer = get_base_model(args.model)()
        self.classifier = nn.ModuleList()
        for _ in range(args.num_subject - 1):
            self.classifier.append(nn.Linear(in_features=self.featurizer.output_dim(), out_features=args.num_classes))

        self.args = args


    def forward(self, minibatches, data_target):
        feature= [self.featurizer(data[0].cuda().float()) for data in minibatches]
        labels = [data[1].cuda().long() for data in minibatches]
        
        nmb = len(minibatches)
        cls_loss = 0.

        for i in range(nmb):
            src_pred_i = self.classifier[i](feature[i])
            cls_loss += F.cross_entropy(src_pred_i, labels[i])
        
        pred = torch.zeros((nmb, data_target.shape[0], self.args.num_classes))
        for i in range(nmb):
            pred[i] = self.classifier[i](self.featurizer(data_target))
        pred = pred.transpose(1,0)  # [l, 15, 5]
        tgt_loss = (pred.std(dim=1).sum() - pred.std(dim=2).sum())*0.0001
        
        loss = cls_loss + tgt_loss
        return loss

    def predict(self, x):
        feature = self.featurizer(x)
        for i in range(self.args.num_subject-1):
            pred_i=self.classifier[i](feature)
            if i==0:
                fp = pred_i
            else:
                fp += pred_i
        fp /= self.args.num_subject-1
        return fp

    def get_parameters(self):
        return self.parameters()
    
