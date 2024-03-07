import torch.nn as nn
import torch.nn.functional as F
import torch
from basemodel.base_model import get_base_model
from DA.loss_func import MMDLoss
import math
import numpy as np

class L3B(nn.Module):
    def __init__(self, args):
        super(L3B, self).__init__()
        self.featurizer = get_base_model(args.model)()
        self.classifier = nn.ModuleList()
        for _ in range(args.num_subject - 1):
            self.classifier.append(nn.Linear(in_features=self.featurizer.output_dim(), out_features=args.num_classes))
    

        self.sub_featurizer = get_base_model(args.model)()
        self.sub_classifier = nn.Linear(in_features=self.sub_featurizer.output_dim(), out_features=args.num_subject)
        self.args = args


    def forward(self, minibatches, data_target):
        feature= [self.featurizer(data[0].cuda().float()) for data in minibatches]
        labels = [data[1].cuda().long() for data in minibatches]
        sub_label = torch.cat([
            torch.full((data[0].shape[0], ), i,
                       dtype=torch.int64, device='cuda')
            for i, data in enumerate(minibatches)
        ])
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

        feature_sub = torch.cat([self.sub_featurizer(data[0].cuda().float()) for data in minibatches])
        sub_pred = self.sub_classifier(feature_sub)
        sub_clf_loss = F.cross_entropy(sub_pred, sub_label)
        
        loss = cls_loss + sub_clf_loss + tgt_loss
        return loss

    def predict(self, x):
        n=self.args.n        
        feature = self.featurizer(x)
        pred = []
        for i in range(self.args.num_subject-1):
            pred_i=self.classifier[i](feature)
            pred.append(pred_i)
        sub_feature = self.sub_featurizer(x)
        sub_pred = self.sub_classifier(sub_feature)
        _, indices = torch.topk(sub_pred, k=n, largest=False)
        sub_pred.scatter_(1, indices, 0)
        sub_pred = F.softmax(sub_pred, dim=1)
        for i in range(self.args.num_subject-1):
            fp_i=(pred[i].transpose(1,0)*sub_pred[:,i]).transpose(1,0)
            if i==0:
                fp = fp_i
            else:
                fp += fp_i
        return fp

    def get_parameters(self):
        return self.parameters()
    
