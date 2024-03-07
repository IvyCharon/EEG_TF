import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from basemodel.base_model import get_base_model
from basemodel.MLP import ReLULayers

class CORAL_DA(nn.Module):
    def __init__(self, args):
        super(CORAL_DA, self).__init__()

        self.featurizer = get_base_model(args.model)()
        self.classifier = nn.Linear(in_features=self.featurizer.output_dim(), out_features=args.num_classes)
        
        self.args = args

    def coral(self, source, target):
        d = source.data.shape[1]
        ns, nt = source.data.shape[0], target.data.shape[0]
        # source covariance
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = xm.t() @ xm / (ns - 1)

        # target covariance
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = xmt.t() @ xmt / (nt - 1)

        # frobenius norm between source and target
        loss = torch.mul((xc - xct), (xc - xct))
        loss = torch.sum(loss) / (4*d*d)
        return loss


    def forward(self, data_source, data_target, label_source):
        data_source, data_target = data_source.type(torch.float32), data_target.type(torch.float32)
        label_source = label_source.long()
        src_fea = self.featurizer(data_source)
        tgt_fea = self.featurizer(data_target)

        src_clf = self.classifier(src_fea)

        clf_loss = F.cross_entropy(src_clf, label_source)

        transfer_loss = self.coral(src_fea, tgt_fea)
        
        return clf_loss, transfer_loss

    def predict(self, x):
        return self.classifier(self.featurizer(x))
    
    def get_parameters(self):
        return self.parameters()
