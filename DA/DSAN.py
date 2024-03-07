import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from basemodel.base_model import get_base_model
from basemodel.MLP import ReLULayers
from DA.loss_func import LMMDLoss

class DSAN(nn.Module):
    def __init__(self, args):
        super(DSAN, self).__init__()

        self.featurizer = get_base_model(args.model)()
        self.classifier = nn.Linear(in_features=self.featurizer.output_dim(), out_features=args.num_classes)
        self.discriminator = LMMDLoss(args)
        
        self.args = args

    def forward(self, data_source, data_target, label_source):
        data_source, data_target = data_source.type(torch.float32), data_target.type(torch.float32)
        label_source = label_source.long()
        src_fea = self.featurizer(data_source)
        tgt_fea = self.featurizer(data_target)

        src_clf = self.classifier(src_fea)
        tgt_clf = self.classifier(tgt_fea)
        tgt_logits = torch.nn.functional.softmax(tgt_clf, dim=1)

        clf_loss = F.cross_entropy(src_clf, label_source)

        transfer_loss = self.discriminator(src_fea, tgt_fea, label_source, tgt_logits)
        
        return clf_loss, transfer_loss

    def predict(self, x):
        return self.classifier(self.featurizer(x))
    
    def get_parameters(self):
        return self.parameters()
