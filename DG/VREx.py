import torch
import torch.nn as nn
import torch.nn.functional as F

from basemodel.base_model import get_base_model
from basemodel.MLP import ReLULayers


class VREx(nn.Module):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""

    def __init__(self, args):
        super(VREx, self).__init__()

        self.featurizer = get_base_model(args.model)()
        self.classifier = nn.Linear(in_features=self.featurizer.output_dim(), out_features=args.num_classes)
        self.register_buffer('update_count', torch.tensor([0]))
        self.args = args

    def forward(self, minibatches):
        if self.update_count >= self.args.anneal_iters:
            penalty_weight = self.args.lam
        else:
            penalty_weight = 1.0

        nll = 0.

        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_logits = self.classifier(self.featurizer(all_x))
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))

        for i, data in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx +
                                data[0].shape[0]]
            all_logits_idx += data[0].shape[0]
            nll = F.cross_entropy(logits, data[1].cuda().long())
            losses[i] = nll

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = mean + penalty_weight * penalty


        self.update_count += 1
        return loss

    def predict(self, x):
        return self.classifier(self.featurizer(x))
    
    def get_parameters(self):
        return self.parameters()