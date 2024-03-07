import torch
import torch.nn as nn
import torch.nn.functional as F

from basemodel.base_model import get_base_model
from basemodel.MLP import ReLULayers
from basemodel.base_model import ReverseLayerF

class DANN_DG(nn.Module):

    def __init__(self, args):
        super(DANN_DG, self).__init__()

        self.featurizer = get_base_model(args.model)()
        self.classifier = nn.Linear(in_features=self.featurizer.output_dim(), out_features=args.num_classes)
        self.discriminator = ReLULayers(layer_num=2, input_dim=self.featurizer.output_dim(), hidden_dim=512, output_dim=args.num_subject-1)
        
        self.args = args

    def forward(self, minibatches):
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        
        all_z = self.featurizer(all_x)

        disc_input = all_z
        disc_input = ReverseLayerF.apply(
            disc_input, self.args.alpha)
        disc_out = self.discriminator(disc_input)
        
        disc_labels = torch.cat([
            torch.full((data[0].shape[0], ), i,
                       dtype=torch.int64, device='cuda')
            for i, data in enumerate(minibatches)
        ])
        
        disc_loss = F.cross_entropy(disc_out, disc_labels)
        all_preds = self.classifier(all_z)
        
        classifier_loss = F.cross_entropy(all_preds, all_y)
        
        loss = classifier_loss+disc_loss
        
        return loss

    def predict(self, x):
        return self.classifier(self.featurizer(x))
    
    def get_parameters(self):
        return self.parameters()
