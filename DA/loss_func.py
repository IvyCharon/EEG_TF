import torch
import torch.nn as nn
import numpy as np
from basemodel.base_model import Discriminator
from basemodel.base_model import ReverseLayerF

class LambdaSheduler(nn.Module):
    def __init__(self, gamma, max_iter):
        super(LambdaSheduler, self).__init__()
        self.gamma = gamma
        self.max_iter = max_iter
        self.curr_iter = 0

    def lamb(self):
        p = self.curr_iter / self.max_iter
        lamb = 2. / (1. + np.exp(-self.gamma * p)) - 1
        return lamb
    
    def step(self):
        self.curr_iter = min(self.curr_iter + 1, self.max_iter)

class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss

class LMMDLoss(nn.Module):
    def __init__(self, args, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, gamma=1.0):
        super(LMMDLoss, self).__init__()
        self.lambda_scheduler = LambdaSheduler(gamma=gamma, max_iter=args.epoch*args.steps_per_epoch)
        self.mmdloss = MMDLoss(kernel_type=kernel_type, kernel_mul=kernel_mul, kernel_num=kernel_num)

        self.num_class = args.num_classes

    def forward(self, source, target, source_label, target_logits):
        if self.mmdloss.kernel_type == 'linear':
            raise NotImplementedError("Linear kernel is not supported yet.")
        
        elif self.mmdloss.kernel_type == 'rbf':
            batch_size = source.size()[0]
            weight_ss, weight_tt, weight_st = self.cal_weight(source_label, target_logits)
            weight_ss = torch.from_numpy(weight_ss).cuda() # B, B
            weight_tt = torch.from_numpy(weight_tt).cuda()
            weight_st = torch.from_numpy(weight_st).cuda()

            kernels = self.mmdloss.guassian_kernel(source, target,
                                    kernel_mul=self.mmdloss.kernel_mul, kernel_num=self.mmdloss.kernel_num, fix_sigma=self.mmdloss.fix_sigma)
            loss = torch.Tensor([0]).cuda()
            if torch.sum(torch.isnan(sum(kernels))):
                return loss
            SS = kernels[:batch_size, :batch_size]
            TT = kernels[batch_size:, batch_size:]
            ST = kernels[:batch_size, batch_size:]

            loss += torch.sum( weight_ss * SS + weight_tt * TT - 2 * weight_st * ST )
            # Dynamic weighting
            lamb = self.lambda_scheduler.lamb()
            self.lambda_scheduler.step()
            loss = loss * lamb
            return loss
    
    def cal_weight(self, source_label, target_logits):
        batch_size = source_label.size()[0]
        source_label = source_label.cpu().data.numpy()
        source_label_onehot = np.eye(self.num_class)[source_label] # one hot

        source_label_sum = np.sum(source_label_onehot, axis=0).reshape(1, self.num_class)
        source_label_sum[source_label_sum == 0] = 100
        source_label_onehot = source_label_onehot / source_label_sum # label ratio

        # Pseudo label
        target_label = target_logits.cpu().data.max(1)[1].numpy()

        target_logits = target_logits.cpu().data.numpy()
        target_logits_sum = np.sum(target_logits, axis=0).reshape(1, self.num_class)
        target_logits_sum[target_logits_sum == 0] = 100
        target_logits = target_logits / target_logits_sum

        weight_ss = np.zeros((batch_size, batch_size))
        weight_tt = np.zeros((batch_size, batch_size))
        weight_st = np.zeros((batch_size, batch_size))

        set_s = set(source_label)
        set_t = set(target_label)
        count = 0
        for i in range(self.num_class): # (B, C)
            if i in set_s and i in set_t:
                s_tvec = source_label_onehot[:, i].reshape(batch_size, -1) # (B, 1)
                t_tvec = target_logits[:, i].reshape(batch_size, -1) # (B, 1)
                
                ss = np.dot(s_tvec, s_tvec.T) # (B, B)
                weight_ss = weight_ss + ss
                tt = np.dot(t_tvec, t_tvec.T)
                weight_tt = weight_tt + tt
                st = np.dot(s_tvec, t_tvec.T)
                weight_st = weight_st + st     
                count += 1

        length = count
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')

class DAANLoss(nn.Module):
    def __init__(self, args, input_dim, gamma=1.0):
        super(DAANLoss, self).__init__()
        self.num_class = args.num_classes
        self.local_classifiers = torch.nn.ModuleList()
        for _ in range(args.num_classes):
            self.local_classifiers.append(Discriminator(input_dim=input_dim))
        self.LambdaSch = LambdaSheduler(gamma=gamma, max_iter=args.epoch*args.steps_per_epoch)
        self.domain_classifier = Discriminator(input_dim=input_dim)
        self.d_g, self.d_l = 0, 0
        self.dynamic_factor = 0.5
        self.args=args

    def forward(self, source, target, source_logits, target_logits):
        lamb = self.LambdaSch.lamb()
        self.LambdaSch.step()
        source_loss_g = self.get_adversarial_result(source, True, lamb)
        target_loss_g = self.get_adversarial_result(target, False, lamb)
        source_loss_l = self.get_local_adversarial_result(source, source_logits, True, lamb)
        target_loss_l = self.get_local_adversarial_result(target, target_logits, False, lamb)
        global_loss = 0.5 * (source_loss_g + target_loss_g) * 0.05
        local_loss = 0.5 * (source_loss_l + target_loss_l) * 0.01

        self.d_g = self.d_g + 2 * (1 - 2 * global_loss.cpu().item())
        self.d_l = self.d_l + 2 * (1 - 2 * (local_loss / self.num_class).cpu().item())

        adv_loss = (1 - self.dynamic_factor) * global_loss + self.dynamic_factor * local_loss
        return adv_loss
    
    def get_adversarial_result(self, x, source=True, lamb=1.0):
        x = ReverseLayerF.apply(x, lamb)
        domain_pred = self.domain_classifier(x)
        if source:
            domain_label = torch.ones(len(x), 1).long()
        else:
            domain_label = torch.zeros(len(x), 1).long()
        loss_fn = nn.BCELoss()
        loss_adv = loss_fn(domain_pred, domain_label.float().to(self.args.device))
        return loss_adv

    def get_local_adversarial_result(self, x, logits, c, source=True, lamb=1.0):
        loss_fn = nn.BCELoss()
        x = ReverseLayerF.apply(x, lamb)
        loss_adv = 0.0

        for c in range(self.num_class):
            logits_c = logits[:, c].reshape((logits.shape[0],1)) # (B, 1)
            features_c = logits_c * x
            domain_pred = self.local_classifiers[c](features_c)
            device = domain_pred.device
            if source:
                domain_label = torch.ones(len(x), 1).long()
            else:
                domain_label = torch.zeros(len(x), 1).long()
            loss_adv = loss_adv + loss_fn(domain_pred, domain_label.float().to(device))
        return loss_adv
    
    def update_dynamic_factor(self, epoch_length):
        if self.d_g == 0 and self.d_l == 0:
            self.dynamic_factor = 0.5
        else:
            self.d_g = self.d_g / epoch_length
            self.d_l = self.d_l / epoch_length
            self.dynamic_factor = 1 - self.d_g / (self.d_g + self.d_l)
        self.d_g, self.d_l = 0, 0

class AdversarialLoss(nn.Module):
    def __init__(self, input_dim, hidden_dim, args, gamma=1.0):
        super(AdversarialLoss, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.discriminator = Discriminator(input_dim, hidden_dim, 1)
        self.lambda_scheduler = LambdaSheduler(gamma=gamma, max_iter=args.epoch*args.steps_per_epoch)
        self.args=args
    
    def forward(self, src_fea, tgt_fea):
        disc_loss_func = nn.BCELoss()
        lamb = self.lambda_scheduler.lamb()
        self.lambda_scheduler.step()

        src_disc_input = ReverseLayerF.apply(src_fea, lamb)
        src_disc_out = self.discriminator(src_disc_input)
        src_domain_label = torch.ones(len(src_disc_input), 1).long()
        src_loss_adv = disc_loss_func(src_disc_out, src_domain_label.float().to(self.args.device))

        tgt_disc_input = ReverseLayerF.apply(tgt_fea, lamb)
        tgt_disc_out = self.discriminator(tgt_disc_input)
        tgt_domain_label = torch.zeros(len(tgt_disc_input), 1).long()
        tgt_loss_adv = disc_loss_func(tgt_disc_out, tgt_domain_label.float().to(self.args.device))

        transfer_loss = 0.5 * (src_loss_adv + tgt_loss_adv)

        return transfer_loss