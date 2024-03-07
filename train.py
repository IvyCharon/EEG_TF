import numpy as np
import os
import file_info as fi
import torch
from random import choice
import random
import argparse
from data_process.process import data_process, get_eeg_dataloader

from tool import print_args, get_model

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main(args):
    set_random_seed(args.seed)
    data, label = data_process(args.model, args.dataset, args.normalize)
    acc, epoch = [], []
    for p in range(fi.total_subjects_number[args.dataset]):
        src_loader, tgt_train_list, tgt_loader = get_eeg_dataloader(p+1, data, label, args.dataset, args.batch_size, args.method)
        acc_, epoch_ = train(src_loader, tgt_train_list, tgt_loader, p+1, args)
        acc.append(acc_.data.cpu())
        epoch.append(epoch_)
        print("finish subject {}, acc = {}, epoch = {}".format(p+1, acc_, epoch_))

    acc_mean=np.mean(acc)
    std=np.std(acc)
    print(args.dataset + ' '+ args.alg)
    print(acc_mean)
    print(std)
    print('----------')

    output_path = os.path.join('output', args.alg, args.dataset, '{}.txt'.format(args.model))
    with open(output_path, 'a') as f:
        # config
        f.write('----\n')
        f.write(print_args(args, []))
        # results
        f.write('total_acc: {}\n'.format(acc))
        f.write('Acc: {}\n'.format(acc_mean))
        f.write('std: {}\n'.format(std))
        f.write('-----\n\n')

def get_optimizer(param, args):
    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(param, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'Adamw':
        optimizer = torch.optim.AdamW(param, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError()
    return optimizer

def get_scheduler(optimizer, args):
    if args.schusech == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epoch * args.steps_per_epoch)
    elif args.schusech == 'lambda':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    else:
        raise NotImplementedError()
    return scheduler

def train(src_loader, tgt_train_list, tgt_loader, tgt_num, args):
    save_model_path = os.path.join('output', args.alg, args.dataset)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    save_model_path = os.path.join(save_model_path, args.model+'-{}.pt'.format(tgt_num))

    if args.method == 'DA':
        n_batch = len(src_loader)
        setattr(args, "steps_per_epoch", n_batch)
        model = get_model(args.alg)(args).to(args.device)
        optimizer = get_optimizer(model.get_parameters(), args)
        if args.schuse:
            scheduler = get_scheduler(optimizer, args)
        best_acc = 0.
        best_epoch = 0
        for epoch in range(args.epoch):
            model.train()
            iter_source = iter(src_loader)
            for _ in range(n_batch):
                data_source, label_source = next(iter_source)
                data_target = choice(tgt_train_list)
                data_source, label_source = data_source.to(
                    args.device), label_source.to(args.device)
                data_target = data_target.to(args.device)
                clf_loss, transfer_loss = model(data_source, data_target, label_source)
                loss = clf_loss + args.transfer_loss_weight * transfer_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if args.schuse:
                    scheduler.step()
            acc = test(model, tgt_loader, args)
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                torch.save(model.state_dict(), save_model_path)
    elif args.method == 'DG':
        train_minibatches_iterator = list(zip(*src_loader))
        setattr(args, "steps_per_epoch", len(train_minibatches_iterator))
        model = get_model(args.alg)(args).to(args.device)
        optimizer = get_optimizer(model.get_parameters(), args)
        if args.schuse:
            scheduler = get_scheduler(optimizer, args)
        best_acc = 0.
        best_epoch = 0
        for epoch in range(args.epoch):
            model.train()
            for bt in train_minibatches_iterator:
                minibatches = [(data) for data in bt]
                loss = model(minibatches)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if args.schuse:
                    scheduler.step()

            acc = test(model, tgt_loader, args)
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                torch.save(model.state_dict(), save_model_path)
    elif args.method == 'baseline':
        n_batch = len(src_loader)
        setattr(args, "steps_per_epoch", n_batch)
        model = get_model('Baseline')(args).to(args.device)
        optimizer = get_optimizer(model.get_parameters(), args)
        if args.schuse:
            scheduler = get_scheduler(optimizer, args)
        best_acc = 0.
        best_epoch = 0
        for epoch in range(args.epoch):
            model.train()
            iter_source = iter(src_loader)
            for _ in range(n_batch):
                x, y = next(iter_source)
                x, y = x.to(args.device), y.to(args.device)
                loss = model(x, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if args.schuse:
                    scheduler.step()
    
            acc = test(model, tgt_loader, args)
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                torch.save(model.state_dict(), save_model_path)
    elif args.method == 'DAMD':
        train_minibatches_iterator = list(zip(*src_loader))
        setattr(args, "steps_per_epoch", len(train_minibatches_iterator))
        model = get_model(args.alg)(args).to(args.device)
        optimizer = get_optimizer(model.get_parameters(), args)
        if args.schuse:
            scheduler = get_scheduler(optimizer, args)
        best_acc = 0.
        best_epoch = 0
        for epoch in range(args.epoch):
            model.train()
            for bt in train_minibatches_iterator:
                minibatches = [(data) for data in bt]
                data_target = choice(tgt_train_list).to(args.device)
                loss = model(minibatches, data_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if args.schuse:
                    scheduler.step()

            acc = test(model, tgt_loader, args)
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                torch.save(model.state_dict(), save_model_path)
    else:
        raise NotImplementedError()
    
    return best_acc, best_epoch

def test(model, tgt_loader, args):
    model.eval()
    correct = 0.
    with torch.no_grad():
        for x, y in tgt_loader:
            x, y = x.to(args.device), y.to(args.device)
            y_pred = torch.max(model.predict(x), 1)[1]
            correct += torch.sum(y_pred==y)
    acc = correct / len(tgt_loader.dataset)
    return acc

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--normalize", type=str, default='standard', choices=['standard', 'minmax', 'no'])
    parser.add_argument('--dataset', type=str, default='SEED', choices=['SEED', 'SEED-IV', 'SEED-V'])
    parser.add_argument('--alg', type=str, default='baseline')
    parser.add_argument('--model', type=str, default='CNN')


    parser.add_argument('--epoch', type=int, default=40, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-3, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--momentum', type=float, default=0.9, help='for optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--schuse', action='store_true')
    parser.add_argument('--schusech', type=str, default='cos')
    parser.add_argument('--optim', type=str, default='SGD')

    parser.add_argument('--seed', type=int, default=723)
    parser.add_argument('--alpha', type=float, default=1, help='DANN_DG dis alpha')   
    parser.add_argument('--transfer_loss_weight', type=float, default=1., help='transfer_loss_weight')
    parser.add_argument('--mmd_gamma', type=float, default=1, help='MMD, CORAL hyper-param')
    
    parser.add_argument('--rsc_f_drop_factor', type=float, default=1./3, help='rsc hyper-param')
    parser.add_argument('--rsc_b_drop_factor', type=float, default=1./3, help='rsc hyper-param')
    
    parser.add_argument('--anneal_iters', type=int, default=500, help='Penalty anneal iters used in VREx')
    parser.add_argument('--lam', type=float, default=1., help="tradeoff hyperparameter used in VREx")
    
    parser.add_argument('--n', type=int, default=5, help="MDDA n")

    args = parser.parse_args()
    setattr(args, "num_classes", fi.num_classes[args.dataset])
    setattr(args, "num_subject", fi.total_subjects_number[args.dataset])
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    if args.alg in fi.DA_Method:
        setattr(args, "method", 'DA')
    elif args.alg in fi.DG_Method:
        setattr(args, "method", 'DG')
    elif args.alg == 'baseline':
        setattr(args, "method", 'baseline')
    elif args.alg in fi.DAMD_Method:
        setattr(args, "method", 'DAMD')
    else:
        raise NotImplementedError

    main(args)

