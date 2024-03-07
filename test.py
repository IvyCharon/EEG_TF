import numpy as np
import os
import file_info as fi
import torch
from random import choice
import argparse
from data_process.process import data_process, get_eeg_dataloader

from tool import print_args, get_model

def main(args):
    data, label = data_process(args.model, args.dataset, args.normalize)
    acc = []
    for p in range(fi.total_subjects_number[args.dataset]):
        _, _, tgt_loader = get_eeg_dataloader(p+1, data, label, args.dataset, args.batch_size, args.method)
        model = get_model(args.alg)(args).to(args.device)
        model_path = os.path.join('output', args.alg, args.dataset, args.model+'-{}.pt'.format(p+1))
        model.load_state_dict(torch.load(model_path))
        acc_ = test(model, tgt_loader, args)
        acc.append(acc_.data.cpu())
        print("finish subject {}, acc = {}".format(p+1, acc_))

    acc_mean=np.mean(acc)
    std=np.std(acc)
    print(args.dataset + ' '+ args.alg)
    print(acc_mean)
    print(std)
    print('----------')

    output_path = os.path.join('output', args.alg, args.dataset, '{}_test.txt'.format(args.model))
    with open(output_path, 'a') as f:
        # config
        f.write('----\n')
        f.write(print_args(args, []))
        # results
        f.write('total_acc: {}\n'.format(acc))
        f.write('Acc: {}\n'.format(acc_mean))
        f.write('std: {}\n'.format(std))
        f.write('-----\n\n')

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
    parser.add_argument('--method', type=str, default='baseline')
    parser.add_argument('--alg', type=str, default='baseline')
    parser.add_argument('--model', type=str, default='CNN')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')    
    parser.add_argument('--n', type=int, default=5, help="MDDA n")

    args = parser.parse_args()
    setattr(args, "num_classes", fi.num_classes[args.dataset])
    setattr(args, "num_subject", fi.total_subjects_number[args.dataset])
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    if args.alg in fi.DA_Method:
        args.method = 'DA'
    elif args.alg in fi.DG_Method:
        args.method = 'DG'
    elif args.alg == 'baseline':
        args.method = 'baseline'
    elif args.alg in fi.DAMD_Method:
        args.method = 'DAMD'
    else:
        raise NotImplementedError

    main(args)

