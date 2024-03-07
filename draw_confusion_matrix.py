import numpy as np
import os
import matplotlib.pyplot as plt
import file_info as fi
import torch
from random import choice
import argparse
from data_process.process import data_process, get_eeg_dataloader

from tool import print_args, get_model

import seaborn as sns

def main(args):
    data, label = data_process(args.model, args.dataset, args.normalize)
    acc = []
    result =[]
    for p in range(fi.total_subjects_number[args.dataset]):
        _, _, tgt_loader = get_eeg_dataloader(p+1, data, label, args.dataset, args.batch_size, args.method)
        if args.alg == 'baseline':
            model = get_model('Baseline')(args).to(args.device)
        else:
            model = get_model(args.alg)(args).to(args.device)
        model_path = os.path.join('output', args.alg, args.dataset, args.model+'-{}.pt'.format(p+1))
        model.load_state_dict(torch.load(model_path))
        acc_, result_ = test(model, tgt_loader, args)
        acc.append(acc_.data.cpu())
        result.append(result_)
        print("finish subject {}, acc = {}".format(p+1, acc_))

    # 0 neg 1 neutral 2 pos
    mtx = np.zeros((3,3))
    for re in result:
        # 对第re个subject
        for (y_pred, y) in re:
            # 对第_个tgt data loader
            for i in range(y_pred.shape[0]):
                mtx[y[i]][y_pred[i]] += 1
    for i in range(mtx.shape[0]):
        mtx[i] = mtx[i]/mtx[i].sum()*100
    print(mtx)

    sns.set(font_scale=1.8)
    plt.rc('font',family='Times New Roman',size=16)
    fig_path = os.path.join('figure', args.alg+'_'+args.dataset+'.jpg')
    ax = sns.heatmap(mtx, fmt='.2f',xticklabels=['negative','neutral','positive'],yticklabels=['negative','neutral','positive'],annot=True,annot_kws={'size':20},cmap='Blues')

    sf = ax.get_figure()
    sf.savefig(fig_path, dpi=800)


    acc_mean=np.mean(acc)
    std=np.std(acc)
    print(args.dataset + ' '+ args.alg)
    print(acc_mean)
    print(std)
    print('----------')


def test(model, tgt_loader, args):
    model.eval()
    correct = 0.
    result = []
    with torch.no_grad():
        for x, y in tgt_loader:
            x, y = x.to(args.device), y.to(args.device)
            y_pred = torch.max(model.predict(x), 1)[1]
            correct += torch.sum(y_pred==y)
            result.append((y_pred, y))
    acc = correct / len(tgt_loader.dataset)
    return acc, result


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--normalize", type=str, default='standard', choices=['standard', 'minmax', 'no'])
    parser.add_argument('--dataset', type=str, default='SEED', choices=['SEED', 'SEED-IV', 'SEED-V'])
    parser.add_argument('--method', type=str, default='baseline')
    parser.add_argument('--alg', type=str, default='baseline')
    parser.add_argument('--model', type=str, default='CNN')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')    
    parser.add_argument('--n', type=int, default=5, help="MDDA n")

    parser.add_argument('--epoch', type=int, default=40, help="max iterations")
    parser.add_argument('--steps_per_epoch', type=int, default=40, help="max iterations")

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

