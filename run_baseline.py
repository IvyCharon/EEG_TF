import numpy as np
import pickle
from sklearn import preprocessing, svm
import argparse
import os
import scipy.io as scio
import re

from torch.utils.data import DataLoader
from data_process.eegdataload import EEGDataset

from tool import print_args, get_model
import file_info as fi
import torch

total_subjects_number = {
    'SEED': 15,
    'SEED-IV': 15,
    'SEED-V': 16,
}

def main(args):
    if args.subject == 0:
        acc_=[]
        for i in range(1,total_subjects_number[args.dataset]+1):
            a=train_test(i, args.normalize, args.dataset)
            acc_.append(a)
        acc=np.mean(acc_)
        std=np.std(acc_)
    else:
        acc, all_c=train_test(args.subject, args.normalize, args.dataset)
    
    path=os.path.join('output','output.txt')
    with open(path,'a') as f:
        # exp info
        f.write('-----\n')
        f.write('dataset: {}\n'.format(args.dataset))
        f.write('model: {}\n'.format(args.model))
        # results
        if args.subject==0:
            f.write('{}\n'.format(acc_))
        f.write('Acc: {}\n'.format(acc))
        f.write('Std: {}\n'.format(std))
        f.write('-----\n\n')


def normalize(data, type):
    if type=='standard':
        scaler = preprocessing.StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)
    elif type=='minmax':
        data = (data-data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    elif type=='no':
        pass
    else:
        raise RuntimeError('Wrong normalization type')
    return data

def data_process(subject_no, normalize_type, p, dataset, model):    # p = 0, 1, 2
    if dataset=='SEED-V':
        if model == 'MLP':
            return data_process_SEED_V(subject_no, normalize_type, p)
        elif model == 'CNN':
            return data_process_SEED_V_spatial(subject_no, normalize_type, p)
        elif model == 'SST_AGCN':
            return data_process_SEED_V_SST(subject_no, normalize_type, p)
    elif dataset=='SEED-IV':
        if model == 'MLP':
            return data_process_SEED_IV(subject_no, normalize_type, p)
        elif model == 'CNN':
            return data_process_SEED_IV_spatial(subject_no, normalize_type, p)
        elif model == 'SST_AGCN':
            return data_process_SEED_IV_SST(subject_no, normalize_type, p)
    elif dataset=='SEED':
        if model == 'MLP':
            return data_process_SEED(subject_no, normalize_type, p)
        elif model == 'CNN':
            return data_process_SEED_spatial(subject_no, normalize_type, p)
        elif model == 'SST_AGCN':
            return data_process_SEED_SST(subject_no, normalize_type, p)
    else:
        raise RuntimeError('Wrong dataset')


def data_process_SEED(subject_no, normalize_type, p):    # p = 0, 1, 2
    # load data
    # for one subject
    # subject_no = args.subject
    data_npz = []
    path=fi.dataset_path['SEED']
    for i in range(3):
        target_path=fi.SEED[subject_no][i]
        data_npz.append(scio.loadmat(path+target_path))

    # dict_keys(['__header__', '__version__', '__globals__', 
    # ['de_movingAve1', 'de_LDS1', 'psd_movingAve1', 'psd_LDS1', ] * 15 )
    # movingAve and LDS are 2 types of smoothing method
    data=[]
    for i in range(3):
        session=data_npz[i]
        data_=session['de_LDS1'].transpose(1,0,2)
        for tr in range(2,16):
            data_ = np.concatenate((data_, session['de_LDS{}'.format(tr)].transpose(1,0,2)))
        data_=data_.reshape((data_.shape[0],-1))
        data_=normalize(data_, normalize_type)
        data.append(data_)
    data = np.concatenate((data[0], data[1], data[2]))
    
    # session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3];
    # session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1];
    # session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0];
    #################
    length = []
    len_part = 0
    session_len = [0]
    for s in range(3):
        session=data_npz[s]
        for tr in range(1,16):
            d=session['de_LDS{}'.format(tr)]
            length.append(d.shape[1])
            len_part=len_part+d.shape[1]
            if tr%5==0:
                session_len.append(len_part)
    assert np.sum(length)==data.shape[0]==len_part
        
    # label
    label_path = path+'label.mat'
    label_dict = scio.loadmat(label_path)
    label=[]
    all_labels = label_dict['label'][0]
    all_labels = np.concatenate((all_labels, all_labels, all_labels))
    for i in range(len(all_labels)):
        for _ in range(length[i]):
            label.append(all_labels[i])
    label=np.array(label)+1
    assert label.shape[0]==data.shape[0]
    

    if p==0:
        test_data=np.concatenate((data[session_len[0]:session_len[1],:], data[session_len[3]:session_len[4],:], data[session_len[6]:session_len[7],:]))
        test_label=np.concatenate((label[session_len[0]:session_len[1]], label[session_len[3]:session_len[4]], label[session_len[6]:session_len[7]]))
        train_data=np.concatenate((data[session_len[1]:session_len[3],:], data[session_len[4]:session_len[6],:], data[session_len[7]:session_len[9],:]))
        train_label=np.concatenate((label[session_len[1]:session_len[3]], label[session_len[4]:session_len[6]], label[session_len[7]:session_len[9]]))
    elif p==1:
        test_data=np.concatenate((data[session_len[1]:session_len[2],:], data[session_len[4]:session_len[5],:], data[session_len[7]:session_len[8],:]))
        test_label=np.concatenate((label[session_len[1]:session_len[2]], label[session_len[4]:session_len[5]], label[session_len[7]:session_len[8]]))
        train_data=np.concatenate((data[session_len[0]:session_len[1],:], data[session_len[2]:session_len[4],:], data[session_len[5]:session_len[7],:], data[session_len[8]:session_len[9],:]))
        train_label=np.concatenate((label[session_len[0]:session_len[1]], label[session_len[2]:session_len[4]], label[session_len[5]:session_len[7]], label[session_len[8]:session_len[9]]))
    elif p==2:
        test_data=np.concatenate((data[session_len[2]:session_len[3],:], data[session_len[5]:session_len[6],:], data[session_len[8]:session_len[9],:]))
        test_label=np.concatenate((label[session_len[2]:session_len[3]], label[session_len[5]:session_len[6]], label[session_len[8]:session_len[9]]))
        train_data=np.concatenate((data[session_len[0]:session_len[2],:], data[session_len[3]:session_len[5],:], data[session_len[6]:session_len[8],:]))
        train_label=np.concatenate((label[session_len[0]:session_len[2]], label[session_len[3]:session_len[5]], label[session_len[6]:session_len[8]]))
    
    return train_data, train_label, test_data, test_label 

def data_process_SEED_spatial(subject_no, normalize_type, p):
    data_npz = []
    path=fi.dataset_path['SEED']
    for i in range(3):
        target_path=fi.SEED[subject_no][i]
        data_npz.append(scio.loadmat(path+target_path))

    data=[]
    for i in range(3):
        session=data_npz[i]
        data_=session['de_LDS1'].transpose(1,0,2)
        for tr in range(2,16):
            data_ = np.concatenate((data_, session['de_LDS{}'.format(tr)].transpose(1,0,2)))
        shape=data_.shape
        data_=data_.reshape((data_.shape[0],-1))
        data_=normalize(data_, normalize_type).reshape(shape)
        data.append(data_)
    data = np.concatenate((data[0], data[1], data[2])) # [l, 62, 5]

    length = []
    len_part = 0
    session_len = [0]
    for s in range(3):
        session=data_npz[s]
        for tr in range(1,16):
            d=session['de_LDS{}'.format(tr)]
            length.append(d.shape[1])
            len_part=len_part+d.shape[1]
            if tr%5==0:
                session_len.append(len_part)
    assert np.sum(length)==data.shape[0]==len_part
    
    # label
    label_path = path+'label.mat'
    label_dict = scio.loadmat(label_path)
    all_labels = label_dict['label'][0]
    all_labels = np.concatenate((all_labels, all_labels, all_labels))
    label=[]
    for i in range(len(all_labels)):
        for _ in range(length[i]):
            label.append(all_labels[i])
    label=np.array(label)+1     # [l, 1]

    l = data.shape[0]
    data_out = np.zeros((9,9,l,5))
    data = data.transpose(1,0,2)    # [62, l, 5]
    for i in range(62):
        sli = data[i]
        data_out[np.where(fi.spatial == fi.channel_order[i])]=sli

    data = data_out.transpose(2, 3, 0, 1)   # [l, 5, 9, 9]

    if p==0:
        test_data=np.concatenate((data[session_len[0]:session_len[1],:], data[session_len[3]:session_len[4],:], data[session_len[6]:session_len[7],:]))
        test_label=np.concatenate((label[session_len[0]:session_len[1]], label[session_len[3]:session_len[4]], label[session_len[6]:session_len[7]]))
        train_data=np.concatenate((data[session_len[1]:session_len[3],:], data[session_len[4]:session_len[6],:], data[session_len[7]:session_len[9],:]))
        train_label=np.concatenate((label[session_len[1]:session_len[3]], label[session_len[4]:session_len[6]], label[session_len[7]:session_len[9]]))
    elif p==1:
        test_data=np.concatenate((data[session_len[1]:session_len[2],:], data[session_len[4]:session_len[5],:], data[session_len[7]:session_len[8],:]))
        test_label=np.concatenate((label[session_len[1]:session_len[2]], label[session_len[4]:session_len[5]], label[session_len[7]:session_len[8]]))
        train_data=np.concatenate((data[session_len[0]:session_len[1],:], data[session_len[2]:session_len[4],:], data[session_len[5]:session_len[7],:], data[session_len[8]:session_len[9],:]))
        train_label=np.concatenate((label[session_len[0]:session_len[1]], label[session_len[2]:session_len[4]], label[session_len[5]:session_len[7]], label[session_len[8]:session_len[9]]))
    elif p==2:
        test_data=np.concatenate((data[session_len[2]:session_len[3],:], data[session_len[5]:session_len[6],:], data[session_len[8]:session_len[9],:]))
        test_label=np.concatenate((label[session_len[2]:session_len[3]], label[session_len[5]:session_len[6]], label[session_len[8]:session_len[9]]))
        train_data=np.concatenate((data[session_len[0]:session_len[2],:], data[session_len[3]:session_len[5],:], data[session_len[6]:session_len[8],:]))
        train_label=np.concatenate((label[session_len[0]:session_len[2]], label[session_len[3]:session_len[5]], label[session_len[6]:session_len[8]]))
    
    return train_data, train_label, test_data, test_label 

def overlap(data, T):
    window_points = T
    step_points = 1
    window_num = data.shape[-1]
    data_overlap = np.zeros((62, 5,T, window_num))
    
    for window_index in range(window_num-T):
        data_win=data[:,int(step_points * window_index):int(step_points * window_index+window_points)]
        a=data_win.reshape(62, 5,T)
        data_overlap[:,:,:, window_index] = a
    for ii in range(window_num-T,window_num):
        data_win = np.zeros((310, T))
        data_win[:, 0:window_num-ii] = data[:,int(step_points * ii):]
        for k in range(window_num-ii, T):
            data_win[:, k] = data[:,-1]
        a=data_win.reshape(62, 5,T)
        data_overlap[:,:,:, ii] = a
    return data_overlap

def data_process_SEED_SST(subject_no, normalize_type, p):
    data_npz = []
    path=fi.dataset_path['SEED']
    for i in range(3):
        target_path=fi.SEED[subject_no][i]
        data_npz.append(scio.loadmat(path+target_path))

    data=[]
    for i in range(3):
        session=data_npz[i]
        data_=session['de_LDS1'].transpose(1,0,2)
        for tr in range(2,16):
            data_ = np.concatenate((data_, session['de_LDS{}'.format(tr)].transpose(1,0,2)))
        shape=data_.shape
        data_=data_.reshape((data_.shape[0],-1))
        data_=normalize(data_, normalize_type).reshape(shape)
        data.append(data_)
    data = np.concatenate((data[0], data[1], data[2])) # [l, 62, 5]

    data = data.transpose(1, 2, 0).reshape((310, -1))   # [310, l]
    data = overlap(data, T=5)   # [62, 5, T, l]
    data = data.transpose(3, 1, 2, 0)   # [l, 5, T, 62]
    s = data.shape
    data = data.reshape((s[0], s[1], s[2], s[3], 1))

    length = []
    len_part = 0
    session_len = [0]
    for s in range(3):
        session=data_npz[s]
        for tr in range(1,16):
            d=session['de_LDS{}'.format(tr)]
            length.append(d.shape[1])
            len_part=len_part+d.shape[1]
            if tr%5==0:
                session_len.append(len_part)
    assert np.sum(length)==data.shape[0]==len_part
    
    # label
    label_path = path+'label.mat'
    label_dict = scio.loadmat(label_path)
    all_labels = label_dict['label'][0]
    all_labels = np.concatenate((all_labels, all_labels, all_labels))
    label=[]
    for i in range(len(all_labels)):
        for _ in range(length[i]):
            label.append(all_labels[i])
    label=np.array(label)+1     # [l, 1]

    if p==0:
        test_data=np.concatenate((data[session_len[0]:session_len[1],:], data[session_len[3]:session_len[4],:], data[session_len[6]:session_len[7],:]))
        test_label=np.concatenate((label[session_len[0]:session_len[1]], label[session_len[3]:session_len[4]], label[session_len[6]:session_len[7]]))
        train_data=np.concatenate((data[session_len[1]:session_len[3],:], data[session_len[4]:session_len[6],:], data[session_len[7]:session_len[9],:]))
        train_label=np.concatenate((label[session_len[1]:session_len[3]], label[session_len[4]:session_len[6]], label[session_len[7]:session_len[9]]))
    elif p==1:
        test_data=np.concatenate((data[session_len[1]:session_len[2],:], data[session_len[4]:session_len[5],:], data[session_len[7]:session_len[8],:]))
        test_label=np.concatenate((label[session_len[1]:session_len[2]], label[session_len[4]:session_len[5]], label[session_len[7]:session_len[8]]))
        train_data=np.concatenate((data[session_len[0]:session_len[1],:], data[session_len[2]:session_len[4],:], data[session_len[5]:session_len[7],:], data[session_len[8]:session_len[9],:]))
        train_label=np.concatenate((label[session_len[0]:session_len[1]], label[session_len[2]:session_len[4]], label[session_len[5]:session_len[7]], label[session_len[8]:session_len[9]]))
    elif p==2:
        test_data=np.concatenate((data[session_len[2]:session_len[3],:], data[session_len[5]:session_len[6],:], data[session_len[8]:session_len[9],:]))
        test_label=np.concatenate((label[session_len[2]:session_len[3]], label[session_len[5]:session_len[6]], label[session_len[8]:session_len[9]]))
        train_data=np.concatenate((data[session_len[0]:session_len[2],:], data[session_len[3]:session_len[5],:], data[session_len[6]:session_len[8],:]))
        train_label=np.concatenate((label[session_len[0]:session_len[2]], label[session_len[3]:session_len[5]], label[session_len[6]:session_len[8]]))
    

    return train_data, train_label, test_data, test_label 


def data_process_SEED_IV(subject_no, normalize_type, p):    # p = 0, 1, 2
    # load data
    # for one subject
    # subject_no = args.subject
    data_npz = []
    path=fi.dataset_path['SEED-IV']
    for i in range(3):
        pa=path+'{}/'.format(i+1)
        target_path=fi.SEED_IV[subject_no][i]
        data_npz.append(scio.loadmat(pa+target_path))

    # dict_keys(['__header__', '__version__', '__globals__', 
    # ['de_movingAve1', 'de_LDS1', 'psd_movingAve1', 'psd_LDS1', ] * 24 )
    # movingAve and LDS are 2 types of smoothing method
    data=[]
    for i in range(3):
        session=data_npz[i]
        data_=session['de_LDS1'].transpose(1,0,2)
        for tr in range(2,25):
            data_ = np.concatenate((data_, session['de_LDS{}'.format(tr)].transpose(1,0,2)))
        data_=data_.reshape((data_.shape[0],-1))
        data_=normalize(data_, normalize_type)
        data.append(data_)
    data = np.concatenate((data[0], data[1], data[2]))
    
    # session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3];
    # session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1];
    # session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0];
    #################
    length = []
    len_part = 0
    session_len = [0]
    for s in range(3):
        session=data_npz[s]
        for tr in range(1,25):
            d=session['de_LDS{}'.format(tr)]
            length.append(d.shape[1])
            len_part=len_part+d.shape[1]
            if tr%8==0:
                session_len.append(len_part)
    assert np.sum(length)==data.shape[0]==len_part
        
    # label
    label=[]
    all_labels = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3, 2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1, 1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
    for i in range(len(all_labels)):
        for _ in range(length[i]):
            label.append(all_labels[i])
    label=np.array(label)
    assert label.shape[0]==data.shape[0]

    if p==0:
        test_data=np.concatenate((data[session_len[0]:session_len[1],:], data[session_len[3]:session_len[4],:], data[session_len[6]:session_len[7],:]))
        test_label=np.concatenate((label[session_len[0]:session_len[1]], label[session_len[3]:session_len[4]], label[session_len[6]:session_len[7]]))
        train_data=np.concatenate((data[session_len[1]:session_len[3],:], data[session_len[4]:session_len[6],:], data[session_len[7]:session_len[9],:]))
        train_label=np.concatenate((label[session_len[1]:session_len[3]], label[session_len[4]:session_len[6]], label[session_len[7]:session_len[9]]))
    elif p==1:
        test_data=np.concatenate((data[session_len[1]:session_len[2],:], data[session_len[4]:session_len[5],:], data[session_len[7]:session_len[8],:]))
        test_label=np.concatenate((label[session_len[1]:session_len[2]], label[session_len[4]:session_len[5]], label[session_len[7]:session_len[8]]))
        train_data=np.concatenate((data[session_len[0]:session_len[1],:], data[session_len[2]:session_len[4],:], data[session_len[5]:session_len[7],:], data[session_len[8]:session_len[9],:]))
        train_label=np.concatenate((label[session_len[0]:session_len[1]], label[session_len[2]:session_len[4]], label[session_len[5]:session_len[7]], label[session_len[8]:session_len[9]]))
    elif p==2:
        test_data=np.concatenate((data[session_len[2]:session_len[3],:], data[session_len[5]:session_len[6],:], data[session_len[8]:session_len[9],:]))
        test_label=np.concatenate((label[session_len[2]:session_len[3]], label[session_len[5]:session_len[6]], label[session_len[8]:session_len[9]]))
        train_data=np.concatenate((data[session_len[0]:session_len[2],:], data[session_len[3]:session_len[5],:], data[session_len[6]:session_len[8],:]))
        train_label=np.concatenate((label[session_len[0]:session_len[2]], label[session_len[3]:session_len[5]], label[session_len[6]:session_len[8]]))
    
    return train_data, train_label, test_data, test_label 

def data_process_SEED_IV_spatial(subject_no, normalize_type, p):
    # load data for one subject
    data_npz = []
    path=fi.dataset_path['SEED-IV']
    for i in range(3):
        pa=path+'{}/'.format(i+1)
        target_path=fi.SEED_IV[subject_no][i]
        data_npz.append(scio.loadmat(pa+target_path))
    data=[]
    for i in range(3):
        session=data_npz[i]
        data_=session['de_LDS1'].transpose(1,0,2)
        for tr in range(2,25):
            data_ = np.concatenate((data_, session['de_LDS{}'.format(tr)].transpose(1,0,2)))
        shape=data_.shape
        data_=data_.reshape((data_.shape[0],-1))
        data_=normalize(data_, normalize_type).reshape(shape)
        data.append(data_)
    data = np.concatenate((data[0], data[1], data[2]))  # [l, 62, 5]
    
    length = []
    len_part = 0
    session_len = [0]
    for s in range(3):
        session=data_npz[s]
        for tr in range(1,25):
            d=session['de_LDS{}'.format(tr)]
            length.append(d.shape[1])
            len_part=len_part+d.shape[1]
            if tr%8==0:
                session_len.append(len_part)
    assert np.sum(length)==data.shape[0]==len_part
        
    # label
    label=[]
    all_labels = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3, 
                  2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1, 
                  1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
    label=[]
    for i in range(len(all_labels)):
        for _ in range(length[i]):
            label.append(all_labels[i])
    label=np.array(label)
    assert label.shape[0]==data.shape[0]

    l = data.shape[0]
    data_out = np.zeros((9,9,l,5))
    data = data.transpose(1,0,2)    # [62, l, 5]
    for i in range(62):
        sli = data[i]
        data_out[np.where(fi.spatial == fi.channel_order[i])]=sli
    data = data_out.transpose(2, 3, 0, 1)   # [l, 5, 9, 9]
        
    if p==0:
        test_data=np.concatenate((data[session_len[0]:session_len[1],:], data[session_len[3]:session_len[4],:], data[session_len[6]:session_len[7],:]))
        test_label=np.concatenate((label[session_len[0]:session_len[1]], label[session_len[3]:session_len[4]], label[session_len[6]:session_len[7]]))
        train_data=np.concatenate((data[session_len[1]:session_len[3],:], data[session_len[4]:session_len[6],:], data[session_len[7]:session_len[9],:]))
        train_label=np.concatenate((label[session_len[1]:session_len[3]], label[session_len[4]:session_len[6]], label[session_len[7]:session_len[9]]))
    elif p==1:
        test_data=np.concatenate((data[session_len[1]:session_len[2],:], data[session_len[4]:session_len[5],:], data[session_len[7]:session_len[8],:]))
        test_label=np.concatenate((label[session_len[1]:session_len[2]], label[session_len[4]:session_len[5]], label[session_len[7]:session_len[8]]))
        train_data=np.concatenate((data[session_len[0]:session_len[1],:], data[session_len[2]:session_len[4],:], data[session_len[5]:session_len[7],:], data[session_len[8]:session_len[9],:]))
        train_label=np.concatenate((label[session_len[0]:session_len[1]], label[session_len[2]:session_len[4]], label[session_len[5]:session_len[7]], label[session_len[8]:session_len[9]]))
    elif p==2:
        test_data=np.concatenate((data[session_len[2]:session_len[3],:], data[session_len[5]:session_len[6],:], data[session_len[8]:session_len[9],:]))
        test_label=np.concatenate((label[session_len[2]:session_len[3]], label[session_len[5]:session_len[6]], label[session_len[8]:session_len[9]]))
        train_data=np.concatenate((data[session_len[0]:session_len[2],:], data[session_len[3]:session_len[5],:], data[session_len[6]:session_len[8],:]))
        train_label=np.concatenate((label[session_len[0]:session_len[2]], label[session_len[3]:session_len[5]], label[session_len[6]:session_len[8]]))
    
    return train_data, train_label, test_data, test_label 

def data_process_SEED_IV_SST(subject_no, normalize_type, p):
    # load data for one subject
    data_npz = []
    path=fi.dataset_path['SEED-IV']
    for i in range(3):
        pa=path+'{}/'.format(i+1)
        target_path=fi.SEED_IV[subject_no][i]
        data_npz.append(scio.loadmat(pa+target_path))
    data=[]
    for i in range(3):
        session=data_npz[i]
        data_=session['de_LDS1'].transpose(1,0,2)
        for tr in range(2,25):
            data_ = np.concatenate((data_, session['de_LDS{}'.format(tr)].transpose(1,0,2)))
        shape=data_.shape
        data_=data_.reshape((data_.shape[0],-1))
        data_=normalize(data_, normalize_type).reshape(shape)
        data.append(data_)
    data = np.concatenate((data[0], data[1], data[2]))  # [l, 62, 5]

    data = data.transpose(1, 2, 0).reshape((310, -1))   # [310, l]
    data = overlap(data, T=5)   # [62, 5, T, l]
    data = data.transpose(3, 1, 2, 0)   # [l, 5, T, 62]
    s = data.shape
    data = data.reshape((s[0], s[1], s[2], s[3], 1))
    
    length = []
    len_part = 0
    session_len = [0]
    for s in range(3):
        session=data_npz[s]
        for tr in range(1,25):
            d=session['de_LDS{}'.format(tr)]
            length.append(d.shape[1])
            len_part=len_part+d.shape[1]
            if tr%8==0:
                session_len.append(len_part)
    assert np.sum(length)==data.shape[0]==len_part
        
    # label
    label=[]
    all_labels = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3, 
                  2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1, 
                  1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
    label=[]
    for i in range(len(all_labels)):
        for _ in range(length[i]):
            label.append(all_labels[i])
    label=np.array(label)

    if p==0:
        test_data=np.concatenate((data[session_len[0]:session_len[1],:], data[session_len[3]:session_len[4],:], data[session_len[6]:session_len[7],:]))
        test_label=np.concatenate((label[session_len[0]:session_len[1]], label[session_len[3]:session_len[4]], label[session_len[6]:session_len[7]]))
        train_data=np.concatenate((data[session_len[1]:session_len[3],:], data[session_len[4]:session_len[6],:], data[session_len[7]:session_len[9],:]))
        train_label=np.concatenate((label[session_len[1]:session_len[3]], label[session_len[4]:session_len[6]], label[session_len[7]:session_len[9]]))
    elif p==1:
        test_data=np.concatenate((data[session_len[1]:session_len[2],:], data[session_len[4]:session_len[5],:], data[session_len[7]:session_len[8],:]))
        test_label=np.concatenate((label[session_len[1]:session_len[2]], label[session_len[4]:session_len[5]], label[session_len[7]:session_len[8]]))
        train_data=np.concatenate((data[session_len[0]:session_len[1],:], data[session_len[2]:session_len[4],:], data[session_len[5]:session_len[7],:], data[session_len[8]:session_len[9],:]))
        train_label=np.concatenate((label[session_len[0]:session_len[1]], label[session_len[2]:session_len[4]], label[session_len[5]:session_len[7]], label[session_len[8]:session_len[9]]))
    elif p==2:
        test_data=np.concatenate((data[session_len[2]:session_len[3],:], data[session_len[5]:session_len[6],:], data[session_len[8]:session_len[9],:]))
        test_label=np.concatenate((label[session_len[2]:session_len[3]], label[session_len[5]:session_len[6]], label[session_len[8]:session_len[9]]))
        train_data=np.concatenate((data[session_len[0]:session_len[2],:], data[session_len[3]:session_len[5],:], data[session_len[6]:session_len[8],:]))
        train_label=np.concatenate((label[session_len[0]:session_len[2]], label[session_len[3]:session_len[5]], label[session_len[6]:session_len[8]]))
    
    return train_data, train_label, test_data, test_label 



def data_process_SEED_V(subject_no, normalize_type, p):    # p = 0, 1, 2
    # load data
    # for one subject
    # subject_no = args.subject
    path=fi.dataset_path['SEED-V']
    path=os.path.join(path, "{}_123.npz".format(subject_no))
    data_npz = np.load(path,allow_pickle=True)
    data_dict = pickle.loads(data_npz['data'])   # dict
    label_dict = pickle.loads(data_npz['label']) # dict
    # keys: [0, 1, ..., 44]
    # 3 sessions, 15 trials, 3 * 15 = 45
    # in each session and each trial, data shape (i, 310), label shape (i,)
    # (i, 310), 310 = 62 channels * 5 freq_band

    data1, data2, data3 = data_dict[0], data_dict[15], data_dict[30]
    label1, label2, label3 = label_dict[0], label_dict[15], label_dict[30]
    for i in range(1,15):
        data1 = np.concatenate((data1, data_dict[i]))
        label1 = np.concatenate((label1, label_dict[i]))
    data1 = normalize(data1, normalize_type)
    
    for i in range(16,30):
        data2 = np.concatenate((data2, data_dict[i]))
        label2 = np.concatenate((label2, label_dict[i]))
    data2 = normalize(data2, normalize_type)
    
    for i in range(31,45):
        data3 = np.concatenate((data3, data_dict[i]))
        label3 = np.concatenate((label3, label_dict[i]))
    data3 = normalize(data3, normalize_type)
    
    data = np.concatenate((data1, data2, data3))
    label = np.concatenate((label1, label2, label3))
    
    len = [data_dict[0].shape[0]]
    len_part = data_dict[0].shape[0]
    session_len = [0]
    
    for i in range(1, 45):
        len.append(data_dict[i].shape[0])
        len_part = len_part+data_dict[i].shape[0]
        if i%5==4:
            session_len.append(len_part)
        
    # seperate train data and test data
    # p, p+3, p+6 -> 3 part for test
    # [183, 439, 681, 866, 1051, 1222, 1386, 1612, 1823]
    # [0, 183), [183, 439)
    if p==0:
        test_data=np.concatenate((data[session_len[0]:session_len[1],:], data[session_len[3]:session_len[4],:], data[session_len[6]:session_len[7],:]))
        test_label=np.concatenate((label[session_len[0]:session_len[1]], label[session_len[3]:session_len[4]], label[session_len[6]:session_len[7]]))
        train_data=np.concatenate((data[session_len[1]:session_len[3],:], data[session_len[4]:session_len[6],:], data[session_len[7]:session_len[9],:]))
        train_label=np.concatenate((label[session_len[1]:session_len[3]], label[session_len[4]:session_len[6]], label[session_len[7]:session_len[9]]))
    elif p==1:
        test_data=np.concatenate((data[session_len[1]:session_len[2],:], data[session_len[4]:session_len[5],:], data[session_len[7]:session_len[8],:]))
        test_label=np.concatenate((label[session_len[1]:session_len[2]], label[session_len[4]:session_len[5]], label[session_len[7]:session_len[8]]))
        train_data=np.concatenate((data[session_len[0]:session_len[1],:], data[session_len[2]:session_len[4],:], data[session_len[5]:session_len[7],:], data[session_len[8]:session_len[9],:]))
        train_label=np.concatenate((label[session_len[0]:session_len[1]], label[session_len[2]:session_len[4]], label[session_len[5]:session_len[7]], label[session_len[8]:session_len[9]]))
    elif p==2:
        test_data=np.concatenate((data[session_len[2]:session_len[3],:], data[session_len[5]:session_len[6],:], data[session_len[8]:session_len[9],:]))
        test_label=np.concatenate((label[session_len[2]:session_len[3]], label[session_len[5]:session_len[6]], label[session_len[8]:session_len[9]]))
        train_data=np.concatenate((data[session_len[0]:session_len[2],:], data[session_len[3]:session_len[5],:], data[session_len[6]:session_len[8],:]))
        train_label=np.concatenate((label[session_len[0]:session_len[2]], label[session_len[3]:session_len[5]], label[session_len[6]:session_len[8]]))
    
    return train_data, train_label, test_data, test_label 

def data_process_SEED_V_spatial(subject_no, normalize_type, p):    # data: [l, 5, 9, 9]
    # load data for one subject
    path=fi.dataset_path['SEED-V']
    path=os.path.join(path, "{}_123.npz".format(subject_no))
    data_npz = np.load(path,allow_pickle=True)
    data_dict = pickle.loads(data_npz['data'])   # dict
    label_dict = pickle.loads(data_npz['label']) # dict
    
    data1, data2, data3 = data_dict[0], data_dict[15], data_dict[30]
    label1, label2, label3 = label_dict[0], label_dict[15], label_dict[30]
    for i in range(1,15):
        data1 = np.concatenate((data1, data_dict[i]))
        label1 = np.concatenate((label1, label_dict[i]))
    data1 = normalize(data1, normalize_type)
    
    for i in range(16,30):
        data2 = np.concatenate((data2, data_dict[i]))
        label2 = np.concatenate((label2, label_dict[i]))
    data2 = normalize(data2, normalize_type)
    
    for i in range(31,45):
        data3 = np.concatenate((data3, data_dict[i]))
        label3 = np.concatenate((label3, label_dict[i]))
    data3 = normalize(data3, normalize_type)
    
    data = np.concatenate((data1, data2, data3))
    label = np.concatenate((label1, label2, label3))

    data=data.reshape((data.shape[0], 62, 5)).transpose(1,0,2)  # [62, l, 5]
    l = data.shape[1]
    data_out = np.zeros((9,9,l,5))
    for i in range(62):
        sli = data[i]
        data_out[np.where(fi.spatial == fi.channel_order[i])]=sli
    data = data_out.transpose(2, 3, 0, 1)   # [l, 5, 9, 9]
    
    len = [data_dict[0].shape[0]]
    len_part = data_dict[0].shape[0]
    session_len = [0]
    for i in range(1, 45):
        len.append(data_dict[i].shape[0])
        len_part = len_part+data_dict[i].shape[0]
        if i%5==4:
            session_len.append(len_part)
        
    # seperate train data and test data
    # p, p+3, p+6 -> 3 part for test
    # [183, 439, 681, 866, 1051, 1222, 1386, 1612, 1823]
    # [0, 183), [183, 439)
    if p==0:
        test_data=np.concatenate((data[session_len[0]:session_len[1],:], data[session_len[3]:session_len[4],:], data[session_len[6]:session_len[7],:]))
        test_label=np.concatenate((label[session_len[0]:session_len[1]], label[session_len[3]:session_len[4]], label[session_len[6]:session_len[7]]))
        train_data=np.concatenate((data[session_len[1]:session_len[3],:], data[session_len[4]:session_len[6],:], data[session_len[7]:session_len[9],:]))
        train_label=np.concatenate((label[session_len[1]:session_len[3]], label[session_len[4]:session_len[6]], label[session_len[7]:session_len[9]]))
    elif p==1:
        test_data=np.concatenate((data[session_len[1]:session_len[2],:], data[session_len[4]:session_len[5],:], data[session_len[7]:session_len[8],:]))
        test_label=np.concatenate((label[session_len[1]:session_len[2]], label[session_len[4]:session_len[5]], label[session_len[7]:session_len[8]]))
        train_data=np.concatenate((data[session_len[0]:session_len[1],:], data[session_len[2]:session_len[4],:], data[session_len[5]:session_len[7],:], data[session_len[8]:session_len[9],:]))
        train_label=np.concatenate((label[session_len[0]:session_len[1]], label[session_len[2]:session_len[4]], label[session_len[5]:session_len[7]], label[session_len[8]:session_len[9]]))
    elif p==2:
        test_data=np.concatenate((data[session_len[2]:session_len[3],:], data[session_len[5]:session_len[6],:], data[session_len[8]:session_len[9],:]))
        test_label=np.concatenate((label[session_len[2]:session_len[3]], label[session_len[5]:session_len[6]], label[session_len[8]:session_len[9]]))
        train_data=np.concatenate((data[session_len[0]:session_len[2],:], data[session_len[3]:session_len[5],:], data[session_len[6]:session_len[8],:]))
        train_label=np.concatenate((label[session_len[0]:session_len[2]], label[session_len[3]:session_len[5]], label[session_len[6]:session_len[8]]))
    
    return train_data, train_label, test_data, test_label 

def data_process_SEED_V_SST(subject_no, normalize_type, p):
    # load data for one subject
    path=fi.dataset_path['SEED-V']
    path=os.path.join(path, "{}_123.npz".format(subject_no))
    data_npz = np.load(path,allow_pickle=True)
    data_dict = pickle.loads(data_npz['data'])   # dict
    label_dict = pickle.loads(data_npz['label']) # dict
    
    data1, data2, data3 = data_dict[0], data_dict[15], data_dict[30]
    label1, label2, label3 = label_dict[0], label_dict[15], label_dict[30]
    for i in range(1,15):
        data1 = np.concatenate((data1, data_dict[i]))
        label1 = np.concatenate((label1, label_dict[i]))
    data1 = normalize(data1, normalize_type)
    
    for i in range(16,30):
        data2 = np.concatenate((data2, data_dict[i]))
        label2 = np.concatenate((label2, label_dict[i]))
    data2 = normalize(data2, normalize_type)
    
    for i in range(31,45):
        data3 = np.concatenate((data3, data_dict[i]))
        label3 = np.concatenate((label3, label_dict[i]))
    data3 = normalize(data3, normalize_type)
    
    data = np.concatenate((data1, data2, data3))
    label = np.concatenate((label1, label2, label3))

    data=data.reshape((data.shape[0], 62, 5))
    data = data.transpose(1, 2, 0).reshape((310, -1))   # [310, l]
    data = overlap(data, T=5)   # [62, 5, T, l]
    data = data.transpose(3, 1, 2, 0)   # [l, 5, T, 62]
    s = data.shape
    data = data.reshape((s[0], s[1], s[2], s[3], 1))

    len = [data_dict[0].shape[0]]
    len_part = data_dict[0].shape[0]
    session_len = [0]
    for i in range(1, 45):
        len.append(data_dict[i].shape[0])
        len_part = len_part+data_dict[i].shape[0]
        if i%5==4:
            session_len.append(len_part)
        
    # seperate train data and test data
    # p, p+3, p+6 -> 3 part for test
    # [183, 439, 681, 866, 1051, 1222, 1386, 1612, 1823]
    # [0, 183), [183, 439)
    if p==0:
        test_data=np.concatenate((data[session_len[0]:session_len[1],:], data[session_len[3]:session_len[4],:], data[session_len[6]:session_len[7],:]))
        test_label=np.concatenate((label[session_len[0]:session_len[1]], label[session_len[3]:session_len[4]], label[session_len[6]:session_len[7]]))
        train_data=np.concatenate((data[session_len[1]:session_len[3],:], data[session_len[4]:session_len[6],:], data[session_len[7]:session_len[9],:]))
        train_label=np.concatenate((label[session_len[1]:session_len[3]], label[session_len[4]:session_len[6]], label[session_len[7]:session_len[9]]))
    elif p==1:
        test_data=np.concatenate((data[session_len[1]:session_len[2],:], data[session_len[4]:session_len[5],:], data[session_len[7]:session_len[8],:]))
        test_label=np.concatenate((label[session_len[1]:session_len[2]], label[session_len[4]:session_len[5]], label[session_len[7]:session_len[8]]))
        train_data=np.concatenate((data[session_len[0]:session_len[1],:], data[session_len[2]:session_len[4],:], data[session_len[5]:session_len[7],:], data[session_len[8]:session_len[9],:]))
        train_label=np.concatenate((label[session_len[0]:session_len[1]], label[session_len[2]:session_len[4]], label[session_len[5]:session_len[7]], label[session_len[8]:session_len[9]]))
    elif p==2:
        test_data=np.concatenate((data[session_len[2]:session_len[3],:], data[session_len[5]:session_len[6],:], data[session_len[8]:session_len[9],:]))
        test_label=np.concatenate((label[session_len[2]:session_len[3]], label[session_len[5]:session_len[6]], label[session_len[8]:session_len[9]]))
        train_data=np.concatenate((data[session_len[0]:session_len[2],:], data[session_len[3]:session_len[5],:], data[session_len[6]:session_len[8],:]))
        train_label=np.concatenate((label[session_len[0]:session_len[2]], label[session_len[3]:session_len[5]], label[session_len[6]:session_len[8]]))
    
    return train_data, train_label, test_data, test_label 





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


def train_test(subject_no, normalize_type, dataset):
    accuracy = []
    for i in range(3):
        train_data, train_label, test_data, test_label = data_process(subject_no, normalize_type, i, dataset, args.model)

        src_loader = DataLoader(
            dataset=EEGDataset(train_data.astype(np.float32), train_label),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True
        )
        tgt_loader = DataLoader(
            dataset=EEGDataset(test_data.astype(np.float32), test_label),
            batch_size=args.batch_size,
            shuffle=False
        )

        n_batch = len(src_loader)
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
        accuracy.append(best_acc)
    print(accuracy)
    accuracy = np.mean(np.array([accuracy[i].cpu() for i in range(3)]))
    
    return accuracy
    
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
    parser.add_argument("--subject", type=int, default=0)
    parser.add_argument("--normalize", type=str, default='standard') # choose from standard, minmax, no
    parser.add_argument('--dataset', type=str, default='SEED-V') # choose from SEED, SEED-IV, SEED-V
    parser.add_argument('--epoch', type=int, default=40, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-3, help="learning rate")
    parser.add_argument('--model', type=str, default='CNN')
    parser.add_argument('--schuse', action='store_true')
    parser.add_argument('--schusech', type=str, default='cos')
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--momentum', type=float, default=0.9, help='for optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    args = parser.parse_args()

    setattr(args, "num_classes", fi.num_classes[args.dataset])
    setattr(args, "num_subject", fi.total_subjects_number[args.dataset])
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    main(args)
