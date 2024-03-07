import os
import pickle

import file_info as fi
import numpy as np
import scipy.io as scio
from data_process.eegdataload import EEGDataset
from sklearn import preprocessing
from torch.utils.data import DataLoader


def data_process(model, dataset, normalize):
    if model == 'SST_AGCN':
        if dataset == 'SEED':
            process_func = data_process_SEED_SST
        elif dataset == 'SEED-IV':
            process_func = data_process_SEED_IV_SST
        elif dataset == 'SEED-V':
            process_func = data_process_SEED_V_SST
        else:
            raise NotImplementedError()
    elif model in ['ResNet18', 'CNN']:
        if dataset == 'SEED':
            process_func = data_process_SEED_spatial
        elif dataset == 'SEED-IV':
            process_func = data_process_SEED_IV_spatial
        elif dataset == 'SEED-V':
            process_func = data_process_SEED_V_spatial
        else:
            raise NotImplementedError()
    elif model == 'MLP':
        if dataset == 'SEED':
            process_func = data_process_SEED
        elif dataset == 'SEED-IV':
            process_func = data_process_SEED_IV
        elif dataset == 'SEED-V':
            process_func = data_process_SEED_V
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    data, label = [], []
    for p in range(fi.total_subjects_number[dataset]):
        data_, label_ = process_func(p+1, normalize)
        data.append(data_)
        if dataset=='SEED':
            label_ = label_+1
        label.append(label_)

    return data, label

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

def data_process_SEED(subject_no, normalize_type):
    # load data for one subject
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
        data_=data_.reshape((data_.shape[0],-1))
        data_=normalize(data_, normalize_type)
        data.append(data_)
    data = np.concatenate((data[0], data[1], data[2]))
    
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
    label=np.array(label)
    assert label.shape[0]==data.shape[0]
    
    return data, label

def data_process_SEED_spatial(subject_no, normalize_type):
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
    for s in range(3):
        session=data_npz[s]
        for tr in range(1,16):
            d=session['de_LDS{}'.format(tr)]
            length.append(d.shape[1])
    assert np.sum(length)==data.shape[0]
    
    # label
    label_path = path+'label.mat'
    label_dict = scio.loadmat(label_path)
    all_labels = label_dict['label'][0]
    all_labels = np.concatenate((all_labels, all_labels, all_labels))
    label=[]
    for i in range(len(all_labels)):
        for _ in range(length[i]):
            label.append(all_labels[i])
    label=np.array(label)     # [l, 1]

    l = data.shape[0]
    data_out = np.zeros((9,9,l,5))
    data = data.transpose(1,0,2)    # [62, l, 5]
    for i in range(62):
        sli = data[i]
        data_out[np.where(fi.spatial == fi.channel_order[i])]=sli

    data_out = data_out.transpose(2, 3, 0, 1)   # [l, 5, 9, 9]

    return data_out, label

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

def data_process_SEED_SST(subject_no, normalize_type):
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
    for s in range(3):
        session=data_npz[s]
        for tr in range(1,16):
            d=session['de_LDS{}'.format(tr)]
            length.append(d.shape[1])
    assert np.sum(length)==data.shape[0]
    
    # label
    label_path = path+'label.mat'
    label_dict = scio.loadmat(label_path)
    all_labels = label_dict['label'][0]
    all_labels = np.concatenate((all_labels, all_labels, all_labels))
    label=[]
    for i in range(len(all_labels)):
        for _ in range(length[i]):
            label.append(all_labels[i])
    label=np.array(label)     # [l, 1]

    return data, label


def data_process_SEED_IV(subject_no, normalize_type):
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
    data = np.concatenate((data[0], data[1], data[2]))
    
    length = []
    for s in range(3):
        session=data_npz[s]
        for tr in range(1,25):
            d=session['de_LDS{}'.format(tr)]
            length.append(d.shape[1])
    assert np.sum(length)==data.shape[0] or np.sum(length)==data.shape[1]


    all_labels = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3, 
                  2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1, 
                  1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
    label=[]
    for i in range(len(all_labels)):
        l=length[i]
        for j in range(l):
            label.append(all_labels[i])
    label=np.array(label)
    data=data.reshape((data.shape[0],-1))

    return data, label

def data_process_SEED_IV_spatial(subject_no, normalize_type):
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
    for s in range(3):
        session=data_npz[s]
        for tr in range(1,25):
            d=session['de_LDS{}'.format(tr)]
            length.append(d.shape[1])
    assert np.sum(length)==data.shape[0]
        
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
    data_out = data_out.transpose(2, 3, 0, 1)   # [l, 5, 9, 9]
        
    return data_out, label

def data_process_SEED_IV_SST(subject_no, normalize_type):
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
    for s in range(3):
        session=data_npz[s]
        for tr in range(1,25):
            d=session['de_LDS{}'.format(tr)]
            length.append(d.shape[1])
    assert np.sum(length)==data.shape[0]
        
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

    return data, label

def data_process_SEED_V(subject_no, normalize_type):    # data: [l, 310]
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
    
    return data, label

def data_process_SEED_V_spatial(subject_no, normalize_type):    # data: [l, 5, 9, 9]
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
    data_out = data_out.transpose(2, 3, 0, 1)   # [l, 5, 9, 9]
    
    return data_out, label

def data_process_SEED_V_SST(subject_no, normalize_type):
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

    return data, label


def get_eeg_dataloader(tgt, data, label, dataset, batch_size, method):
    if method == 'DA':
        return get_eeg_dataloader_DA(tgt, data, label, dataset, batch_size)
    elif method == 'DG':
        return get_eeg_dataloader_DG(tgt, data, label, dataset, batch_size)
    elif method == 'baseline':
        return get_eeg_dataloader_Baseline(tgt, data, label, dataset, batch_size)
    elif method == 'DAMD':
        return get_eeg_dataloader_DAMD(tgt, data, label, dataset, batch_size)
    else:
        raise NotImplementedError()

def get_eeg_dataloader_DA(tgt, data, label, dataset, batch_size):
    # divide src, tgt data
    all_subject = list(range(1,fi.total_subjects_number[dataset]+1))
    src_sub = list(set(all_subject)-set([tgt]))
    isFirst = True
    for sub in src_sub:
        if isFirst:
            src_data, src_label = data[sub-1], label[sub-1]
            isFirst = False
        else:
            data_, label_ = data[sub-1], label[sub-1]
            src_data = np.concatenate((src_data, data_))
            src_label = np.concatenate((src_label, label_))
    tgt_data, tgt_label = data[tgt-1], label[tgt-1]

    source_loader = DataLoader(
        dataset=EEGDataset(src_data.astype(np.float32), src_label),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    
    tgt_train_loader = DataLoader(
        dataset=EEGDataset(tgt_data.astype(np.float32), tgt_label),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    iter_target = iter(tgt_train_loader)
    n_batch = len(tgt_train_loader)
    tgt_train_list = []
    for _ in range(n_batch):
        data_target, _ = next(iter_target)
        tgt_train_list.append(data_target)

    tgt_test_loader = DataLoader(
        dataset=EEGDataset(tgt_data.astype(np.float32), tgt_label),
        batch_size=batch_size,
        shuffle=False
    )

    return source_loader, tgt_train_list, tgt_test_loader

def get_eeg_dataloader_DG(tgt, data, label, dataset, batch_size):
    all_subject = list(range(1,fi.total_subjects_number[dataset]+1))
    src_sub = list(set(all_subject)-set([tgt]))
    src_list = []

    for i in src_sub:
        src_list.append(EEGDataset(data[i-1].astype(np.float32), label[i-1], i))

    src_loaders = [DataLoader(
        dataset=env,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
        for env in src_list]
    
    tgt_data, tgt_label = data[tgt-1], label[tgt-1]

    tgt_loader = DataLoader(
        dataset=EEGDataset(tgt_data.astype(np.float32), tgt_label),
        batch_size=batch_size,
        shuffle=False
    )
    
    return src_loaders, [], tgt_loader

def get_eeg_dataloader_Baseline(tgt, data, label, dataset, batch_size):
    all_subject = list(range(1,fi.total_subjects_number[dataset]+1))
    src_sub = list(set(all_subject)-set([tgt]))
    isFirst = True
    for sub in src_sub:
        if isFirst:
            src_data, src_label = data[sub-1], label[sub-1]
            isFirst = False
        else:
            data_, label_ = data[sub-1], label[sub-1]
            src_data = np.concatenate((src_data, data_))
            src_label = np.concatenate((src_label, label_))
    tgt_data, tgt_label = data[tgt-1], label[tgt-1]

    source_loader = DataLoader(
        dataset=EEGDataset(src_data.astype(np.float32), src_label),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    tgt_data, tgt_label = data[tgt-1], label[tgt-1]

    tgt_loader = DataLoader(
        dataset=EEGDataset(tgt_data.astype(np.float32), tgt_label),
        batch_size=batch_size,
        shuffle=False
    )
    
    return source_loader, [], tgt_loader

def get_eeg_dataloader_DAMD(tgt, data, label, dataset, batch_size):
    all_subject = list(range(1,fi.total_subjects_number[dataset]+1))
    src_sub = list(set(all_subject)-set([tgt]))
    src_list = []

    for i in src_sub:
        src_list.append(EEGDataset(data[i-1].astype(np.float32), label[i-1], i))

    src_loaders = [DataLoader(
        dataset=env,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
        for env in src_list]
    
    tgt_data, tgt_label = data[tgt-1], label[tgt-1]

    tgt_train_loader = DataLoader(
        dataset=EEGDataset(tgt_data.astype(np.float32), tgt_label),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    iter_target = iter(tgt_train_loader)
    n_batch = len(tgt_train_loader)
    tgt_train_list = []
    for _ in range(n_batch):
        data_target, _ = next(iter_target)
        tgt_train_list.append(data_target)


    tgt_loader = DataLoader(
        dataset=EEGDataset(tgt_data.astype(np.float32), tgt_label),
        batch_size=batch_size,
        shuffle=False
    )
    
    return src_loaders, tgt_train_list, tgt_loader

def get_eeg_dataloader_sub_n(tgt, data, label, dataset, batch_size, method, sub_n):
    # divide src, tgt data
    if method == 'DA':
        all_subject = list(range(1,fi.total_subjects_number[dataset]+1))
        if sub_n==-1:
            src_sub = list(set(all_subject)-set([tgt]))
        else:
            src_sub = list(set(all_subject)-set([tgt]))[:sub_n]
        isFirst = True
        for sub in src_sub:
            if isFirst:
                src_data, src_label = data[sub-1], label[sub-1]
                isFirst = False
            else:
                data_, label_ = data[sub-1], label[sub-1]
                src_data = np.concatenate((src_data, data_))
                src_label = np.concatenate((src_label, label_))
        tgt_data, tgt_label = data[tgt-1], label[tgt-1]

        source_loader = DataLoader(
            dataset=EEGDataset(src_data.astype(np.float32), src_label),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        
        tgt_train_loader = DataLoader(
            dataset=EEGDataset(tgt_data.astype(np.float32), tgt_label),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )

        iter_target = iter(tgt_train_loader)
        n_batch = len(tgt_train_loader)
        tgt_train_list = []
        for _ in range(n_batch):
            data_target, _ = next(iter_target)
            tgt_train_list.append(data_target)

        tgt_test_loader = DataLoader(
            dataset=EEGDataset(tgt_data.astype(np.float32), tgt_label),
            batch_size=batch_size,
            shuffle=False
        )

        return source_loader, tgt_train_list, tgt_test_loader
    elif method == 'baseline':
        all_subject = list(range(1,fi.total_subjects_number[dataset]+1))
        if sub_n==-1:
            src_sub = list(set(all_subject)-set([tgt]))
        else:
            src_sub = list(set(all_subject)-set([tgt]))[:sub_n]
        isFirst = True
        for sub in src_sub:
            if isFirst:
                src_data, src_label = data[sub-1], label[sub-1]
                isFirst = False
            else:
                data_, label_ = data[sub-1], label[sub-1]
                src_data = np.concatenate((src_data, data_))
                src_label = np.concatenate((src_label, label_))
        tgt_data, tgt_label = data[tgt-1], label[tgt-1]

        source_loader = DataLoader(
            dataset=EEGDataset(src_data.astype(np.float32), src_label),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        
        tgt_data, tgt_label = data[tgt-1], label[tgt-1]

        tgt_loader = DataLoader(
            dataset=EEGDataset(tgt_data.astype(np.float32), tgt_label),
            batch_size=batch_size,
            shuffle=False
        )
        
        return source_loader, [], tgt_loader
    elif method=='DAMD':
        all_subject = list(range(1,fi.total_subjects_number[dataset]+1))
        if sub_n==-1:
            src_sub = list(set(all_subject)-set([tgt]))
        else:
            src_sub = list(set(all_subject)-set([tgt]))[:sub_n]
        src_list = []

        for i in src_sub:
            src_list.append(EEGDataset(data[i-1].astype(np.float32), label[i-1], i))

        src_loaders = [DataLoader(
            dataset=env,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True)
            for env in src_list]
        
        tgt_data, tgt_label = data[tgt-1], label[tgt-1]

        tgt_train_loader = DataLoader(
            dataset=EEGDataset(tgt_data.astype(np.float32), tgt_label),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )

        iter_target = iter(tgt_train_loader)
        n_batch = len(tgt_train_loader)
        tgt_train_list = []
        for _ in range(n_batch):
            data_target, _ = next(iter_target)
            tgt_train_list.append(data_target)


        tgt_loader = DataLoader(
            dataset=EEGDataset(tgt_data.astype(np.float32), tgt_label),
            batch_size=batch_size,
            shuffle=False
        )
        
        return src_loaders, tgt_train_list, tgt_loader


