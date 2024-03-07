import numpy as np
import pickle
from sklearn import preprocessing, svm
import argparse
import scipy.io as scio
from dataset import file_info as fi


def main(args):
    if args.subject == 0:
        acc_=[]
        all_c=[]
        for i in range(1,fi.total_subjects_number[args.dataset]+1):
            a,c=train_test(i, args.normalize, args.dataset)
            acc_.append(a)
            all_c.append(c)
        acc=np.mean(acc_)
    else:
        acc, all_c=train_test(args.subject, args.normalize, args.dataset)
    
    print(acc)


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

def data_process(subject_no, normalize_type, p, dataset):    # p = 0, 1, 2
    if dataset=='SEED-V':
        return data_process_SEED_V(subject_no, normalize_type, p)
    elif dataset=='SEED-IV':
        return data_process_SEED_IV(subject_no, normalize_type, p)
    elif dataset=='SEED':
        return data_process_SEED(subject_no, normalize_type, p)
    else:
        raise RuntimeError('Wrong dataset')

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

def data_process_SEED_V(subject_no, normalize_type, p):    # p = 0, 1, 2
    # load data
    # for one subject
    # subject_no = args.subject
    path=fi.dataset_path['SEED-V']
    data_npz = np.load(path + "{}_123.npz".format(subject_no))
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

def train_test(subject_no, normalize_type, dataset):
    accuracy = []
    all_c = []
    for i in range(3):
        # data process
        train_data, train_label, test_data, test_label = data_process(subject_no, normalize_type, i, dataset)

        # classification
        # tol
        # set 3 svm, and train-test each
        best_res = {}
        best_res['c'] = 0
        best_res['acc'] = 0
        for c in range(-10,10):
            clf = svm.LinearSVC(C=2**c)
            clf.fit(train_data, train_label)
            acc = clf.score(test_data, test_label)
            if acc>best_res['acc']:
                best_res['acc']=acc
                best_res['c']=2**c
        accuracy.append(best_res['acc'])
        all_c.append(best_res['c'])
    
    final_acc=np.mean(accuracy)
    return final_acc, all_c
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--normalize", type=str, default='standard') # choose from standard, minmax, no
    parser.add_argument('--dataset', type=str, default='SEED-V') # choose from SEED, SEED-IV, SEED-V
    args = parser.parse_args()
    
    main(args)
