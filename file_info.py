import os

import numpy as np
import yaml

total_subjects_number = {
    'SEED': 15,
    'SEED-IV': 15,
    'SEED-V': 16,
}

num_classes = {
    'SEED': 3,
    'SEED-IV': 4,
    'SEED-V': 5,
}

kernel_size = {
    'SEED': 3,
    'SEED-IV': 7,
    'SEED-V': 5,
}

# SEED-IV
SEED_IV = {
    1: ['1_20160518.mat', '1_20161125.mat', '1_20161126.mat'],
    2: ['2_20150915.mat', '2_20150920.mat', '2_20151012.mat'],
    3: ['3_20150919.mat', '3_20151018.mat', '3_20151101.mat'],
    4: ['4_20151111.mat', '4_20151118.mat', '4_20151123.mat'],
    5: ['5_20160406.mat', '5_20160413.mat', '5_20160420.mat'],
    6: ['6_20150507.mat', '6_20150511.mat', '6_20150512.mat'],
    7: ['7_20150715.mat', '7_20150717.mat', '7_20150721.mat'],
    8: ['8_20151103.mat', '8_20151110.mat', '8_20151117.mat'],
    9: ['9_20151028.mat', '9_20151119.mat', '9_20151209.mat'],
    10: ['10_20151014.mat', '10_20151021.mat', '10_20151023.mat'],
    11: ['11_20150916.mat', '11_20150921.mat', '11_20151011.mat'],
    12: ['12_20150725.mat', '12_20150804.mat', '12_20150807.mat'],
    13: ['13_20151115.mat', '13_20151125.mat', '13_20161130.mat'],
    14: ['14_20151205.mat', '14_20151208.mat', '14_20151215.mat'],
    15: ['15_20150508.mat', '15_20150514.mat', '15_20150527.mat'],
}

# SEED
SEED = {
    1: ['1_20131027.mat', '1_20131030.mat', '1_20131107.mat'],
    2: ['2_20140404.mat', '2_20140413.mat', '2_20140419.mat'],
    3: ['3_20140603.mat', '3_20140611.mat', '3_20140629.mat'],
    4: ['4_20140621.mat', '4_20140702.mat', '4_20140705.mat'],
    5: ['5_20140411.mat', '5_20140418.mat', '5_20140506.mat'],
    6: ['6_20130712.mat', '6_20131016.mat', '6_20131113.mat'],
    7: ['7_20131027.mat', '7_20131030.mat', '7_20131106.mat'],
    8: ['8_20140511.mat', '8_20140514.mat', '8_20140521.mat'],
    9: ['9_20140620.mat', '9_20140627.mat', '9_20140704.mat'],
    10: ['10_20131130.mat', '10_20131204.mat', '10_20131211.mat'],
    11: ['11_20140618.mat', '11_20140625.mat', '11_20140630.mat'],
    12: ['12_20131127.mat', '12_20131201.mat', '12_20131207.mat'],
    13: ['13_20140527.mat', '13_20140603.mat', '13_20140610.mat'],
    14: ['14_20140601.mat', '14_20140615.mat', '14_20140627.mat'],
    15: ['15_20130709.mat', '15_20131016.mat', '15_20131105.mat'],
}

dataset_path = {k: os.path.expanduser(v) for k, v in {
    'SEED': '~/Lab/dataset/SEED/SEED_EEG/ExtractedFeatures/',
    'SEED-IV': '~/Lab/dataset/SEED_IV/eeg_feature_smooth/',
    'SEED-V': '~/Lab/dataset/SEED-V/EEG_DE_features/',
}.items()}


DG_Method = {
    'DANN_DG', 'CORAL_DG', 'RSC', 'VREx'
}

DA_Method = {
    'DANN_DA', 'CORAL_DA', 'DAAN', 'DSAN'
}

DAMD_Method = {
    'CORAL_DAMD', 'DAAN_DAMD', 'DSAN_DAMD', 'DANN_DAMD',
    'L3B', 'L2B', 'test_L'
}

### DA config

def configDA(args):
    alg=args.alg
    path = os.path.join("DeepDAConfig", alg+'.yaml')
    f = open(path, 'r')
    cfg = f.read()
    d = yaml.load(cfg)
    for k, v in d.items():
        setattr(args, k, v)


channel_order = [ "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", 
                 "F5", "F3", "F1", "FZ", "F2", "F4", "F6", 
                 "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", 
                 "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", 
                 "C1", "CZ", "C2", "C4", "C6", "T8", "TP7", 
                 "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", 
                 "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", 
                 "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", 
                 "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2", ]

spatial = np.array([
    [0, 0, 0, "FP1", "FPZ", "FP2", 0, 0, 0],
    [0, 0, 0, "AF3", 0, "AF4", 0, 0, 0],
    ["F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8"],
    ["FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8"],
    ["T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8"], 
    ["TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8"], 
    ["P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8"],
    [0, "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", 0], 
    [0, 0, "CB1", "O1", "OZ", "O2", "CB2", 0, 0], 
])
