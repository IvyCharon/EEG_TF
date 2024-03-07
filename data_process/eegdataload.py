class EEGDataset(object):
    def __init__(self, data, label, subject_no=-1):
        self.data=data
        self.label=label
        self.subject_no=subject_no

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.data.shape[0]
