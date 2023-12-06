import torch
from torch.utils.data import Dataset
import os
import numpy as np
import random
import scipy.signal as sp
from scipy.signal import butter, lfilter


def butter_bandpass_filter(data, lowcut=8, highcut=30, samplingRate=512, order=4):
    y = np.zeros_like(data).astype(np.float32)
    nyq = 0.5 * samplingRate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    for i in range(data.shape[0]):
        y[i, :] = lfilter(b, a, data[i, :])
    return y


def iir_bandpass_filter(data, lowcut=8, highcut=30, samplingRate=512, order=4):
    y = np.zeros_like(data).astype(np.float32)
    nyq = 0.5 * samplingRate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sp.iirfilter(order, [low, high], btype='band')
    for i in range(data.shape[0]):
        y[i, :] = sp.filtfilt(b, a, data[i, :])
    return y

def get_mean_data(data_path,down_sample,SAMPLING_FREQ,label,x1,x2):
    train_list = os.listdir(os.path.join(data_path, str(label)))
    bci_data = np.load(os.path.join(data_path, str(label), train_list[0]))[:, x1:x2]#[:, ::down_sample]
    bci_data = iir_bandpass_filter(bci_data, 7, 70, SAMPLING_FREQ, 4).astype(np.float32)
    bci_data = np.expand_dims(bci_data, axis=0)
    for i in range(len(train_list) - 1):
        mini_bci_data = np.load(os.path.join(data_path, str(label), train_list[i + 1]))[:, x1:x2]
        mini_bci_data = iir_bandpass_filter(mini_bci_data, 7, 70, SAMPLING_FREQ, 4).astype(np.float32)
        mini_bci_data = np.expand_dims(mini_bci_data, axis=0)
        bci_data = np.concatenate((bci_data, mini_bci_data), axis=0)
    data_mean = np.mean(bci_data, axis=0)
    data_mean = np.squeeze(data_mean)
    return data_mean


def get_template(data_path,down_sample,SAMPLING_FREQ,class_num,x1,x2):
    data_mean_list = []
    for i in range(0,int(class_num)):
        mean_data = get_mean_data(data_path,down_sample,SAMPLING_FREQ,i,x1,x2)
        data_mean_list.append(mean_data)
    return data_mean_list

class DatasetProcessing(Dataset):
    def __init__(self, data_path, phase, win_train=1, down_sample=1, sample_freq=1000, transform=None, device='cuda'):
        self.bci_data_path = os.path.join(data_path, phase) #data/train data/test
        self.transform = transform
        self.bci_file_name = []
        self.sample_freq = int(sample_freq/down_sample)
        self.win_train = int(self.sample_freq*win_train)
        self.phase = phase
        self.down_sample = down_sample
        self.label = []
        self.device = device
        self.class_num = 0.
        self.class_dict = {}
        for class_name in os.listdir(self.bci_data_path):
            class_bci_file = os.listdir(os.path.join(self.bci_data_path, class_name))
            self.bci_file_name.extend(class_bci_file)
            self.label.extend([self.class_num]*len(class_bci_file))
            self.class_dict[self.class_num] = class_name
            self.class_num += 1.
        self.label = np.array(self.label).astype(np.float32)

    def __getitem__(self, index):

        def simple_batch_norm_1d(x,dim):
            eps = 1e-5
            x_mean = torch.mean(x, dim=dim, keepdim=True)
            x_var = torch.mean((x - x_mean) ** 2, dim=dim, keepdim=True)
            x_hat = (x - x_mean) / torch.sqrt(x_var + eps)
            return x_hat
        label_name = self.class_dict[self.label[index]]
        time_start = random.randint(160, int(self.sample_freq*4 + 35 - self.win_train))
        x1 = time_start
        x2 = time_start + self.win_train
        # c = [24, 28, 29, 30, 41, 42, 43, 60, 61]
        temp_data = get_template(self.bci_data_path,self.down_sample,self.sample_freq,self.class_num, x1, x2)
        bci_data = np.load(os.path.join(self.bci_data_path, label_name, self.bci_file_name[index]))[:, x1:x2]
        bci_data = torch.from_numpy(iir_bandpass_filter(bci_data, 7, 70, self.sample_freq, 4).astype(np.float32)).to(self.device)

        for i in range(len(temp_data)):
            temp_data[i] = torch.from_numpy(temp_data[i]).to(self.device)

        if self.transform is None:
            bci_data = simple_batch_norm_1d(bci_data,dim=0).T
            for i in range(len(temp_data)):
                temp_data[i] = torch.unsqueeze(simple_batch_norm_1d(temp_data[i],dim=0),2)
        temp_data = torch.cat(temp_data, dim=2)
        bci_data = torch.unsqueeze(bci_data, 0)

        label = torch.from_numpy(np.array(self.label[index])).to(self.device)
        return bci_data, temp_data, label

    def __len__(self):
        return len(self.bci_file_name)


def data_generator_np(data_path, batch_size, win_train=1, down_sample=1, sample_freq=1000, device='cuda'):
    train_dataset = DatasetProcessing(data_path, 'train',win_train=win_train, down_sample=down_sample, sample_freq=sample_freq, device=device)
    test_dataset = DatasetProcessing(data_path, 'test',win_train=win_train, down_sample=down_sample, sample_freq=sample_freq, device=device)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0)

    return train_loader, test_loader



