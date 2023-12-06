import numpy as np
import os
from data_loader.DataProcessing import Learner, Filter
import warnings
import scipy.io as sio
warnings.filterwarnings('ignore')

data_path = './40targetdata/' #./40targetdata/ #./processed_ssvep_data/session2/'
down_sample = 4
SAMPLING_FREQ = 1000/down_sample
LOWER_CUTOFF = 7.  #62 3   64 7
UPPER_CUTTOF = 70. #62 50  64 70
filt_type='iir'
win_type = None
FILT_ORDER = 4
c = [47,53,54,55,56,57,60,61,62]# 64 [47,53,54,55,56,57,60,61,62] 62[24, 28, 29, 30, 41, 42, 43, 60, 61]
time_win = 1.0
sub_list = os.listdir(data_path)


def get_cca_reference_signals(data_len, target_freq, sampling_rate, harmo_num):
    reference_signals = []
    t = np.arange(0, (data_len / (sampling_rate)), step=1.0 / (sampling_rate))
    for i in range(harmo_num):
        reference_signals.append(np.sin(np.pi * 2 * (i+1) * target_freq * t))
        reference_signals.append(np.cos(np.pi * 2 * (i+1) * target_freq * t))
    reference_signals = np.array(reference_signals)[:, ::down_sample]
    return reference_signals

def get_data(full_path,type='train',label=4):
    label_list = list(range(label))
    dp = Filter(LOWER_CUTOFF, UPPER_CUTTOF, SAMPLING_FREQ, FILT_ORDER, filt_type=filt_type, win_type=win_type)
    ################################train_1##############################
    train_list = os.listdir(os.path.join(full_path, type, str(label_list[0])))
    bci_data = np.load(os.path.join(full_path, type, str(label_list[0]), train_list[0]))[c, #[:, ::down_sample]
               int(SAMPLING_FREQ * (0.64)): int(SAMPLING_FREQ * (time_win + 0.64))]
    bci_data = np.expand_dims(bci_data, axis=0)
    for i in range(len(train_list) - 1):
        if len(train_list)==1:
            break
        mini_bci_data = np.load(os.path.join(full_path, type, str(label_list[0]), train_list[i + 1]))[c,
                        : int(SAMPLING_FREQ * time_win)]
        mini_bci_data = dp.ApplyFilter(mini_bci_data)
        mini_bci_data = np.expand_dims(mini_bci_data, axis=0)
        bci_data = np.concatenate((bci_data, mini_bci_data), axis=0)
    train_label_cal = [0.] * (len(train_list))

    for i in range(1,len(label_list)):
        train_list = os.listdir(os.path.join(full_path, type, str(label_list[i])))
        for j in range(len(train_list)):
            mini_bci_data = np.load(os.path.join(full_path, type, str(label_list[i]), train_list[j]))[c,
                            int(SAMPLING_FREQ * (0.64)): int(SAMPLING_FREQ * (time_win + 0.64))]
            mini_bci_data = dp.ApplyFilter(mini_bci_data)
            mini_bci_data = np.expand_dims(mini_bci_data, axis=0)
            bci_data = np.concatenate((bci_data, mini_bci_data), axis=0)
        labels_cal = [float(label_list[i])] * len(train_list)
        train_label_cal.extend(labels_cal)
    train_data_cal = bci_data
    return train_data_cal, train_label_cal

freq_list = sio.loadmat('./Freq_Phase.mat')['freqs']
# freq_list = [12, 8.57, 6.67, 5.45]
freq_bank = [get_cca_reference_signals(int(time_win*1000), i, 1000, 5) for i in freq_list[0,:]]
cca_p_list, fbcca_p_list, ecca_p_list, trca_p_list, l1mcca_p_list, mset_p_list, itcca_p_list = [],[],[],[],[],[],[]
cca_r_list, fbcca_r_list, ecca_r_list, trca_r_list, l1mcca_r_list, mset_r_list, itcca_r_list = [],[],[],[],[],[],[]
cca_f_list, fbcca_f_list, ecca_f_list, trca_f_list, l1mcca_f_list, mset_f_list, itcca_f_list = [],[],[],[],[],[],[]
cca_a_list, fbcca_a_list, ecca_a_list, trca_a_list, l1mcca_a_list, mset_a_list, itcca_a_list = [],[],[],[],[],[],[]
for j in range(len(sub_list)):
    sub = sub_list[j]
    print(sub)
    full_path = os.path.join(data_path,sub)
    train_data_cal, train_label_cal = get_data(full_path,type='train',label=freq_list.shape[1])
    #######################################################valid_1###################
    val_data_cal, val_label_cal = get_data(full_path,type='test',label=freq_list.shape[1])


    learner = Learner(labels=freq_list.shape[1])
    print('cca processing')# [[7, 16], [16,30], [24,45],[32,70]] #[[3, 14], [9, 26],[14, 38], [19, 50]]
    cca_accuracy, cca_p, cca_r = learner.cca_cross_validation(train_data=train_data_cal, train_label=train_label_cal, test_data=val_data_cal,
                                                test_label=val_label_cal, CCA='cca', freq=freq_bank, band_list=[[7, 16], [16,30], [24,45],[32,70]], n_iter=5)
    print('fbcca processing')
    fbcca_accuracy, fbcca_p, fbcca_r = learner.cca_cross_validation(train_data=train_data_cal, train_label=train_label_cal, test_data=val_data_cal,
                                                test_label=val_label_cal, CCA='fbcca', freq=freq_bank, band_list=[[7, 16], [16,30], [24,45],[32,70]], n_iter=5)
    print('ecca processing')
    ecca_accuracy, ecca_p, ecca_r = learner.cca_cross_validation(train_data=train_data_cal, train_label=train_label_cal, test_data=val_data_cal,
                                                test_label=val_label_cal, CCA='ecca', freq=freq_bank, band_list=[[7, 16], [16,30], [24,45],[32,70]], n_iter=5)
    print('itcca processing')
    itcca_accuracy, itcca_p, itcca_r = learner.cca_cross_validation(train_data=train_data_cal, train_label=train_label_cal, test_data=val_data_cal,
                                                test_label=val_label_cal, CCA='itcca', freq=freq_bank, band_list=[[7, 16], [16,30], [24,45],[32,70]], n_iter=5)
    print('trca processing')
    trca_accuracy, trca_p, trca_r = learner.cca_cross_validation(train_data=train_data_cal, train_label=train_label_cal, test_data=val_data_cal,
                                                test_label=val_label_cal, CCA='trca', freq=freq_bank, band_list=[[7, 16], [16,30], [24,45],[32,70]], n_iter=5)
    print('lacca processing')
    l1cca_accuracy, l1cca_p, l1cca_r = learner.cca_cross_validation(train_data=train_data_cal, train_label=train_label_cal, test_data=val_data_cal,
                                                test_label=val_label_cal, CCA='l1mcca', freq=freq_bank, band_list=[[7, 16], [16,30], [24,45],[32,70]], n_iter=5)
    print('msetcca processing')
    msetcca_accuracy, msetcca_p, msetcca_r = learner.cca_cross_validation(train_data=train_data_cal, train_label=train_label_cal, test_data=val_data_cal,
                                                test_label=val_label_cal, CCA='msetcca', freq=freq_bank, band_list=[[7, 16], [16,30], [24,45],[32,70]], n_iter=5)#

    cca_a_list.append(cca_accuracy)
    fbcca_a_list.append(fbcca_accuracy)
    ecca_a_list.append(ecca_accuracy)
    trca_a_list.append(trca_accuracy)
    l1mcca_a_list.append(l1cca_accuracy)
    mset_a_list.append(msetcca_accuracy)
    itcca_a_list.append(itcca_accuracy)
    cca_p_list.append(cca_p)
    fbcca_p_list.append(fbcca_p)
    ecca_p_list.append(ecca_p)
    trca_p_list.append(trca_p)
    l1mcca_p_list.append(l1cca_p)
    mset_p_list.append(msetcca_p)
    itcca_p_list.append(itcca_p)
    cca_r_list.append(cca_r)
    fbcca_r_list.append(fbcca_r)
    ecca_r_list.append(ecca_r)
    trca_r_list.append(trca_r)
    l1mcca_r_list.append(l1cca_r)
    mset_r_list.append(msetcca_r)
    itcca_r_list.append(itcca_r)
    cca_f_list.append(2.*cca_r*cca_p/(cca_r+cca_p))
    fbcca_f_list.append(2.*fbcca_r*fbcca_p/(fbcca_r+fbcca_p))
    ecca_f_list.append(2.*ecca_r*ecca_p/(ecca_r+ecca_p))
    trca_f_list.append(2.*trca_r*trca_p/(trca_r+trca_p))
    l1mcca_f_list.append(2.*l1cca_r*l1cca_p/(l1cca_r+l1cca_p))
    mset_f_list.append(2.*msetcca_r*msetcca_p/(msetcca_r+msetcca_p))
    itcca_f_list.append(2.*itcca_r*itcca_p/(itcca_r+itcca_p))


print('mean p:', np.mean(cca_p_list), np.mean(fbcca_p_list),np.mean(ecca_p_list),np.mean(itcca_p_list),np.mean(trca_p_list),np.mean(l1mcca_p_list),np.mean(mset_p_list))
print('mean r:', np.mean(cca_r_list), np.mean(fbcca_r_list),np.mean(ecca_r_list),np.mean(itcca_r_list),np.mean(trca_r_list),np.mean(l1mcca_r_list),np.mean(mset_r_list))
print('mean f:', np.mean(cca_f_list), np.mean(fbcca_f_list),np.mean(ecca_f_list),np.mean(itcca_f_list),np.mean(trca_f_list),np.mean(l1mcca_f_list),np.mean(mset_f_list))
print('mean acc:', np.mean(cca_a_list), np.mean(fbcca_a_list),np.mean(ecca_a_list),np.mean(itcca_a_list),np.mean(trca_a_list),np.mean(l1mcca_a_list),np.mean(mset_a_list))
