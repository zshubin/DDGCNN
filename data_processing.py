import scipy.io as sio
import os
import random
import numpy as np
import shutil

"process public dataset"
phase_list = ['train', 'test']
data_path = '/workspace/mount_sdd/zhangshubin/eeg_data'
processed_data_path = './processed_erp_data/'
os.makedirs(processed_data_path, exist_ok=True)
sess_list = os.listdir(data_path)
print(sess_list)
label_list = [1, 2,]# 3, 4]
for sess_index, sess in enumerate(sess_list):
    print(sess)
    os.makedirs(os.path.join(processed_data_path, sess),exist_ok=True)
    sub_list = os.listdir(os.path.join(data_path, sess))
    for index, sub in enumerate(sub_list):
        print(sub)
        os.mkdir(os.path.join(processed_data_path, sess, sub))
        for phase in phase_list:
            os.makedirs(os.path.join(processed_data_path, sess, sub, phase),exist_ok=True)

            ori_data = \
                sio.loadmat(os.path.join(data_path, sess, sub,
                                         "sess{}_subj{}_EEG_ERP.mat".format(str(sess_index + 1).zfill(2),
                                                                              sub[1:].zfill(2))),
                            variable_names=["EEG_ERP_{}".format(phase)])['EEG_ERP_{}'.format(phase)]
            data = ori_data['x'][0][0]
            label_array = ori_data['y_dec'][0][0][0]
            split_time = ori_data['t'][0][0][0]
            for label in label_list:
                os.mkdir(os.path.join(processed_data_path, sess, sub, phase, str(label)))
                ssvep_label_index = np.where(label_array == label)[0]
                for i in range(ssvep_label_index.shape[0]):
                    start_time_index = ssvep_label_index[i]
                    start_time = split_time[start_time_index]
                    if start_time == split_time[-1]:
                        stop_time = data.shape[0]
                    else:
                        stop_time = split_time[start_time_index + 1] + 1
                    trail_bci_data = data[start_time:stop_time, :].T.astype(np.float32)
                    np.save(os.path.join(processed_data_path, sess, sub, phase, str(label),
                                         'sess{}_{}_{}_{}_{}.npy'.format(str(sess_index + 1).zfill(2), phase, sub,
                                                                         str(label), str(i))),
                            trail_bci_data)
            if phase == 'train':
                os.makedirs(os.path.join(processed_data_path, sess, sub, 'valid'),exist_ok=True)
                full_train_list = os.listdir(os.path.join(processed_data_path, sess, sub, phase))
                valid_list = random.sample(full_train_list, int(0.1 * len(full_train_list)))
                for i in range(len(valid_list)):
                    shutil.move(os.path.join(processed_data_path, sess, sub, phase,valid_list[i]),os.path.join(processed_data_path, sess, sub, 'valid'))


# "process private dataset"
# phase_list = ['train', 'test']
# data_path = './private_data'
# processed_data_path = './processed_private_ssvep_data/'
# split_ratio = 0.8
# os.mkdir(processed_data_path)
# sub_list = os.listdir(os.path.join(data_path,'data'))
# sub_name_list = []
# for i in range(len(sub_list)):
#     sub = sub_list[i].split('.')[0]
#     sub_name_list.append(sub)
#
# # label_list = list(np.arange(20))
# for index, sub in enumerate(sub_name_list):
#     sub_name = sub.split('_')[0]
#     print(sub_name)
#     os.mkdir(os.path.join(processed_data_path, sub_name))
#
#     ori_data = sio.loadmat(os.path.join(data_path, 'data', sub + '.mat'))['data']
#     label_file = open(os.path.join(data_path, 'label', sub_name +'_sess0' + '_label'+'.txt'))
#     label_dict = {}
#     for line in label_file.readlines():
#         label_list = line.split('\t')
#         label_list = [i.strip() for i in label_list]
#         label, point = int(label_list[2]), int(label_list[0]) - 1
#         if int(label)-1 in label_dict.keys():
#             label_dict[int(label)-1].append(point)
#         else:
#             label_dict[int(label)-1] = [point]
#
#     os.mkdir(os.path.join(processed_data_path, sub_name, 'train'))
#     os.mkdir(os.path.join(processed_data_path, sub_name, 'valid'))
#     os.mkdir(os.path.join(processed_data_path, sub_name, 'test'))
#     for label in label_dict.keys():
#         time_point_list=label_dict[label]
#         data_list = []
#         for i in range(len(time_point_list)):
#             start_time = time_point_list[i]
#             data = ori_data[:62,start_time-1:start_time-1+4000].astype(np.float32)
#             data_list.append(data)
#         random.shuffle(data_list)
#         os.mkdir(os.path.join(processed_data_path, sub_name, 'train', str(label)))
#         for j in range(int(split_ratio*len(data_list))):
#             data = data_list[j]
#             np.save(os.path.join(processed_data_path, sub_name, 'train', str(label),'trail_{}.npy'.format(j)), data)
#         os.mkdir(os.path.join(processed_data_path, sub_name, 'test', str(label)))
#         for j in range(int(split_ratio*len(data_list)),int((split_ratio+0.1)*len(data_list))):
#             data = data_list[j]
#             np.save(os.path.join(processed_data_path, sub_name, 'test', str(label),'trail_{}.npy'.format(j)), data)
#         os.mkdir(os.path.join(processed_data_path, sub_name, 'valid', str(label)))
#         for j in range(int((split_ratio+0.1)*len(data_list)),int((split_ratio+0.2)*len(data_list))):
#             data = data_list[j]
#             np.save(os.path.join(processed_data_path, sub_name, 'valid', str(label),'trail_{}.npy'.format(j)), data)
# print('Process finished')
