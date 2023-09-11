import os, sys, inspect

cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split \
    (inspect.getfile( inspect.currentframe() ))[0],'algorithms')))

if cmd_subfolder not in sys.path:
            sys.path.insert(0, cmd_subfolder)

import numpy as np # numpy - used for array and matrices operations
import math as math # used for basic mathematical operations
from scipy.signal import butter, lfilter
import scipy.signal as sp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import ShuffleSplit, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from mne.decoding import CSP
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import scipy
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import Lasso
from scipy.sparse.linalg import eigs
from sklearn.metrics import precision_recall_fscore_support


def plot_spectogram(data_in, fs):
    f, t, Sxx = sp.spectrogram(data_in, fs, nperseg=50, noverlap=49)
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def KL_divergence(p,q):
    return scipy.stats.entropy(p, q)


def plot_embedding(datao, datac, label, datao_kl, datac_kl):
    # x_min, x_max = np.min(datao, 0), np.max(datao, 0)
    # data1 = (data1 - x_min) / (x_max - x_min)
    # x_min, x_max = np.min(data2, 0), np.max(data2, 0)
    # data2 = (data2 - x_min) / (x_max - x_min)

    ax = plt.subplot(1, 2, 1, projection='3d')
    ax.set_title('origin data (KL divergence:{:.3f})'.format(datao_kl))
    for i in range(datao.shape[0]):
        if label[i] == 1.0:
            color = 'r'
        else:
            color = 'y'
        ax.scatter(datao[i, 0], datao[i, 1], datao[i, 2], c=color)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax = plt.subplot(1, 2, 2, projection='3d')
    ax.set_title('CSP feature (KL divergence:{:.3f})'.format(datac_kl))
    for i in range(datac.shape[0]):
        if label[i] == 1.0:
            color = 'r'
        else:
            color = 'y'
        ax.scatter(datac[i, 0], datac[i, 1], datac[i, 2], c=color)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def my_cca(x,y):
    z = np.concatenate((x,y),axis=0)
    C = np.cov(z)
    sx = x.shape[0]
    sy = y.shape[0]
    Cxx = C[0:sx, 0:sx] + 1e-8*np.eye(sx)
    Cxy = C[0:sx, sx:sx+sy]
    Cyx = Cxy.T
    Cyy = C[sx:sx+sy, sx:sx+sy] + 1e-8*np.eye(sy)
    invCyy = np.linalg.pinv(Cyy)
    # r, Wx = scl.eig(np.dot(np.dot(np.dot(np.linalg.inv(Cxx),Cxy),invCyy),Cyx))
    r, Wx = np.linalg.eig(np.matmul(np.matmul(np.matmul(np.linalg.pinv(Cxx),Cxy),invCyy),Cyx))
    ind = np.argsort(-r)
    r = r[ind]
    Wx = Wx[:, ind]
    r = np.sqrt(r.real)
    r = np.nan_to_num(r)
    V = np.fliplr(Wx)
    r = np.flipud(r)
    I = np.argsort(r.real,axis=0)
    r = r.real[I]
    r = np.flipud(r)
    Wx = np.zeros((V.shape[0],I.shape[0]),dtype=np.float32)
    for i in range(I.shape[0]):
        Wx[:,i] = V[:,I[i]]
    Wx = np.fliplr(Wx)
    Wy = np.matmul(np.matmul(invCyy,Cyx),Wx)
    return Wx, Wy, r


def my_cca1(x,y):
    meanx = np.expand_dims(np.mean(x,1), 1)
    meany = np.expand_dims(np.mean(y,1), 1)
    s11,s22,s12,s21 = 0,0,0,0
    for i in range(y.shape[1]):
        s11 += np.dot(np.expand_dims(x[:, i], 1) - meanx, (np.expand_dims(x[:, i], 1) - meanx).T)
        s22 += np.dot(np.expand_dims(y[:, i], 1) - meany, (np.expand_dims(y[:, i], 1) - meany).T)
        s12 += np.dot(np.expand_dims(x[:, i], 1) - meanx, (np.expand_dims(y[:, i], 1) - meany).T)
        s21 += np.dot(np.expand_dims(y[:, i], 1) - meany, (np.expand_dims(x[:, i], 1) - meanx).T)
    s11 /= (y.shape[1] - 1) + 1e-8*np.eye(x.shape[0])
    s12 /= (y.shape[1] - 1)
    s22 /= (y.shape[1] - 1) + 1e-8*np.eye(y.shape[0])
    s21 /= (y.shape[1] - 1)
    eigvaluea, eigvectora = np.linalg.eig(np.matmul(np.matmul(np.matmul(np.linalg.pinv(s11),s12),
                                                              np.linalg.pinv(s22)),
                                                    s21))
    eigvalueb, eigvectorb = np.linalg.eig(np.matmul(np.matmul(np.matmul(np.linalg.pinv(s22),s21),
                                                              np.linalg.pinv(s11)),
                                                    s12))
    indexa = np.argmax(np.nan_to_num(np.sqrt(eigvaluea).real))
    wx = eigvectora[:,indexa]
    indexb = np.argmax(np.nan_to_num(np.sqrt(eigvalueb).real))
    wy = eigvectorb[:, indexb]
    return np.expand_dims(wx,1), np.expand_dims(wy,1)


class Learner:
    def __init__(self, model=None):#frea_har = 7
        self.clf = model
        self.tsne = TSNE(n_components=3, init='pca', random_state=0)

    def DesignLDA(self):
        self.svc = LDA()

    def DesignCSP(self, n_comp):
        self.csp = CSP(n_components=n_comp, reg=None, log=None, cov_est='epoch', transform_into='average_power')#log=True

    def CCA_fit_transform(self,n_components, epoch_data_full, epoch_label_full, freq_list):
        self.cca = CCA(n_components)
        epoch_label_full = np.squeeze(np.array(epoch_label_full))
        predicted_result = np.zeros(epoch_data_full.shape[0])
        for j in range(epoch_data_full.shape[0]):
            corr = np.zeros(n_components)
            result = np.zeros(len(freq_list))
            epoch_data = epoch_data_full[j, :, :]
            for freq_idx in range(len(freq_list)):
                self.cca.fit(epoch_data.T, np.squeeze(freq_list[freq_idx]).T)
                O1_a, O1_b = self.cca.transform(epoch_data.T, np.squeeze(freq_list[freq_idx]).T)
                for ind_val in range(0, n_components):
                    corr[ind_val] = np.abs(np.corrcoef(O1_a[:, ind_val], O1_b[:, ind_val])[0, 1])
                    result[freq_idx] = np.max(corr)
            predicted_result[j] = np.argmax(result)
        p_class, r_class, f_class, support_micro = precision_recall_fscore_support(y_true=epoch_label_full, y_pred=predicted_result, labels=[0,1,2,3], average=None)
        acc = np.sum(predicted_result == epoch_label_full)/epoch_data_full.shape[0]
        return acc, p_class.mean(), r_class.mean()

    def FBCCA_fit_transform(self, n_components, epoch_data_full, epoch_label_full, band_list, freq_list, sampling_freq, filt_order=4, filt_type='iir', win_type=None, a=1.25, b=0.25):
        self.cca = CCA(n_components)
        epoch_label_full = np.squeeze(np.array(epoch_label_full))
        predicted_result = np.zeros(epoch_data_full.shape[0])
        for j in range(epoch_data_full.shape[0]):
            result = np.zeros(len(freq_list))
            epoch_data = epoch_data_full[j, :, :]
            for freq_idx in range(len(freq_list)):
                corr = 0
                for i in range(len(band_list)):
                    dp = Filter(band_list[i][0], band_list[i][1], sampling_freq, filt_order, filt_type=filt_type,
                                win_type=win_type)
                    filter_data = dp.ApplyFilter(epoch_data)
                    self.cca.fit(filter_data.T, np.squeeze(freq_list[freq_idx]).T)
                    O1_a, O1_b = self.cca.transform(epoch_data.T, np.squeeze(freq_list[freq_idx]).T)
                    for ind_val in range(0, n_components):
                        corr += ((i+1)**(-a)+b)*np.corrcoef(O1_a[:, ind_val], O1_b[:, ind_val])[0, 1]**2
                result[freq_idx] = corr
            predicted_result[j] = np.argmax(result)
        p_class, r_class, f_class, support_micro = precision_recall_fscore_support(y_true=epoch_label_full, y_pred=predicted_result, labels=[0,1,2,3], average=None)
        acc = np.sum(predicted_result == epoch_label_full)/epoch_data_full.shape[0]
        return acc, p_class.mean(), r_class.mean()

    def ECCA_fit_transform(self, epoch_data_train, epoch_label_train, freq_list, epoch_data_test, epoch_label_test):
        epoch_label_test = np.squeeze(np.array(epoch_label_test))
        epoch_data_train, epoch_data_test = epoch_data_train.astype(np.float32), epoch_data_test.astype(np.float32)
        label_list = sorted(list(set(epoch_label_train)))
        data_mean_list = []
        for i in range(len(label_list)):
            label = float(label_list[i])
            class_index = [index for (index, value) in enumerate(epoch_label_train) if value == label]
            data_class = epoch_data_train[class_index]
            data_mean = np.mean(data_class, axis=0)
            data_mean_list.append(data_mean)

        predicted_result = np.zeros(epoch_data_test.shape[0])
        for j in range(epoch_data_test.shape[0]):
            result = np.zeros(len(label_list))
            epoch_data = epoch_data_test[j, :, :]
            for freq_idx in range(len(label_list)):
                coeff = np.zeros(4)
                ref_data = freq_list[freq_idx]
                #rho 1
                wn1, wn2 = my_cca1(epoch_data, ref_data)
                weighted_train = np.dot(wn2.T, ref_data)
                weighted_test = np.dot(wn1.T, epoch_data)
                coeff[0] = np.abs(np.corrcoef(weighted_test, weighted_train)[0, 1])
                #rho 2
                wn, _ = my_cca1(epoch_data, data_mean_list[freq_idx])
                weighted_train = np.dot(wn.T, data_mean_list[freq_idx])
                weighted_test = np.dot(wn.T, epoch_data)
                coeff[1] = np.corrcoef(weighted_test, weighted_train)[0, 1]
                #rho 3
                wn, _ = my_cca1(epoch_data, ref_data)
                weighted_train = np.dot(wn.T, data_mean_list[freq_idx])
                weighted_test = np.dot(wn.T, epoch_data)
                coeff[2] = np.corrcoef(weighted_test, weighted_train)[0, 1]
                #rho 4
                wn, _ = my_cca1(data_mean_list[freq_idx], ref_data)
                weighted_train = np.dot(wn.T, data_mean_list[freq_idx])
                weighted_test = np.dot(wn.T, epoch_data)
                coeff[3] = np.corrcoef(weighted_test, weighted_train)[0, 1]
                result[freq_idx] = np.sum(np.sign(coeff)*np.square(coeff))
            predicted_result[j] = np.argmax(result)
        p_class, r_class, f_class, support_micro = precision_recall_fscore_support(y_true=epoch_label_test, y_pred=predicted_result, labels=[0,1,2,3], average=None)
        acc = np.sum(predicted_result == epoch_label_test)/epoch_data_test.shape[0]
        return acc, p_class.mean(), r_class.mean()

    def MCCA_fit_transform(self,n_components, epoch_data_train, epoch_data_test, epoch_label_test, freq_list, max_iter):
        epoch_label_test = np.squeeze(np.array(epoch_label_test))
        result = np.zeros(epoch_data_test.shape[0],dtype=np.float32)
        z_list = []
        epoch_data_train = np.transpose(epoch_data_train,(1,2,0)) # channel * time * trail
        for freq_idx in range(len(freq_list)): #freq 2H * time
            w3 = np.random.randn((epoch_data_train.shape[2],1))
            for i in range(max_iter):
                epoch_data_train_ = np.dot(epoch_data_train, w3)  # channel * time
                v1, w1, r1 = my_cca(np.squeeze(freq_list[freq_idx]), epoch_data_train_)
                v1, w1 = v1[:, 0:n_components], w1[:, 0:n_components]
                w1 /= np.linalg.norm(w1, axis=0, keepdims=True)  # channel * 1
                v1 /= np.linalg.norm(v1, axis=0, keepdims=True)  # 2H * 1
                epoch_data_train_ = np.dot(w1, epoch_data_train)  # time * trail
                freq_ = np.dot(v1, np.squeeze(freq_list[freq_idx]))  # 1 * trail
                v2, w3, r2 = my_cca(freq_, epoch_data_train_)
                w3 = w3[:, 0:n_components]
                w3 /= np.linalg.norm(w3, axis=0, keepdims=True)
            z = np.dot(np.dot(w1.T, epoch_data_train), w3)
            z_list.append(z)
        for test_idx in range(epoch_data_test.shape[0]):
            epoch_data = epoch_data_test[test_idx,:,:]
            epoch_predict = np.zeros(len(z_list),dtype=np.float32)
            for ref_idx in range(len(z_list)):
                ref_data = z_list(ref_idx)
                wx, wy, r = my_cca(epoch_data, ref_data)
                epoch_predict[ref_idx] = np.max(r)
            result[test_idx] = np.argmax(epoch_predict)
        p_class, r_class, f_class, support_micro = precision_recall_fscore_support(y_true=epoch_label_test, y_pred=result, labels=[0,1,2,3], average=None)
        acc = np.sum(result == epoch_label_test)/epoch_data_test.shape[0]
        return acc, p_class.mean(), r_class.mean()

    def TRCA_fit_transform(self, epoch_data_train, epoch_label_train, epoch_data_test, epoch_label_test):
        epoch_data_train, epoch_data_test = epoch_data_train.astype(np.float32), epoch_data_test.astype(np.float32)
        label_list = sorted(list(set(epoch_label_train)))
        data_mean_list = []
        train_data = []
        for i in range(len(label_list)):
            label = float(label_list[i])
            class_index = [index for (index, value) in enumerate(epoch_label_train) if value == label]
            data_class = epoch_data_train[class_index]
            data_mean = np.mean(data_class, axis=0)
            data_class = np.transpose(data_class, (1, 2, 0))
            train_data.append(data_class)
            data_mean_list.append(data_mean)

        epoch_label_test = np.squeeze(np.array(epoch_label_test))
        predicted_result = np.zeros(epoch_data_test.shape[0], dtype=np.float32)
        for j in range(epoch_data_test.shape[0]):
            corr = np.zeros(len(label_list))
            epoch_data = np.squeeze(epoch_data_test[j, :, :])
            for freq_idx in range(len(data_mean_list)):
                data_class = np.squeeze(train_data[freq_idx])
                S = np.zeros((data_class.shape[0], data_class.shape[0]),dtype=np.float32)
                for trail_i in range(data_class.shape[2]):
                    x1 = np.squeeze(data_class[:, :, trail_i])
                    x1 -= np.repeat(np.expand_dims(np.mean(x1, 1), 1), x1.shape[1], 1)
                    for trail_j in range(trail_i+1, data_class.shape[2]):
                        x2 = np.squeeze(data_class[:, :, trail_j])
                        x2 -= np.repeat(np.expand_dims(np.mean(x2, 1), 1), x2.shape[1],1)
                        S += (np.matmul(x1,x2.T)+np.matmul(x2, x1.T))
                UX = np.squeeze(data_class[:, :, 0])
                for idd in range(1,data_class.shape[2]):
                    UX = np.concatenate((UX,np.squeeze(data_class[:,:,idd])),1)
                UX -= np.repeat(np.expand_dims(np.mean(UX, 1), 1), UX.shape[1],1)
                Q = np.matmul(UX,UX.T)
                c, V = eigs(S, k=6, M=Q)
                V = np.expand_dims(V[:, 0], 1)
                weighted_train = np.squeeze(np.matmul(V.T, data_mean_list[freq_idx]))
                weighted_test = np.squeeze(np.matmul(V.T, epoch_data))
                corr[freq_idx] = np.corrcoef(weighted_train, weighted_test)[0, 1]
            predicted_result[j] = np.argmax(corr)
        p_class, r_class, f_class, support_micro = precision_recall_fscore_support(y_true=epoch_label_test, y_pred=predicted_result, labels=[0,1,2,3], average=None)
        acc = np.sum(predicted_result == epoch_label_test)/epoch_data_test.shape[0]
        return acc, p_class.mean(), r_class.mean()

    def IT_CCA_fit_transform(self, n_components, epoch_data_train, epoch_label_train, epoch_data_test, epoch_label_test):
        self.cca = CCA(n_components)
        epoch_data_train, epoch_data_test = epoch_data_train.astype(np.float32), epoch_data_test.astype(np.float32)
        label_list = sorted(list(set(epoch_label_train)))
        data_mean_list = []
        for i in range(len(label_list)):
            label = float(label_list[i])
            class_index = [index for (index, value) in enumerate(epoch_label_train) if value == label]
            data_class = epoch_data_train[class_index]
            data_mean = np.mean(data_class, axis=0)
            data_mean_list.append(data_mean)

        epoch_label_test = np.squeeze(np.array(epoch_label_test))
        predicted_result = np.zeros(epoch_data_test.shape[0],dtype=np.float32)
        for j in range(epoch_data_test.shape[0]):
            corr = np.zeros(n_components)
            result = np.zeros(len(data_mean_list))
            epoch_data = epoch_data_test[j, :, :]
            for freq_idx in range(len(data_mean_list)):
                self.cca.fit(epoch_data.T, data_mean_list[freq_idx].T)
                O1_a, O1_b = self.cca.transform(epoch_data.T, data_mean_list[freq_idx].T)
                for ind_val in range(0, n_components):
                    corr[ind_val] = np.corrcoef(O1_a[:, ind_val], O1_b[:, ind_val])[0, 1]
                    result[freq_idx] = np.max(corr)
            predicted_result[j] = np.argmax(result)
        p_class, r_class, f_class, support_micro = precision_recall_fscore_support(y_true=epoch_label_test, y_pred=predicted_result, labels=[0,1,2,3], average=None)
        acc = np.sum(predicted_result == epoch_label_test)/epoch_data_test.shape[0]
        return acc, p_class.mean(), r_class.mean()

    def Mset_CCA_fit_transform(self, epoch_data_train, epoch_label_train, epoch_data_test, epoch_label_test, K):
        epoch_data_train, epoch_data_test = epoch_data_train.astype(np.float32), epoch_data_test.astype(np.float32)
        label_list = sorted(list(set(epoch_label_train)))
        Temp_list = []  # num_class * trail * channel * time
        for i in range(len(label_list)):
            label = float(label_list[i])
            class_index = [index for (index, value) in enumerate(epoch_label_train) if value == label]
            data_class = epoch_data_train[class_index]
            V = np.zeros((data_class.shape[0], data_class.shape[1],data_class.shape[1]), dtype=np.float32)
            X = np.zeros((data_class.shape[0], data_class.shape[1],data_class.shape[2]), dtype=np.float32)
            W = np.zeros((data_class.shape[0], data_class.shape[1], K), dtype=np.float32)
            Temp = np.zeros((data_class.shape[0] * K, data_class.shape[2]), dtype=np.float32)
            for j in range(data_class.shape[0]):
                Xwhit = data_class[j, :, :]  # channel * time
                npot = Xwhit.shape[1]
                Xwhit = Xwhit - np.repeat(np.expand_dims(np.mean(Xwhit,1),1), npot, 1)
                C = np.dot(Xwhit, Xwhit.T)/npot
                val, vec = np.linalg.eig(C+ 1e-8*np.eye(C.shape[0]))
                ind = np.argsort(val)
                val = val[ind]
                vec = vec[:, ind]
                val = np.diag(val)
                V[j, :, :] = np.matmul(np.linalg.pinv(vec.T), np.sqrt(val))
                X[j, :, :] = np.matmul(V[j, :, :], Xwhit)
            Y = X[0, :, :]
            for j in range(1,data_class.shape[0]):
                Y = np.concatenate((Y, X[j,:,:]), axis=0)
            R = np.cov(Y)
            S = np.diag(np.diag(R))
            rho, tempW = eigs(R-S, K, S)
            for j in range(data_class.shape[0]):
                W[j, :, :] = tempW[j*data_class.shape[1]:(j+1)*data_class.shape[1], :]/np.linalg.norm(tempW[j*data_class.shape[1]:(j+1)*data_class.shape[1], :])
            for j in range(data_class.shape[0]):
                W[j, :, :] = np.matmul(W[j, :, :].T, V[j, :, :]).T
            for j in range(data_class.shape[0]):
                Temp[j*K:(j+1)*K,:] = np.matmul(W[j, :, :].T,data_class[j, :, :])
            Temp_list.append(Temp)

        epoch_label_test = np.squeeze(np.array(epoch_label_test))
        result = np.zeros(epoch_data_test.shape[0], dtype=np.float32)
        for test_idx in range(epoch_data_test.shape[0]):
            epoch_data = epoch_data_test[test_idx, :, :]
            epoch_predict = np.zeros(len(Temp_list), dtype=np.float32)
            for ref_idx in range(len(Temp_list)):
                ref_data = Temp_list[ref_idx]
                wx, wy, r = my_cca(epoch_data, ref_data)
                epoch_predict[ref_idx] = np.max(r)
            result[test_idx] = np.argmax(epoch_predict)
        p_class, r_class, f_class, support_micro = precision_recall_fscore_support(y_true=epoch_label_test, y_pred=result, labels=[0,1,2,3], average=None)
        acc = np.sum(result == epoch_label_test)/epoch_data_test.shape[0]
        return acc, p_class.mean(), r_class.mean()

    def L1_MCCA_fit_transform(self, epoch_data_train, epoch_data_test, epoch_label_test, freq_list, max_iter, max_iter_, lambda1, lambda2, lambda3):
        epoch_label_test = np.squeeze(np.array(epoch_label_test))
        result = np.zeros(epoch_data_test.shape[0],dtype=np.float32)
        z_list = []
        epoch_data_train = np.transpose(epoch_data_train,(1,2,0)) # channel * time * trail
        for freq_idx in range(len(freq_list)): #freq 2H * time
            w3 = np.random.randn(epoch_data_train.shape[2],1)
            w3 /= np.linalg.norm(w3, axis=0, keepdims=True)
            for i in range(max_iter):
                projx3 = np.matmul(epoch_data_train, w3)
                v, w1, r = my_cca(freq_list[freq_idx],np.squeeze(projx3))
                v = np.expand_dims(v[:,0],1)
                w1 = np.expand_dims(w1[:,0],1)
                v /= np.linalg.norm(v, axis=0, keepdims=True)
                w1 /= np.linalg.norm(w1, axis=0, keepdims=True)
                projx1 = np.einsum('ij, jkl->ikl', w1.T, epoch_data_train)
                projref1 = np.matmul(v.T, np.squeeze(freq_list[freq_idx]))
                lasso_func = Lasso(alpha=lambda3)
                lasso_func.fit(np.squeeze(projx1), projref1.T)
                w3 = np.expand_dims(lasso_func.coef_,1)
                w3 /= np.linalg.norm(w3, axis=0, keepdims=True)
            tt = np.einsum('ij, jkl->ikl', w1.T, epoch_data_train)
            z = np.matmul(np.squeeze(tt), w3).T
            z_list.append(z)
        for test_idx in range(epoch_data_test.shape[0]):
            epoch_data = epoch_data_test[test_idx,:,:]
            epoch_predict = np.zeros(len(z_list),dtype=np.float32)
            for ref_idx in range(len(z_list)):
                ref_data = z_list[ref_idx]
                wx, wy, r = my_cca(epoch_data, ref_data)
                epoch_predict[ref_idx] = np.max(r)
            result[test_idx] = np.argmax(epoch_predict)
        p_class, r_class, f_class, support_micro = precision_recall_fscore_support(y_true=epoch_label_test, y_pred=result, labels=[0,1,2,3], average=None)
        acc = np.sum(result == epoch_label_test)/epoch_data_test.shape[0]
        return acc, p_class.mean(), r_class.mean()

    def CCA_M3_fit_transform(self, n_components, epoch_data_train, epoch_label_train, epoch_data_test, epoch_label_test, num_classes):
        self.cca = CCA(n_components)
        epoch_data_train, epoch_data_test = epoch_data_train.astype(np.float32), epoch_data_test.astype(np.float32)
        label_list = [i for i in range(num_classes)]
        data_mean_list = []
        O1_a_list = []
        for i in range(len(label_list)):
            label = float(label_list[i])
            class_index = [index for (index, value) in enumerate(epoch_label_train) if value == label]
            data_class = epoch_data_train[class_index]
            data_mean = np.mean(data_class, axis=0)
            data_X = np.repeat(data_mean, data_class.shape[0], axis=1)
            data_class = np.concatenate([data_class[i, :, :] for i in range(data_class.shape[0])], axis=1)
            self.cca.fit(data_class.T.astype(np.float32), data_X.T.astype(np.float32))
            O1_a, O1_b = self.cca.transform(data_class.T.astype(np.float32), data_X.T.astype(np.float32))
            O1_a = O1_a.astype(np.float32)
            data_mean_list.append(data_mean)
            O1_a_list.append(O1_a)

        epoch_label_test = np.squeeze(np.array(epoch_label_test))
        predicted_result = np.zeros(epoch_data_test.shape[0],dtype=np.float32)
        for i in range(epoch_data_test.shape[0]):
            epoch_data = epoch_data_test[i, :, :]
            epoch_result = np.zeros(len(label_list), dtype=np.float32)
            for j in range(len(label_list)):
                data_mean = data_mean_list[j]
                O1_a = O1_a_list[j]
                corr = np.zeros(int(O1_a.shape[0] / epoch_data_train.shape[2]), dtype=np.float32)
                for ind_val in range(0, int(O1_a.shape[0] / epoch_data_train.shape[2])):
                    corr[ind_val] = np.corrcoef(
                        np.dot(O1_a[ind_val * epoch_data_train.shape[2]:(ind_val + 1) * epoch_data_train.shape[2], :].T,
                               epoch_data.T),
                        np.dot(O1_a[ind_val * epoch_data_train.shape[2]:(ind_val + 1) * epoch_data_train.shape[2], :].T,
                               data_mean.T))[0, 1]
                epoch_result[j] = np.max(corr)
            predicted_result[i] = np.argmax(epoch_result)
        acc = np.sum(predicted_result == epoch_label_test)/epoch_label_test.shape[0]
        return acc

    def AssembleLearner(self):
        self.clf = Pipeline([('CSP', self.csp), ('SVC', self.svc)])

    def Learn(self, train_epochs, train_labels):
        self.clf.fit(train_epochs, train_labels)

    def EvaluateSet(self, eval_epochs, eval_labels):

        self.set_eval_score = self.clf.score(eval_epochs, eval_labels)
        return self.set_eval_score

    def compute_time_avg(self, epochs, ch, epochs_idx):
        sel_epochs = epochs[epochs_idx, ch, :]

        avg = np.mean(sel_epochs, axis=0)

        return avg

    def show_spec(self,full_epoch, full_label, sample_freq):

        class1_index = [index for (index, value) in enumerate(full_label) if value == 1.]
        class2_index = [index for (index, value) in enumerate(full_label) if value == 0.]


        avg1 = self.compute_time_avg(full_epoch, 0, class1_index)
        avg2 = self.compute_time_avg(full_epoch, 1, class1_index)
        plot_spectogram(avg1, sample_freq)
        plot_spectogram(avg2, sample_freq)

    def gen_csp_tsne_fig(self, full_epoch, full_label):
        self.csp.fit(full_epoch, full_label)

        class1_index = [index for (index, value) in enumerate(full_label) if value == 1.]
        class2_index = [index for (index, value) in enumerate(full_label) if value == 0.]

        data1_class1 = full_epoch[class1_index]
        data1_class2 = full_epoch[class2_index]
        csp_data = self.csp.transform(full_epoch)
        data2_class1 = csp_data[class1_index]
        data2_class2 = csp_data[class2_index]
        ####kl-diver#####
        data1_kl_1 = KL_divergence(data1_class1, data1_class2)
        data1_kl_1[np.isinf(data1_kl_1)] = 0.
        data1_kl_1 = np.mean(data1_kl_1)
        data1_kl_2 = KL_divergence(data1_class2,data1_class1)
        data1_kl_2[np.isinf(data1_kl_2)] = 0.
        data1_kl_2 = np.mean(data1_kl_2)
        data_klo = (data1_kl_1+data1_kl_2)/2.

        data2_kl_1 = KL_divergence(data2_class2, data2_class1)
        data2_kl_1[np.isinf(data2_kl_1)] = 0.
        data2_kl_1 = np.mean(data2_kl_1)
        data2_kl_2 = KL_divergence(data2_class1, data2_class2)
        data2_kl_2[np.isinf(data2_kl_2)] = 0.
        data2_kl_2 = np.mean(data2_kl_2)
        data_klc = (data2_kl_1+data2_kl_2)/2.
        ####tsne#####
        resulto = self.tsne.fit_transform((full_epoch ** 2).mean(axis=2))
        resultc = self.tsne.fit_transform(csp_data)
        plot_embedding(resulto, resultc, full_label, data_klo, data_klc)

    def cross_evaluate_set(self, full_epochs, full_labels, n_iter=10, shuffle=True, randaom_seed=2022):
        val_score = []
        self.folds = KFold(n_splits=n_iter, shuffle=shuffle, random_state=randaom_seed)
        for fold_, (trn_idx, val_idx) in enumerate(self.folds.split(full_epochs, full_labels)):
            trn_idx = list(trn_idx.astype(np.int32))
            val_idx = list(val_idx.astype(np.int32))
            print("fold n°{}".format(fold_ + 1))
            train_data = np.concatenate([np.expand_dims(full_epochs[i],0) for i in trn_idx],axis=0)
            train_label = np.concatenate([np.expand_dims(full_labels[i],0) for i in trn_idx],axis=0)
            val_data = np.concatenate([np.expand_dims(full_epochs[i],0) for i in val_idx],axis=0)
            val_label = np.concatenate([np.expand_dims(full_labels[i],0) for i in val_idx],axis=0)
            self.Learn(train_data,train_label)
            val_score.append(self.clf.score(val_data, val_label))
        self.mean_score = np.mean(val_score)
        return self.mean_score

    def cca_cross_validation(self, train_data, train_label, test_data, test_label, CCA, freq=None, band_list=None, n_iter=10, n_comonents=1, shuffle=True, randaom_seed=2022):
        val_score = []
        val_p = []
        val_r = []
        self.folds = KFold(n_splits=n_iter, shuffle=shuffle, random_state=randaom_seed)
        for fold_, (trn_idx, val_idx) in enumerate(self.folds.split(train_data)):
            trn_idx = list(trn_idx.astype(np.int32))
            print("fold n°{}".format(fold_ + 1))
            epoch_train_data = train_data[trn_idx]
            epoch_train_label = [train_label[i] for i in trn_idx]
            if CCA=='cca':
                acc , p_class, r_class = self.CCA_fit_transform(n_comonents, test_data, test_label, freq)
            elif CCA=='fbcca':
                acc , p_class, r_class = self.FBCCA_fit_transform(n_comonents, test_data, test_label, band_list, freq, 250)
            elif CCA=='ecca':
                acc , p_class, r_class = self.ECCA_fit_transform(epoch_train_data, epoch_train_label, freq, test_data, test_label)
            elif CCA=='trca':
                acc , p_class, r_class = self.TRCA_fit_transform(epoch_train_data, epoch_train_label, test_data, test_label)
            elif CCA=='itcca':
                acc , p_class, r_class = self.IT_CCA_fit_transform(n_comonents, epoch_train_data, epoch_train_label, test_data, test_label)
            elif CCA=='l1mcca':
                acc , p_class, r_class = self.L1_MCCA_fit_transform(epoch_train_data, test_data, test_label, freq, 200, 100, 0, 0, 0.05)
            elif CCA=='msetcca':
                acc , p_class, r_class = self.Mset_CCA_fit_transform(epoch_train_data, epoch_train_label, test_data, test_label, 1)
            else:
                print('Method not exist!')
                exit()
            val_score.append(acc)
            val_p.append(p_class)
            val_r.append(r_class)
        self.mean_score = np.mean(val_score)
        self.p = np.mean(val_p)
        self.r = np.mean(val_r)
        return self.mean_score*100., p_class*100., r_class*100.

    def EvaluateEpoch(self, epoch, out_param='label'):
        if out_param == 'prob':
            self.guess = self.clf.predict_proba(epoch)
        else:
            self.guess = self.clf.predict(epoch)
        return self.guess


class Filter:
    def __init__(self, fl, fh, srate, forder, filt_type='iir', band_type='band', win_type=None):
        
        nyq = 0.5 * srate
        low = fl / nyq
        high = fh / nyq
        self.win_type = win_type

        if filt_type == 'iir':
            self.b, self.a = sp.iirfilter(forder, [low, high], btype=band_type)

        elif filt_type == 'fir':
            if self.win_type == "butterworth":
                self.b, self.a = butter(forder, [low, high], btype=band_type)
            else:
                self.b = sp.firwin(forder, [low, high], window=self.win_type, pass_zero=False)
                self.a = [1]


    def ApplyFilter(self, data_in):
        if self.win_type == "butterworth":
            data_out = lfilter(self.b, self.a, data_in)
        else:
            data_out = sp.filtfilt(self.b, self.a, data_in)

        return data_out

    def ComputeEnergy(self, data_in):

        data_squared = data_in ** 2
        energy = np.mean(data_squared, axis = 0)

        return energy

    def GenerateWindow(self, win_len, n_seg, w_type = 'black'):
        ov = 0.5 # windows overlap

        seg_len = int(win_len / math.floor((n_seg * ov) + 1))

        print(seg_len)

        if w_type == 'han':
            win_seg = np.hanning(seg_len)

        if w_type == 'ham':
            win_seg = np.hamming(seg_len)

        if w_type == 'black':
            win_seg = np.blackman(seg_len)

        self.window = np.zeros(win_len)

        idx = np.array(range(seg_len))
        for i in range(n_seg):
            new_idx = idx + seg_len*ov*i
            new_idx = new_idx.astype(int)
            self.window[new_idx] = self.window[new_idx] + win_seg


def magnitude_spectrum_features(segmented_data, FFT_PARAMS):
    '''
    Returns magnitude spectrum features. Fast Fourier Transform computed based on
    the FFT parameters provided as input.

    Args:
        segmented_data (numpy.ndarray): epoched eeg data of shape
        (num_classes, num_channels, num_trials, number_of_segments, num_samples).
        FFT_PARAMS (dict): dictionary of parameters used for feature extraction.
        FFT_PARAMS['resolution'] (float): frequency resolution per bin (Hz).
        FFT_PARAMS['start_frequency'] (float): start frequency component to pick from (Hz).
        FFT_PARAMS['end_frequency'] (float): end frequency component to pick upto (Hz).
        FFT_PARAMS['sampling_rate'] (float): sampling rate (Hz).

    Returns:
        (numpy.ndarray): magnitude spectrum features of the input EEG.
        (n_fc, num_channels, num_classes, num_trials, number_of_segments).
    '''

    num_classes = segmented_data.shape[0]
    num_chan = segmented_data.shape[1]
    num_trials = segmented_data.shape[2]
    number_of_segments = segmented_data.shape[3]
    fft_len = segmented_data[0, 0, 0, 0, :].shape[0]

    NFFT = round(FFT_PARAMS['sampling_rate'] / FFT_PARAMS['resolution'])
    fft_index_start = int(round(FFT_PARAMS['start_frequency'] / FFT_PARAMS['resolution']))
    fft_index_end = int(round(FFT_PARAMS['end_frequency'] / FFT_PARAMS['resolution'])) + 1

    features_data = np.zeros(((fft_index_end - fft_index_start),
                              segmented_data.shape[1], segmented_data.shape[0],
                              segmented_data.shape[2], segmented_data.shape[3]))

    for target in range(0, num_classes):
        for channel in range(0, num_chan):
            for trial in range(0, num_trials):
                for segment in range(0, number_of_segments):
                    temp_FFT = np.fft.fft(segmented_data[target, channel, trial, segment, :], NFFT) / fft_len
                    magnitude_spectrum = 2 * np.abs(temp_FFT)
                    features_data[:, channel, target, trial, segment] = magnitude_spectrum[
                                                                        fft_index_start:fft_index_end, ]

    return features_data


def complex_spectrum_features(segmented_data, FFT_PARAMS):
    '''
    Returns complex spectrum features. Fast Fourier Transform computed based on
    the FFT parameters provided as input. The real and imaginary parts of the input
    signal are concatenated into a single feature vector.

    Args:
        segmented_data (numpy.ndarray): epoched eeg data of shape
        (num_classes, num_channels, num_trials, number_of_segments, num_samples).
        FFT_PARAMS (dict): dictionary of parameters used for feature extraction.
        FFT_PARAMS['resolution'] (float): frequency resolution per bin (Hz).
        FFT_PARAMS['start_frequency'] (float): start frequency component to pick from (Hz).
        FFT_PARAMS['end_frequency'] (float): end frequency component to pick upto (Hz).
        FFT_PARAMS['sampling_rate'] (float): sampling rate (Hz).

    Returns:
        (numpy.ndarray): complex spectrum features of the input EEG.
        (2*n_fc, num_channels, num_classes, num_trials, number_of_segments)
    '''

    num_classes = segmented_data.shape[0]
    num_chan = segmented_data.shape[1]
    num_trials = segmented_data.shape[2]
    number_of_segments = segmented_data.shape[3]
    fft_len = segmented_data[0, 0, 0, 0, :].shape[0]

    NFFT = round(FFT_PARAMS['sampling_rate'] / FFT_PARAMS['resolution'])
    fft_index_start = int(round(FFT_PARAMS['start_frequency'] / FFT_PARAMS['resolution']))
    fft_index_end = int(round(FFT_PARAMS['end_frequency'] / FFT_PARAMS['resolution'])) + 1

    features_data = np.zeros((2 * (fft_index_end - fft_index_start),
                              segmented_data.shape[1], segmented_data.shape[0],
                              segmented_data.shape[2], segmented_data.shape[3]))

    for target in range(0, num_classes):
        for channel in range(0, num_chan):
            for trial in range(0, num_trials):
                for segment in range(0, number_of_segments):
                    temp_FFT = np.fft.fft(segmented_data[target, channel, trial, segment, :], NFFT) / fft_len
                    real_part = np.real(temp_FFT)
                    imag_part = np.imag(temp_FFT)
                    features_data[:, channel, target, trial, segment] = np.concatenate((
                        real_part[fft_index_start:fft_index_end, ],
                        imag_part[fft_index_start:fft_index_end, ]), axis=0)

    return features_data