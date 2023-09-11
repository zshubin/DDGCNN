import numpy as np
import json

from data_loader.DataProcessing import Filter, Learner

def loadDataAsMatrix(path):
    """Loads text file content as numpy matrix
    Parameters
    ----------
    path : path to text file
    
    cols : order of columns to be read

    Returns
    -------
    matrix : numpy matrix, shape as written in txt

    Examples
    --------
    >>> data_path = "/PATH/TO/FILE/somematrix.txt"
    >>> matrix_data = loadAsMatrix(data_path)
    """
    
    matrix = np.load(open(path,"rb"))

    return matrix

def extractEpochs(data, e, smin, smax , ev_id):
    """Extracts the epochs from data based on event information
    Parameters
    ----------
    data : raw data in mne format
    
    event_id : labels of each class
    
    tmin: time in seconds at which the epoch starts (event as reference) 
    
    tmax: time in seconds at which the epoch ends (event as reference) 

    Returns
    -------
    epochs: epochs in mne format
    
    labels: labels of each extracted epoch

    Examples
    --------
    >>> data, sfreq = loadBiosig(data_eval_path)
    >>> raw = mne.io.RawArray(data, info)
    >>> csv_path = "/PATH/TO/CSVFILE/events.csv"
    >>> raw = addEvents(raw, eval_events_path)
    >>> event_id = dict(LH=769, RH=770)
    >>> tmin, tmax = 1, 3 # epoch starts 1 sec after event and ends 3 sec after
    >>> epochs_train, labels_train = extractEpochs(raw, event_id, tmin, tmax)
    
    """

    events_list = e[:,2]

    cond = False

    for i in range(len(ev_id)):
        cond += (events_list == ev_id[i])

    idx = np.where(cond)[0]
    s = e[idx, 0]

    sBegin = s + smin
    sEnd = s + smax

    n_epochs = len(sBegin)
    n_channels = data.shape[0]
    n_samples = smax - smin

    epochs = np.zeros([n_epochs, n_channels, n_samples])

    labels = events_list[idx]

    bad_epoch_list = [] 
    for i in range(n_epochs):
        epoch = data[:,sBegin[i]:sEnd[i]]

        # Check if epoch is complete
        if epoch.shape[1] == n_samples:
            epochs[i,:,:] = epoch
        else:
            print('Incomplete epoch detected...')
            bad_epoch_list.append(i)

    labels = np.delete(labels, bad_epoch_list)
    epochs = np.delete(epochs, bad_epoch_list, axis = 0)

    return epochs, labels


def saveMatrixAsTxt(data_in, path, mode = 'a'):

    with open(path, mode) as data_file:    
        np.save(data_file, data_in)

def loadChannelLabels(path):
    # if os.path.exists("data/rafael/precal_config"):
    with open(path, "r") as data_file:    
        data = json.load(data_file)

    return data["ch_labels"].split(' ')

def readEvents(events_path):

    e = np.load(events_path)
    # insert dummy column to fit mne event list format
    t_events = np.insert(e, 1, values=0, axis=1)
    t_events = t_events.astype(int) # convert to integer

    return t_events


