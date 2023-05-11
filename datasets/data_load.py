import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.io import savemat
from datasets.AuxFunction import FFT, add_noise


# generate Training Dataset and Testing Dataset
def get_files(args):
    Subdir = []
    data_train = []
    data_validation = []
    data_test = []

    sub_root = os.path.join(args.data_dir)  # the location of dataset
    file_name = os.listdir(sub_root)  # the all the data
    for j in file_name:
        Subdir.append(os.path.join(sub_root, j))

    for i in tqdm(range(len(Subdir))):
        data = data_load(args, Subdir[i])  # loading the dataset
        if i < 3*len(Subdir)//4:
            if i % 3 == 2:
                data_train += list(np.concatenate([data, np.zeros(args.sample_size).reshape(-1, 1)], axis=1))
            elif i % 3 == 1:
                data_validation += list(np.concatenate([data, np.zeros(args.sample_size).reshape(-1, 1)], axis=1))
            else:
                data_test += list(np.concatenate([data, np.zeros(args.sample_size).reshape(-1, 1)], axis=1))
        else:
            data_test += list(np.concatenate([data, np.ones(args.sample_size).reshape(-1, 1)], axis=1))

    save_dir = os.path.join(
        './mats'.format(args.snr))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    savemat('{}/snr_{}.mat'.format(save_dir, args.snr),
            {'data_train': data_train, 'data_validation': data_validation, 'data_test': data_test})


def data_load(args, root):
    name = locals()
    for i in range(3):
        name['data' + str(i)] = pd.read_csv(root, sep=',', usecols=[i+3],  header=None)
        name['data' + str(i)] = name['data' + str(i)].values.reshape(-1,)
        name['data' + str(i)] = add_noise(name['data' + str(i)], args.snr)
        name['data' + str(i)] = (name['data' + str(i)] - name['data' + str(i)].min()) / (
                name['data' + str(i)].max() - name['data' + str(i)].min())

    data = []
    start, end = 10000, 10000+args.sample_length
    for i in range(args.sample_size):
        x = np.concatenate(
            [FFT(name['data0'][start:end]), FFT(name['data1'][start:end]), FFT(name['data2'][start:end])], axis=0)
        data.append(x)
        start += args.sample_length
        end += args.sample_length

    return data

