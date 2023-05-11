import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def datasets(args, mode):
    dataset = Set_division(args, mode)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=args.batch_size,
                             shuffle=True if mode == 'train' else False)
    return data_loader, len(dataset)


class Set_division(Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode
        train_data = loadmat('./mats/snr_{}'.format(args.snr))['data_train']
        np.random.shuffle(train_data)
        train_data = torch.tensor(train_data, dtype=torch.float)
        validation_data = loadmat('./mats/snr_{}'.format(args.snr))['data_validation']
        validation_data = torch.tensor(validation_data, dtype=torch.float)
        test_data = loadmat('./mats/snr_{}'.format(args.snr))['data_test']
        test_data = torch.tensor(test_data, dtype=torch.float)

        self.train_data = train_data.to(torch.device(args.device))
        self.validation_data = validation_data.to(torch.device(args.device))
        self.test_data = test_data.to(torch.device(args.device))

    def __len__(self):
        if self.mode == 'train':
            return self.train_data.shape[0]
        elif self.mode == 'val':
            return self.validation_data.shape[0]
        else:
            return self.test_data.shape[0]

    def __getitem__(self, index):
        if self.mode == 'train':
            return self.train_data[index, :-1], self.train_data[index, -1]
        elif self.mode == 'val':
            return self.validation_data[index, :-1], self.validation_data[index, -1]
        else:
            return self.test_data[index, :-1], self.test_data[index, -1]