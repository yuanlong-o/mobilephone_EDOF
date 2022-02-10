import torch
import os
import numpy as np
import tifffile
from skimage import transform

class Train_Dataset(torch.utils.data.Dataset):
    def __init__(self, dirc_data, transform_train):
        self.dirc_data = dirc_data
        self.list_data = os.listdir(self.dirc_data)
        self.list_data.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
        # self.list_data.sort()
        self.transform_train = transform_train

    def __getitem__(self, index):
        data = tifffile.imread(os.path.join(self.dirc_data, self.list_data[index % len(self.list_data)]))
        
        data = data / (2 ** 16 - 1)
        data = data.transpose((1, 2, 0)).astype(np.float32)

        w = int(data.shape[1] / 2)

        dataA = data[:, : w, :]
        dataB = data[:, w :, :]

        data = {'dataA' : dataB, 'dataB' : dataA}

        data = self.transform_train(data)

        return data

    def __len__(self):
        return len(self.list_data) * 20


class Test_Dataset(torch.utils.data.Dataset):
    def __init__(self, dirc_data, transform_test):
        self.dirc_data = dirc_data
        self.list_data = os.listdir(self.dirc_data)
        self.list_data.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
        # self.list_data.sort()
        self.transform_test = transform_test

    def __getitem__(self, index):
        data = tifffile.imread(os.path.join(self.dirc_data, self.list_data[index]))

        data = data / (2 ** 16 - 1)
        data = data.transpose((1, 2, 0)).astype(np.float32)

        w = int(data.shape[1] / 2)

        dataA = data[:, : w, :]
        dataB = data[:, w :, :]

        data = {'dataA' : dataB, 'dataB' : dataA}

        data = self.transform_test(data)

        return data

    def __len__(self):
        return len(self.list_data)


class ToTensor(object):
    '''Convert numpy arrays to tensors'''
    def __call__(self, data):
        # convert H x W x C to C x H x W
        
        dataA, dataB = data['dataA'], data['dataB']

        dataA = dataA.transpose((2, 0, 1)).astype(np.float32)
        dataB = dataB.transpose((2, 0, 1)).astype(np.float32)

        return {'dataA': torch.from_numpy(dataA), 'dataB': torch.from_numpy(dataB)}


class Normalize(object):
    def __call__(self, data):
        # Normalize [0, 1] -> [-1, 1]
        
        dataA, dataB = data['dataA'], data['dataB']
        dataA = 2 * dataA - 1
        dataB = 2 * dataB - 1
        
        return {'dataA': dataA, 'dataB': dataB}


class RandomHorizonFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):

        dataA, dataB = data['dataA'], data['dataB']
        if np.random.rand() > self.p:
            dataA = np.fliplr(dataA)
            dataB = np.fliplr(dataB)
        
        return {'dataA': dataA, 'dataB': dataB}


class Rescale(object):
    def __init__(self, output_size):
        self.output_size = output_size
    
    def __call__(self, data):
        dataA, dataB = data['dataA'], data['dataB']
        
        dataA = transform.resize(dataA, self.output_size)
        dataB = transform.resize(dataB, self.output_size)

        return {'dataA': dataA, 'dataB': dataB}


class RandomCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, data):
        dataA, dataB = data['dataA'], data['dataB']
        
        h, w = dataA.shape[0], dataA.shape[1]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        dataA = dataA[top: top + new_h, left: left + new_w]
        dataB = dataB[top: top + new_h, left: left + new_w]

        return {'dataA': dataA, 'dataB': dataB}


class ToNumpy(object):
    '''Convert numpy arrays to tensors.'''
    def __call__(self, data):
        # convert B x C x H x W to B x H x W x C

        # return data.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
        return data.to('cpu').detach().numpy()


class Denormalize(object):
    def __call__(self, data):
        # Denomalize [-1, 1] => [0, 1]
        
        return (data + 1) / 2


def normalize_batch(batch):
    """Normalize batch using ImageNet mean and std
    """
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

    return (batch - mean) / std

def rgb2yuv(img):
    '''only use the Y channel'''
    h, w, c = img.shape
    yuv = np.zeros((h, w, 1)).astype(np.float32)
    yuv[:, :, 0] = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    return yuv