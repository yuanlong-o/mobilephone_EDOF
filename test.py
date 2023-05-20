import torch
import torch.nn as nn
import os
import tifffile
import numpy as np
from model import UNet
from dataset import *
from config import *
from torchvision import transforms
from collections import OrderedDict

dirc_data = args.dirc_data
dirc_ckpt = args.dirc_ckpt
dirc_result = args.dirc_result
if not os.path.exists(dirc_result):
    os.makedirs(dirc_result)
name_data = args.name_data

batch_size = 1

in_channels = args.in_channels
out_channels = args.out_channels
kernel_channels = args.kernel_channels

os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2, 3"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

# load dataset
dirc_test = os.path.join(dirc_data, name_data, 'test')
transform_test = transforms.Compose([Normalize(), ToTensor()])
transform_inv = transforms.Compose([Denormalize(), ToNumpy()])

dataset_test = Test_Dataset(dirc_test, transform_test)

loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

num_test = len(dataset_test)
num_batch_test = int(num_test / batch_size)

# setup network
netG = UNet(in_channels, out_channels, kernel_channels)

if torch.cuda.device_count() > 1:
    # multi gpu
    netG = nn.DataParallel(netG)

netG.to(device)

# setup loss
criterion_L1 = nn.L1Loss().to(device)

# load checkpoint
state_dict = torch.load(os.path.join(dirc_ckpt, 'checkpoint_latest.pth'))['netG']
netG.load_state_dict(state_dict)

# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k.replace('module.', '')
#     new_state_dict[name] = v
# netG.load_state_dict(new_state_dict)

# test
with torch.no_grad():
    netG.eval()

    for i, data in enumerate(loader_test):
        print(i)
        # load data
        input = data['dataA'].to(device)
        label = data['dataB'].to(device)

        # forward
        output = netG(input)
        
        # save results
        input = transform_inv(input)
        output = transform_inv(output)
        label = transform_inv(label)

        input = (input * 65535).astype(np.uint16)
        output = (output * 65535).astype(np.uint16)
        label = (label * 65535).astype(np.uint16)

        for j in range(batch_size):
            count = batch_size * (i - 1) + j + 1
            fileset = {'count': count, 'input': "input_%d.tiff" % count, 'output': "output_%d.tiff" % count, 'label': "label_%d.tiff" % count}

            tifffile.imwrite(os.path.join(dirc_result, fileset['input']), input[j, :, :, :])
            tifffile.imwrite(os.path.join(dirc_result, fileset['output']), output[j, :, :, :])
            tifffile.imwrite(os.path.join(dirc_result, fileset['label']), label[j, :, :, :])