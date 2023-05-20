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

in_channels = args.in_channels
out_channels = args.out_channels
kernel_channels = args.kernel_channels

os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2, 3"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load dataset
dirc_test = os.path.join(dirc_data, name_data, 'input')
img_list = os.listdir(dirc_test)
img_list.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))

transform_test = transforms.Compose([Normalize(), ToTensor()])
transform_inv = transforms.Compose([Denormalize(), ToNumpy()])


# setup network
netG = UNet(in_channels, out_channels, kernel_channels)
if torch.cuda.device_count() > 1:
    # multi gpu
    netG = nn.DataParallel(netG)
netG.to(device)

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

    for i in range(len(img_list)):
        print(i + 1)

        # load data
        input = tifffile.imread(os.path.join(dirc_test, img_list[i])).astype(np.float32) / (2 ** 16 - 1)
        input = 2 * input - 1 # normalize: [0, 1] --> [-1, 1]
        input = torch.from_numpy(input).unsqueeze(0)

        input = input.to(device)

        # forward
        output = netG(input)
        
        # save results
        output = (output + 1) / 2
        output = (output.squeeze().detach().cpu().numpy() * 65535).astype(np.uint16)

        tifffile.imwrite(os.path.join(dirc_result, img_list[i].replace('input', 'output')), output)