import torch
from model import UNet, UNet_Pruning
from ptflops import get_model_complexity_info

in_channels = 3
out_channels = 3
origin_kernel_channels = 64
pruning_kernel_channels = 32

with torch.cuda.device(0):
  model_origin = UNet(in_channels, out_channels, origin_kernel_channels)
  model_pruning = UNet_Pruning(in_channels, out_channels, pruning_kernel_channels)
  macs_origin, params_origin = get_model_complexity_info(model_origin, (3, 2160, 2560), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  macs_pruning, params_pruning = get_model_complexity_info(model_pruning, (3, 2160, 2560), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)

  print('Computational complexity of genarator:\t[1]origin:%s\t[2]pruning:%s' % (macs_origin, macs_pruning))
  print('Number of generator parameters:\t[1]origin:%s\t[2]pruning:%s' % (params_origin, params_pruning))

