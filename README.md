# Network for EDOF deconvolution

Pytorch implementation of our network for EDOF deconvolution

Our network takes blurry image as an input and procude the corresponding sharp estimate. 

The model we used is the corrected Pix2Pix network with MSE Loss + Perceptual loss based on VGG-19 model.


## How to run

### Prerequisites
- pytorch >= 1.5 and torchvision >=0.6 (recommend torch1.8.1 torchvision 0.8.0)
- python >= 3.6
- tifffile
- skimage
- scipy == 1.2.1
- ptflops
- tensorboard


### Datasets
Please organize the datasets using the following hierarchy:
```
- datasets
    - 4x_average_depth
        - train
          - img_1.tiff
            ...
          - img_99.tiff
        - test
          - img_1.tiff
            ...
          - img_99.tiff
```


### Train
You can use the following script to obtain the train the model:
```bash
python train.py --dirc_data ./datasets --name_data 4x_average_depth
```

All the configuration and hyperparameters can be modified in the file of 'config.py'

Note:
If you want to train the model on your data, you should concatenate the label and input images in the dimension of width (format:[label, input]). It can be implemented by numpy function easily.


### Test
You can use the following script to obtain the testing results:
```bash
python test.py --dirc_data ./datasets --name_data 4x_average_depth
```


### Tensorboard
Loss and results during the course of training have saved in the tensorboard. You can use the following command obtain them:
 ```bash
tensorboard --logdir ./log
```


## Calculate parameter
You can use the following command to calculate the network's parameter:
```bash
python params.py
```

Results of our models:(input size is [2160, 2560, 3])

|  Model  |  Parameters(M)  | FLOPs(GMac) |
| :-----: | :-------------: | :---------: |
| origin  |      54.88      |   4218.37   |
| pruning |      12.21      |   700.54    |


## Acknowledgments
Code refers to [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
