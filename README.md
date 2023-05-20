# Generation of training pairs for network

To achieve the simulation-supervision training and increase the generalization of the neural network, we proposed a forward propagation model to simulate physically the imaging process of the miniaturized microscope.

The corresponding implementation is written in MATLAB.

# Network for EDOF deconvolution

Pytorch implementation of our network for EDOF deconvolution

Our network takes blurry image as an input and procude the corresponding sharp estimate. 

The model we used is the corrected Pix2Pix network with MSE Loss + Perceptual loss based on VGG-19 model.


## How to run

### Prerequisites
- NVIDIA GPU (24 GB Memory) + CUDA
- pytorch >= 1.5 and torchvision >=0.6 (recommend torch1.8.1 torchvision 0.8.0)
- python >= 3.6
- tifffile
- skimage
- scipy == 1.2.1
- ptflops
- tensorboard

### Test with our demo data
We upload test data(including input and output images) and pre-trained network checkpoint on https://doi.org/10.5281/zenodo.7950911.

Please use the following script to test:
```bash
python test_demo.py --dirc_data ./EDOF_data_ckpt/data --name_data 4x_average_depth --dirc_ckpt ./EDOF_data_ckpt/network_ckpt --dirc_result ./result
```

### Train and Test with your own datasets

If you want to apply our model to your own datasets, please organize the datasets using the following hierarchy:
```
- datasets
    - <your_dataset_name>
        - train
          - img_1.tiff
            ...
          - img_99.tiff
        - test
          - img_1.tiff
            ...
          - img_99.tiff
```
Note: You should concatenate the label and input images in the dimension of width (format:[label, input]). It can be implemented by numpy function easily.

You can use the following script to train the model:
```bash
python train.py --dirc_data ./datasets --name_data <your_dataset_name> --dirc_ckpt ./output_dir --dirc_log ./output_dir/log
```

If you want to finetune the network with our pre-trained checkpoint, please use the following script:
```bash
python train.py --dirc_data ./datasets --name_data <your_dataset_name> --dirc_pretrain <your_pretrained_ckpt_dir> --dirc_ckpt ./output_dir --dirc_log ./output_dir/log
```

You can use the following script to test the model:
```bash
python test.py --dirc_data ./datasets --name_data <your_dataset_name> --dirc_ckpt ./output_dir --dirc_result ./output_dir/result
```

### Tensorboard
Loss and output images during training have saved in the tensorboard. You can view it by:
 ```bash
tensorboard --logdir ./log
```

## Calculate parameters
You can use the following script to calculate the network's parameter:
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
