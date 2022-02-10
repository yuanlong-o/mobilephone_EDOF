import torch
import torch.nn as nn
import torch.optim as optim
import os
import utils
from model import UNet, PatchGAN, UNet_Pruning
from dataset import *
from config import args
from initial import init_weights, set_requires_grad
from loss import VGG19
from torchvision import transforms
from statistics import mean
from torch.utils.tensorboard import SummaryWriter

# config
dirc_data = args.dirc_data
name_data = args.name_data

dirc_ckpt = args.dirc_ckpt
if not os.path.exists(dirc_ckpt):
    os.makedirs(dirc_ckpt)

dirc_log = args.dirc_log
if not os.path.exists(dirc_log):
    os.makedirs(dirc_log)

input_size = args.input_size
batch_size = args.batch_size

in_channels = args.in_channels
out_channels = args.out_channels
kernel_channels = args.kernel_channels

lr = args.lr

num_epoch = args.num_epoch

os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2, 3"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load dataset
dirc_train = os.path.join(dirc_data, name_data, 'test')

transform_train = transforms.Compose([RandomCrop((input_size, input_size)), RandomHorizonFlip(), Normalize(), ToTensor()])
transform_inv = transforms.Compose([Denormalize()])
dataset_train = Train_Dataset(dirc_train, transform_train)
loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

num_train = len(dataset_train)
num_batch_train = int(num_train / batch_size)

# setup network
netG = UNet(in_channels, out_channels, kernel_channels)
netD = PatchGAN(2 * out_channels, kernel_channels)
vgg = VGG19(requires_grad=False)

if torch.cuda.device_count() > 1:
    # multi gpu
    netG = nn.DataParallel(netG)
    netD = nn.DataParallel(netD)
    vgg = nn.DataParallel(vgg)

netG.to(device)
netD.to(device)
vgg.to(device)

if args.dirc_pretrain == '':
    init_weights(netG, init_type='normal')
    init_weights(netD, init_type='normal')
else:
    netG.load_state_dict(torch.load('%s/checkpoint_latest.pth' % args.dirc_pretrain)['netG'])
    netD.load_state_dict(torch.load('%s/checkpoint_latest.pth' % args.dirc_pretrain)['netD'])

# setup loss
criterion_L2 = nn.MSELoss().to(device)
criterion_L1 = nn.L1Loss().to(device)
criterion_GAN = nn.BCEWithLogitsLoss().to(device)

# setup optimizer
paramsG = netG.parameters()
paramsD = netD.parameters()

optimizerG = optim.AdamW(paramsG, lr=lr, betas=(0.9, 0.999))
optimizerD = optim.AdamW(paramsD, lr=lr, betas=(0.9, 0.999))

schedulerG = utils.WarmupLinearDecayLR(
            optimizerG,
            warmup_factor=0.01,
            warmup_iters=10,
            warmup_method="linear",
            end_epoch=num_epoch,
            final_lr_factor=0.01)
schedulerD = utils.WarmupLinearDecayLR(
            optimizerG,
            warmup_factor=0.01,
            warmup_iters=10,
            warmup_method="linear",
            end_epoch=num_epoch,
            final_lr_factor=0.01)

# optimizerG = optim.Adam(paramsG, lr=lr, betas=(0.5, 0.999))
# optimizerD = optim.Adam(paramsD, lr=lr, betas=(0.5, 0.999))

# schedulerG = optim.lr_scheduler.MultiStepLR(optimizerG, milestones=[100,200], gamma=0.1, last_epoch=-1)
# schedulerD = optim.lr_scheduler.MultiStepLR(optimizerD, milestones=[100,200], gamma=0.1, last_epoch=-1)

# setup tensorboard
writer_train = SummaryWriter(log_dir=dirc_log)

# train
print_num = 10
for epoch in range(1, num_epoch + 1):

    netG.train()
    netD.train()

    lossG_GAN_train = []
    lossG_L2_train = []
    lossG_Content_train = []
    lossD_real_train = []
    lossD_fake_train = []

    for i, data in enumerate(loader_train, 1):
        # load data
        input = data['dataA'].to(device)
        label = data['dataB'].to(device)

        # forward netG
        output = netG(input)
        features_label = vgg(normalize_batch(transform_inv(label)))
        features_output = vgg(normalize_batch(transform_inv(output)))

        # backward netD
        set_requires_grad(netD, True)
        optimizerD.zero_grad()

        real = torch.cat([input, label], dim=1)
        fake = torch.cat([input, output], dim=1)

        predict_real = netD(real)
        predict_fake = netD(fake.detach())

        lossD_real = criterion_GAN(predict_real, torch.ones_like(predict_real))
        lossD_fake = criterion_GAN(predict_fake, torch.zeros_like(predict_fake))
        lossD = 0.5 * (lossD_real + lossD_fake)

        lossD.backward()

        # update optimizerD
        optimizerD.step()

        # backward netG
        set_requires_grad(netD, False)
        optimizerG.zero_grad()

        fake = torch.cat([input, output], dim=1)
        predict_fake = netD(fake)

        lossG_GAN = criterion_GAN(predict_fake, torch.ones_like(predict_fake))
        lossG_L2 = criterion_L2(label, output)
        lossG_Content = criterion_L2(features_label, features_output)
        lossG = lossG_GAN + 100 * lossG_L2 + lossG_Content

        lossG.backward()

        #update optimizerG
        optimizerG.step()

        # get loss
        lossG_GAN_train +=[lossG_GAN.item()]
        lossG_L2_train +=[lossG_L2.item()]
        lossG_Content_train +=[lossG_Content.item()]
        lossD_real_train +=[lossD_real.item()]
        lossD_fake_train +=[lossD_fake.item()]

        if i % print_num == 0:
            print('Train: Epoch %d: Lr: %f Batch %04d/%04d: ' 'LossG_GAN: %.4f LossG_L2: %.4f LossG_Content: %.4f LossD_Real: %.4f LossD_Fake: %.4f' 
                %(epoch, optimizerG.param_groups[0]['lr'], i, num_batch_train, mean(lossG_GAN_train), mean(lossG_L2_train), mean(lossG_Content_train), \
                mean(lossD_real_train), mean(lossD_fake_train)))
    
    writer_train.add_scalar('LossG_GAN_Train', mean(lossG_GAN_train), epoch)
    writer_train.add_scalar('LossG_L2_Train', mean(lossG_L2_train), epoch)
    writer_train.add_scalar('LossG_Content_Train', mean(lossG_Content_train), epoch)
    writer_train.add_scalar('LossD_Real_Train', mean(lossD_real_train), epoch)
    writer_train.add_scalar('LossD_Fake_Train', mean(lossD_fake_train), epoch)

    # save images_train in tensorboard
    if (epoch % 10) == 0:
        for j in range(batch_size):
            writer_train.add_image('Epoch' + str(epoch) + '_label' + str(j + 1), transform_inv(label)[j, :, :, :])
            writer_train.add_image('Epoch' + str(epoch) + '_output' + str(j + 1), transform_inv(output)[j, :, :, :])

    # save model
    if (epoch % 50) == 0:
        torch.save({'netG': netG.state_dict(), 'netD': netD.state_dict()}, '%s/checkpoint_%04d.pth' % (dirc_ckpt, epoch))
    torch.save({'netG': netG.state_dict(), 'netD': netD.state_dict()}, '%s/checkpoint_latest.pth' % dirc_ckpt)

    schedulerG.step()
    schedulerD.step()

writer_train.close()

