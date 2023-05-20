import argparse

# config the parameters
parser = argparse.ArgumentParser(description='Train the EDOF network')

parser.add_argument('--dirc_data', default='', dest='dirc_data')
parser.add_argument('--dirc_ckpt', default='', dest='dirc_ckpt')
parser.add_argument('--dirc_pretrain', default='', dest='dirc_pretrain')
parser.add_argument('--dirc_log', default='', dest='dirc_log')
parser.add_argument('--dirc_result', default='', dest='dirc_result')

parser.add_argument('--name_data', type=str, default='', dest='name_data')

parser.add_argument('--input_size', type=int, default=512, dest='input_size')
parser.add_argument('--batch_size', type=int, default=16, dest='batch_size')

parser.add_argument('--num_epoch', type=int,  default=300, dest='num_epoch')

parser.add_argument('--in_channels', type=int, default=3, dest='in_channels')
parser.add_argument('--out_channels', type=int, default=3, dest='out_channels')
parser.add_argument('--kernel_channels', type=int, default=64, dest='kernel_channels')

parser.add_argument('--lr', type=float, default=2e-4, dest='lr')

args = parser.parse_args()

