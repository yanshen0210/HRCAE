
import argparse
import os
from utils.logger import setlogger
import logging

from utils.train_val_test import train, val, test
from datasets.data_load import get_files


def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    '''data prepare'''
    parser.add_argument('--save_data', type=bool, default=True, help='whether saving as mat')
    parser.add_argument('--data_dir', type=str, default="./data", help='the directory of the data folder')
    parser.add_argument('--sample_length', type=int, default=1024, help='the length of each sample')
    parser.add_argument('--sample_size', type=int, default=200, help='the number of samples for each note')

    parser.add_argument('--snr', type=int, default=3, help='the snr of the Gaussian noise')
    parser.add_argument('--model_name', type=str, default='HRCAE', help='HRCAE, CAE, MAE')
    parser.add_argument('--lambd', type=int, default=0.2, help='the parameter of the cos loss')

    parser.add_argument('--batch_size', type=int, default=128, help='the number of samples for each batch')
    parser.add_argument('--epoch', type=int, default=300, help='the number of epoch')
    parser.add_argument('--lr', type=float, default=0.01, help='the initial learning rate')
    parser.add_argument('--steps', type=str, default='20,200', help='the learning rate decay for step and stepLR')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--lr_scheduler', type=str, choices=['MultiStep', 'Exp', 'StepLR'], default='MultiStep',
                        help='the learning rate schedule')
    parser.add_argument('--print_epoch', type=int, default=5, help='the epoch of log printing')
    parser.add_argument('--percentage_threshold', type=int, default=95, help='the percentage of the value threshold')
    parser.add_argument('--device', type=str, default='cuda:0', help='choose GPU or CPU')

    args = parser.parse_args()
    return args


args = parse_args()
# saving the data as mat
if args.save_data:
    get_files(args)

save_dir = os.path.join(
    './results/snr_{}'.format(args.snr))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# set the logger
setlogger(os.path.join(save_dir, args.model_name + '.log'))

# save the args
for k, v in args.__dict__.items():
    logging.info("{}: {}".format(k, v))

train(args)
threshold = val(args)
test(args, threshold)













