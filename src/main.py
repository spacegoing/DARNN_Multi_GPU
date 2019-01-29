# -*- coding: utf-8 -*-
from __future__ import print_function, division
import torch
import argparse
import numpy as np
from tensorboardX import SummaryWriter
from DARNN import Encoder, Decoder
from CsiDataSet import CSI300Dataset
from VerParams import Version


def set_seed(seed=1):
  '''
  https://github.com/pytorch/pytorch/issues/11278
  https://github.com/pytorch/pytorch/issues/11278
  https://github.com/pytorch/pytorch/issues/12207
  '''
  import random
  import os
  random.seed(seed)
  np.random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)

  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True


# Determinism Seed
set_seed()

from torch import nn
from torch import optim

# Parameters settings
parser = argparse.ArgumentParser(description="DA-RNN")

# Dataset setting
parser.add_argument(
    '--norm_csi_dir',
    type=str,
    default='/project/chli/scp/CSI300_NORM/',
    help='normalized csi300 csv dir')
parser.add_argument(
    '--num_workers',
    type=int,
    default=12,
    help='number of data loading workers (default 3)')
parser.add_argument(
    '--dataset_split_ratio',
    default=[0.8, 0.1, 0.1],
    type=list,
    help='train, valid, test dataset split ratio')
parser.add_argument(
    '--x_columns',
    default=['o', 'h', 'l', 'v', 'a'],
    type=list,
    help='list of features\' (X) column names')
parser.add_argument(
    '--y_columns',
    default=['c'],
    type=list,
    help='list of target (Y) column names')
parser.add_argument(
    '--pin_memory', type=bool, default=True, help='pin memory page')
parser.add_argument(
    '--debug', type=bool, default=False, help='debug with small data')

# Encoder / Decoder parameters setting
parser.add_argument(
    '--hid_dim_encoder',
    type=int,
    default=32,
    help='size of hidden states for the encoder m [64, 128]')
parser.add_argument(
    '--hid_dim_decoder',
    type=int,
    default=32,
    help='size of hidden states for the decoder p [64, 128]')
parser.add_argument(
    '--ind_steps',
    type=int,
    default=0,
    help='window length for computing indicator')
parser.add_argument(
    '--lag_steps',
    type=int,
    default=20,
    help='the number of lag time steps (history window length T)')
parser.add_argument(
    '--pred_steps',
    type=int,
    default=1,
    help='y_{t+pred_steps} = p(y_t,...,y_{timesteps-1}, x_t,...,x_{timesteps-1})'
)

# Training parameters setting
parser.add_argument(
    '--param_version', type=int, default=None, help='int versioning params')
parser.add_argument(
    '--epochs',
    type=int,
    default=10,
    help='number of epochs to train [10, 200, 500]')
parser.add_argument(
    '--lr',
    type=float,
    default=0.001,
    help='learning rate [0.001] reduced by 0.1 after each 10000 iterations')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument(
    '--batchsize', type=int, default=512, help='input batch size [128]')
parser.add_argument('--shuffle', type=bool, default=True, help='shuffle batch')
parser.add_argument(
    '--task_type', default='single', type=str, help='single or multi')
parser.add_argument(
    '--pred_type', default='shift', type=str, help='steps or shift')

# debug
parse_cli = False
opt = parser.parse_args('')
if parse_cli:
  opt = parser.parse_args()

if __name__ == "__main__":
  # debug
  # from importlib import reload
  opt.debug = False
  opt.num_workers = 20

  # import os
  # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  ver = Version()
  ver.set_ver_opt(opt.param_version, opt)
  suffix = 'L%dP%dHdim%d' % (opt.lag_steps, opt.pred_steps, opt.hid_dim_encoder)
  writer = SummaryWriter(comment=suffix)

  csi300 = CSI300Dataset()
  train_dataset, valid_dataset, test_dataset, \
    train_loader, valid_loader, test_loader = csi300.get_dataset_loader(
      opt)
  feat_dim = 13

  encoder = Encoder(opt.lag_steps, feat_dim, opt.hid_dim_encoder)
  decoder = Decoder(opt.lag_steps, opt.hid_dim_encoder, opt.hid_dim_decoder)

  # device = ('cpu')
  # Multi-GPU Support
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if torch.cuda.device_count() > 1:
    encoder = nn.DataParallel(encoder)
    decoder = nn.DataParallel(decoder)
  encoder.to(device)
  decoder.to(device)

  criterion = nn.MSELoss()
  encoder_optimizer = optim.Adam(encoder.parameters(), lr=opt.lr)
  decoder_optimizer = optim.Adam(decoder.parameters(), lr=opt.lr)

  # Train Loops
  n_batches_count = 1
  epoch_batch_loss_list = list()
  for epoch in range(opt.epochs):
    batch_loss_list = list()
    for data_dict in train_loader:
      # Prepare Data On Devices
      X = data_dict['X'].type(torch.FloatTensor).to(device)
      Y = data_dict['Y'].type(torch.FloatTensor).squeeze().to(device)
      Ygt = data_dict['Y_gt'].type(torch.FloatTensor).to(device)

      # Forward Pass
      H = encoder(X)
      Ypred = decoder(H, Y)
      loss = criterion(Ypred.squeeze(), Ygt)

      # Gradient Descent
      encoder_optimizer.zero_grad()
      decoder_optimizer.zero_grad()
      loss.backward()
      encoder_optimizer.step()
      decoder_optimizer.step()

      # Log Stats
      if n_batches_count % 100 == 0:
        writer.add_scalar('train/loss', loss.item(), n_batches_count)
      if n_batches_count % 50000 == 0:
        for p in encoder_optimizer.param_groups:
          p['lr'] *= 0.9
        for p in decoder_optimizer.param_groups:
          p['lr'] *= 0.9
      n_batches_count += 1

    print(batch_loss_list)
    epoch_batch_loss_list.append(batch_loss_list)
