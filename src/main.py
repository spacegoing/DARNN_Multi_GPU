# -*- coding: utf-8 -*-
from __future__ import print_function, division
import torch
import argparse
import numpy as np
import pandas as pd
from DARNN import Encoder, Decoder
from PrepareData import Nas100Dataset


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
import torch.nn.functional as F

# Parameters settings
parser = argparse.ArgumentParser(description="DA-RNN")

# Dataset setting
parser.add_argument(
    '--dataroot',
    type=str,
    default="../nasdaq/nasdaq100_padding.csv",
    help='path to dataset')
parser.add_argument(
    '--num_workers',
    type=int,
    default=1,
    help='number of data loading workers [2]')
parser.add_argument(
    '--batchsize', type=int, default=128, help='input batch size [128]')
parser.add_argument(
    '--train_ratio', type=float, default=0.7, help='train set ratio')
parser.add_argument('--shuffle', type=bool, default=False, help='shuffle batch')
parser.add_argument(
    '--pin_memory', type=bool, default=True, help='pin memory page')
parser.add_argument(
    '--debug', type=bool, default=False, help='debug with small data')

# Encoder / Decoder parameters setting
parser.add_argument(
    '--hid_dim_encoder',
    type=int,
    default=128,
    help='size of hidden states for the encoder m [64, 128]')
parser.add_argument(
    '--hid_dim_decoder',
    type=int,
    default=128,
    help='size of hidden states for the decoder p [64, 128]')
parser.add_argument(
    '--timesteps',
    type=int,
    default=9,
    help='the number of time steps in the window T [9]')
parser.add_argument(
    '--pred_timesteps',
    type=int,
    default=1,
    help=
    'y_{t+pred_timesteps} = p(y_t,...,y_{timesteps-1}, x_t,...,x_{timesteps-1})'
)

# Training parameters setting
parser.add_argument(
    '--epochs',
    type=int,
    default=80,
    help='number of epochs to train [10, 200, 500]')
parser.add_argument(
    '--lr',
    type=float,
    default=0.001,
    help='learning rate [0.001] reduced by 0.1 after each 10000 iterations')
parser.add_argument('--seed', default=1, type=int, help='manual seed')

if __name__ == "__main__":
  # import os
  # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  opt = parser.parse_args('')

  dataset = Nas100Dataset()
  pre_dataset, train_loader = dataset.get_data_loader(opt)
  feat_dim = 81

  encoder = Encoder(opt.timesteps, feat_dim, opt.hid_dim_encoder)
  decoder = Decoder(opt.timesteps, opt.hid_dim_encoder, opt.hid_dim_decoder)

  # device = ('cpu')
  # Multi-GPU Support
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # if torch.cuda.device_count() > 1:
  #   encoder = nn.DataParallel(encoder)
  #   decoder = nn.DataParallel(decoder)
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
      Ygt = data_dict['Ygt'].type(torch.FloatTensor).to(device)

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
      batch_loss_list.append(loss.item())
      if n_batches_count % 50000 == 0:
        for p in encoder_optimizer.param_groups:
          p['lr'] *= 0.9
        for p in decoder_optimizer.param_groups:
          p['lr'] *= 0.9
      n_batches_count += 1

    print(batch_loss_list)
    epoch_batch_loss_list.append(batch_loss_list)
