# -*- coding: utf-8 -*-
from __future__ import print_function, division
import torch
import argparse
import numpy as np
from tensorboardX import SummaryWriter
from DARNN import Encoder, Decoder
from CsiDataSet import CSI300MultiTaskDataset
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
  opt.ind_steps = 10
  opt.hid_dim_encoder = 128
  opt.hid_dim_decoder = 128

  # import os
  # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  ver = Version()
  ver.set_ver_opt(opt.param_version, opt)
  suffix = 'L%dP%dHdim%d' % (opt.lag_steps, opt.pred_steps, opt.hid_dim_encoder)
  writer = SummaryWriter(comment=suffix)

  csi300 = CSI300MultiTaskDataset()
  train_dataset, valid_dataset, test_dataset, \
    train_loader, valid_loader, test_loader = csi300.get_dataset_loader(
      opt)

  trend_feat_dim = 5
  volat_feat_dim = 5
  multi_feat_dim = 2 * opt.hid_dim_encoder + trend_feat_dim

  # multi task training sampling probility
  def sample_task_id():
    multi_train_sample_prob = [1 / 3] * 3
    return np.argmax(np.random.multinomial(1, multi_train_sample_prob))

  trend_encoder = Encoder(opt.lag_steps, trend_feat_dim, opt.hid_dim_encoder)
  trend_decoder = Decoder(opt.lag_steps, opt.hid_dim_encoder,
                          opt.hid_dim_decoder)
  volat_encoder = Encoder(opt.lag_steps, volat_feat_dim, opt.hid_dim_encoder)
  volat_decoder = Decoder(opt.lag_steps, opt.hid_dim_encoder,
                          opt.hid_dim_decoder)
  multi_encoder = Encoder(opt.lag_steps, multi_feat_dim, opt.hid_dim_encoder)
  multi_decoder = Decoder(opt.lag_steps, opt.hid_dim_encoder,
                          opt.hid_dim_decoder)
  multi_list = [[trend_encoder, trend_decoder],
                [volat_encoder, volat_decoder],
                [multi_encoder, multi_decoder]]

  # device = ('cpu')
  # Multi-GPU Support
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if torch.cuda.device_count() > 1:
    for i in multi_list:
      i[0] = nn.DataParallel(i[0])
      i[1] = nn.DataParallel(i[1])
  for encoder, decoder in multi_list:
    encoder.to(device)
    decoder.to(device)

  criterion = nn.MSELoss()
  optimizer_list = list()
  for encoder, decoder in multi_list:
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=opt.lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=opt.lr)
    optimizer_list.append([encoder_optimizer, decoder_optimizer])

  # multi have optimizer of [encoder, decoder, trend_encoder, volat_encoder]
  optimizer_list[-1].append(optimizer_list[0][0])
  optimizer_list[-1].append(optimizer_list[1][0])

  # define task forward
  def task_forward(task_id, multi_list, data_dict):
    if task_id == 0:  # trend task
      H = multi_list[0][0](data_dict['X_trend'])
      Ypred = multi_list[0][1](H, data_dict['Y_trend'])
      loss = criterion(Ypred.squeeze(), data_dict['Y_gt_trend'])
    elif task_id == 1:
      H = multi_list[1][0](data_dict['X_volat'])
      Ypred = multi_list[1][1](H, data_dict['Y_volat'])
      loss = criterion(Ypred.squeeze(), data_dict['Y_gt_volat'])
    else:
      H_trend = multi_list[0][0](data_dict['X_trend'])
      H_volat = multi_list[1][0](data_dict['X_volat'])
      X_multi = torch.cat([data_dict['X_trend'], H_trend, H_volat], -1)
      H = multi_list[2][0](X_multi)
      Ypred = multi_list[2][1](H, data_dict['Y_multi'])
      loss = criterion(Ypred.squeeze(), data_dict['Y_gt_multi'])

    return loss

  # Train Loops
  n_batches_count = 1
  epoch_batch_loss_list = list()
  for epoch in range(opt.epochs):
    batch_loss_list = list()
    for data_dict in train_loader:
      task_id = sample_task_id()

      # Prepare Data On Devices
      cuda_data_dict = dict()
      for k in data_dict:
        cuda_data_dict[k] = data_dict[k].type(torch.FloatTensor).to(device)

      optimizers = optimizer_list[task_id]
      for o in optimizers:
        o.zero_grad()

      # Forward Pass
      loss = task_forward(task_id, multi_list, cuda_data_dict)

      # Gradient Descent
      loss.backward()
      # todo: rescale_gradients()
      for o in optimizers:
        o.step()

      # Log Stats
      if n_batches_count % 100 == 0:
        writer.add_scalar('train/loss_%d' % task_id, loss.item(),
                          n_batches_count)
      if n_batches_count % 50000 == 0:
        for p in encoder_optimizer.param_groups:
          p['lr'] *= 0.9
        for p in decoder_optimizer.param_groups:
          p['lr'] *= 0.9
      n_batches_count += 1

    print(batch_loss_list)
    epoch_batch_loss_list.append(batch_loss_list)
