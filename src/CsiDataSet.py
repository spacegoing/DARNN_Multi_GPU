# -*- coding: utf-8 -*-
from torch.utils.data import Dataset, DataLoader
# todo: torch must be imported first, otherwise segmentation fault
# do not know why ver: 0.4.0
import numpy as np
import pandas as pd
from LagShiftUtils import DateRolling


class Trainset(Dataset):

  def __init__(self, csi_df, opt):
    super().__init__()

    self.csi_df = csi_df
    self.roller = DateRolling(csi_df, opt.lag_steps, opt.pred_steps)
    self.opt = opt

  def __getitem__(self, idx):
    sample = self.roller.get_sample_by_index(idx)
    return {
        'X': sample['X'].loc[:, self.opt.x_columns].values,
        'Y': sample['X'].loc[:, self.opt.y_columns].values,
        'Y_gt': sample['Y'].loc[self.opt.y_columns].item(),
        'idx': idx
    }

  def __len__(self):
    return self.roller.num_samples

  def get_idx_df(self, idx):
    return self.roller.get_sample_by_index(idx)


class CSI300Dataset:

  def __init__(self, opt):
    self.csi_df = pd.read_csv(
        opt.norm_csi_dir + 'csi300norm.csv',
        nrows=1e6 if opt.debug else None,
        parse_dates=['datetime'],
        index_col=['con_code', 'datetime'])

  def get_dataset_loader(self, opt):
    # timesteps = 9  # t, t-1, ... t-8
    # pred_timesteps = 1
    # batchsize = 128
    batchsize = opt.batchsize
    shuffle = opt.shuffle
    num_workers = opt.num_workers
    pin_memory = opt.pin_memory
    dataset_split_ratio = opt.dataset_split_ratio

    # split train valid test dataset by number of stocks
    split_index_ser = self.csi_df.iloc[:, 0].groupby(
        level='con_code').count().cumsum()
    stocks_num = split_index_ser.shape[0]
    train_stocks_num = int(stocks_num * dataset_split_ratio[0])
    train_index = split_index_ser.iloc[train_stocks_num - 1]
    valid_stocks_num = int(stocks_num * sum(dataset_split_ratio[:2]))
    valid_index = split_index_ser.iloc[valid_stocks_num - 1]
    train_df = self.csi_df.iloc[:train_index]
    valid_df = self.csi_df.iloc[train_index:valid_index]
    test_df = self.csi_df.iloc[valid_index:]

    train_dataset, valid_dataset, test_dataset = Trainset(
        train_df, opt), Trainset(valid_df, opt), Trainset(test_df, opt)
    train_loader, valid_loader, test_loader = [
        DataLoader(
            i,
            batch_size=batchsize,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory)
        for i in [train_dataset, valid_dataset, test_dataset]
    ]
    return train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader


if __name__ == "__main__":
  from main import opt

  opt.debug = True
  csi300 = CSI300Dataset(opt)
  # for debug
  # csi_df = pd.read_csv(
  #     opt.norm_csi_dir + 'csi300norm.csv',
  #     nrows=1e6 if opt.debug else None,
  #     parse_dates=['datetime'],
  #     index_col=['con_code', 'datetime'])
  # csi300 = CSI300Dataset(csi_df, opt)

  train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader = csi300.get_dataset_loader(
      opt)
  for i, d in enumerate(train_loader):
    if i % 100 == 0 and i != 0:
      break

  # # double check
  # i=-1
  # aa=pre_dataset.X_train_df.iloc[i].values.reshape(timesteps,-1)
  # print((aa==d['X'][i,:,:]).sum())

  # aaa=pre_dataset.X_scaler.inverse_transform(aa)
  # print((aaa[0] == X.iloc[0]).sum())
  # neq = np.where((aaa[0] == X.iloc[0])==False)[0]
  # for i in neq:
  #   print("%f %f" % (aaa[0,i], X.iloc[0,i]))

  # opt.timesteps = 9
  # opt.pred_timesteps = 1
  # opt.batchsize = 128
  # opt.shuffle = False
  # opt.num_workers = 1
  # opt.pin_memory = True
