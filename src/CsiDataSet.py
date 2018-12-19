# -*- coding: utf-8 -*-
from torch.utils.data import Dataset, DataLoader
# todo: torch must be imported first, otherwise segmentation fault
# do not know why ver: 0.4.0
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

lag_pred = [(10, 1), (10, 2), (10, 5), (15, 1), (15, 2), (15, 5), (20, 1),
            (20, 2), (20, 5), (20, 10)]
lag_steps, pred_steps = lag_pred[2]


class Trainset(Dataset):

  def __init__(self, csi_df, lag_steps):
    super().__init__()

    self.csi_df = csi_df

    self.x_columns = [
        i + '_%d' % j
        for j in reversed(range(lag_steps))
        for i in ['o', 'h', 'l', 'v', 'a']
    ]
    self.y_columns = ['c_%d' % i for i in reversed(range(lag_steps))]
    self.lag_steps = lag_steps

  def __getitem__(self, idx):
    d = {
        'X': self.get_x_mat(idx),
        'Y': self.get_y_mat(idx),
        'Y_gt': self.csi_df.iloc[idx]['c_gt'],
        'idx': idx
    }
    return d

  def __len__(self):
    return self.csi_df.shape[0]

  def get_x_mat(self, idx):
    '''
    reshape df (1, feat_dim * timesteps) -> (timesteps, feat_dim)
    '''
    return self.csi_df.iloc[idx].loc[self.x_columns].values.reshape(
        self.lag_steps, -1).astype(np.float)

  def get_y_mat(self, idx):
    '''
    reshape df (1, feat_dim * timesteps) -> (timesteps, feat_dim)
    '''
    return self.csi_df.iloc[idx].loc[self.y_columns].values.reshape(
        self.lag_steps, -1).astype(np.float)

  def get_idx_df_label(self, idx):
    return self.csi_df.iloc[idx][['con_code', 'datetime']]


class CSI300Dataset:

  def __init__(self, opt):
    csi_df = pd.read_csv(
        opt.norm_csi_dir +
        'csi300norm_lag%d_pred%d.csv.old' % (opt.lag_steps, opt.pred_steps),
        nrows=1e6 if opt.debug else None)
    # todo: temp columns fix
    tmp_columns = ['con_code', 'datetime'] + [
        i + '_%d' % j
        for j in reversed(range(lag_steps))
        for i in ['o', 'h', 'l', 'c', 'v', 'a']
    ] + ['c_gt']
    csi_df.columns = tmp_columns
    self.csi_df = csi_df

  def get_dataset_loader(self, opt):
    # timesteps = 9  # t, t-1, ... t-8
    # pred_timesteps = 1
    # batchsize = 128
    lag_steps = opt.lag_steps
    batchsize = opt.batchsize
    shuffle = opt.shuffle
    num_workers = opt.num_workers
    pin_memory = opt.pin_memory
    dataset_split_ratio = opt.dataset_split_ratio

    # split train valid test dataset
    split_index_ser = self.csi_df.groupby(
        'con_code')['datetime'].count().cumsum()
    stocks_num = split_index_ser.shape[0]
    train_index = split_index_ser.iloc[int(stocks_num * dataset_split_ratio[0])]
    valid_index = split_index_ser.iloc[int(
        stocks_num * sum(dataset_split_ratio[:2]))]
    train_df = self.csi_df.iloc[:train_index]
    valid_df = self.csi_df.iloc[train_index:valid_index]
    test_df = self.csi_df.iloc[valid_index:]

    train_dataset, valid_dataset, test_dataset = Trainset(
        train_df, lag_steps), Trainset(valid_df, lag_steps), Trainset(
            test_df, lag_steps)
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
  csi300 = CSI300Dataset(opt)
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
