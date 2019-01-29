# -*- coding: utf-8 -*-
from torch.utils.data import Dataset, DataLoader
# todo: torch must be imported first, otherwise segmentation fault
# do not know why ver: 0.4.0
import pandas as pd
from LagShiftUtils import DateRolling


class Trainset(Dataset):
  '''
  Only for single task, csi300 with or without indicator usage
  '''

  def __init__(self, csi_df, opt):
    super().__init__()

    self.csi_df = csi_df
    self.roller = DateRolling(csi_df, opt.lag_steps, opt.pred_steps)
    self.opt = opt

  def __getitem__(self, idx):
    '''
    Y and X have to be same shape. so ['c'] rather than 'c'
    '''
    sample = self.roller.get_sample_by_index(idx)
    return {
        'X':
            sample.loc[:, sample.columns.difference(self.opt.y_columns)]
            .values[:self.opt.lag_steps],
        'Y':
            sample.loc[:, self.opt.y_columns].values[:self.opt.lag_steps],
        'Y_gt':
            sample.loc[:, self.opt.y_columns].iloc[-1].item(),
        'idx':
            idx
    }

  def __len__(self):
    return self.roller.num_samples

  def get_idx_df(self, idx):
    return self.roller.get_sample_by_index(idx)


class CSI300Dataset:

  def get_dataset_loader(self, opt):
    proced_csi_dir = [
        '/project/chli/dataset_csi300/train_norm/',
        '/project/chli/dataset_csi300/valid_norm/',
        '/project/chli/dataset_csi300/test_norm/'
    ]
    train_df, valid_df, test_df = [
        pd.read_csv(
            i + 'csi300_ta_norm.csv',
            nrows=1e6 if opt.debug else None,
            parse_dates=['datetime'],
            index_col=['con_code', 'datetime']) for i in proced_csi_dir
    ]

    # timesteps = 9  # t, t-1, ... t-8
    # pred_timesteps = 1
    # batchsize = 128
    batchsize = opt.batchsize
    shuffle = opt.shuffle
    num_workers = opt.num_workers
    pin_memory = opt.pin_memory

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


class MultiTaskTrainset(Dataset):
  '''
  input df: norm_csi (origin input without indicators)
  task_type: 'multi'
  '''

  def __init__(self, csi_df, opt):
    super().__init__()

    self.csi_df = csi_df
    self.roller = DateRolling(csi_df, opt.lag_steps, opt.pred_steps,
                              opt.ind_steps)
    self.opt = opt

  def __getitem__(self, idx):
    '''
    X will be [batchsize, lag_steps, feat_dim] 3 dim
    Y will be [batchsize, lag_steps] 2 dim
    '''
    sample = self.roller.get_std_sample_by_index(idx)
    return {
        'X_trend':
            sample
            .loc[:, sample.columns.difference(['c', 'std', 'multi_target'])]
            .values[:self.opt.lag_steps],
        'Y_trend':
            sample.loc[:, 'c'].values[:self.opt.lag_steps],
        'Y_gt_trend':
            sample.loc[:, 'c'].iloc[-1].item(),
        'X_volat':
            sample
            .loc[:, sample.columns.difference(['c', 'std', 'multi_target'])]
            .values[:self.opt.lag_steps],
        'Y_volat':
            sample.loc[:, 'std'].values[:self.opt.lag_steps],
        'Y_gt_volat':
            sample.loc[:, 'std'].iloc[-1].item(),
        'Y_multi':
            sample.loc[:, 'multi_target'].values[:self.opt.lag_steps],
        'Y_gt_multi':
            sample.loc[:, 'multi_target'].iloc[-1].item(),
        'idx':
            idx
    }

  def __len__(self):
    return self.roller.num_samples

  def get_idx_df(self, idx):
    return self.roller.get_std_sample_by_index(idx)


class CSI300MultiTaskDataset:

  def get_dataset_loader(self, opt):
    proced_csi_dir = [
        '/project/chli/dataset_csi300/train_norm/',
        '/project/chli/dataset_csi300/valid_norm/',
        '/project/chli/dataset_csi300/test_norm/'
    ]
    train_df, valid_df, test_df = [
        pd.read_csv(
            i + 'csi300_norm.csv',
            nrows=1e6 if opt.debug else None,
            parse_dates=['datetime'],
            index_col=['con_code', 'datetime']) for i in proced_csi_dir
    ]

    # timesteps = 9  # t, t-1, ... t-8
    # pred_timesteps = 1
    # batchsize = 128
    batchsize = opt.batchsize
    shuffle = opt.shuffle
    num_workers = opt.num_workers
    pin_memory = opt.pin_memory

    train_dataset, valid_dataset, test_dataset = MultiTaskTrainset(
        train_df, opt), MultiTaskTrainset(valid_df, opt), MultiTaskTrainset(
            test_df, opt)
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

  opt.debug = False
  opt.ind_steps = 10
  # single task ind_steps must be 0
  opt.task_type = 'multi'  # 'multi' 'single'
  opt.pred_type = 'shift'  # 'shift' 'steps'

  csi300 = CSI300Dataset()
  train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader = csi300.get_dataset_loader(
      opt)
  for i, d in enumerate(train_loader):
    if i % 100 == 0 and i != 0:
      break
  for k, v in d.items():
    print(k)
    print(v.shape)

  multi_csi300 = CSI300MultiTaskDataset()
  mtrain_dataset, mvalid_dataset, mtest_dataset, mtrain_loader, mvalid_loader, mtest_loader = multi_csi300.get_dataset_loader(
      opt)
  for i, d in enumerate(mtrain_loader):
    if i % 100 == 0 and i != 0:
      break
  for k, v in d.items():
    print(k)
    print(v.shape)

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
