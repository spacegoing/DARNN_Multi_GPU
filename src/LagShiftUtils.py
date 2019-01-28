# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


class DateRolling:

  def __init__(self, df, lag_steps, pred_steps, ind_steps=0):
    '''
    df: multi-level index dataframe
      level0: con_code
      level1: datetime
      values: feature column
    ind_steps: window length of calculating indicator (e.g. MACD-10)
    lag_steps: window length for LSTM
    pred_steps: predictive window length for LSTM
    '''
    self.df = df
    self.ind_steps = ind_steps
    self.lag_steps = lag_steps
    self.pred_steps = pred_steps
    self.win_len = lag_steps + pred_steps
    # ind_win_len measures how many items (raw inputs) in
    # data. For example,
    # ind_steps already put ind_steps (say 3) items in it
    # so there are 1 ind in window.
    # let lag_steps be 4, there needs to be 4 inds,
    # namely 3 more inds, so there needs to be 3 more
    # raw items in window. together it would be
    # 3 + 4 -1 = 6 items + pred_steps
    self.ind_win_len = ind_steps + (lag_steps - 1) + pred_steps

    # per stock per date: how many minutes it contains
    # multiindex (con_code, date)
    stock_date_count_ser = self.df.iloc[:, 0].groupby(
        [df.index.get_level_values(0),
         df.index.get_level_values(1).date]).count()
    stock_date_count_ser.name = 'count'

    # per stock per date: the first minute's index in `df`
    # multiindex (con_code, date)
    date_begin_idx_ser = stock_date_count_ser.cumsum() - stock_date_count_ser
    date_begin_idx_ser.name = 'date_be'

    # multiindex (con_code, date): count, date_be
    index_df = pd.concat([stock_date_count_ser, date_begin_idx_ser], axis=1)
    index_df.index.set_names('date', level=1, inplace=True)
    self.index_df = index_df

    # self.be_index_arr is np.ndarray
    self.be_index_arr = self.get_be_index_bydate(index_df, ind_steps)
    self.num_samples = self.be_index_arr.shape[0]

  def get_be_index_bydate(self, index_df, ind_steps=0):
    '''
    Return:
    np.array: array of oldest lag_step's row index in df
    for each rolling window
    '''

    idx_list = list()
    for idx, (count, be) in index_df.iterrows():
      if ind_steps:
        # how many valid items in date have sufficient length
        # (win_len) till the end of the array (including the
        # beginning)
        date_num = count - self.ind_win_len + 1
      else:
        date_num = count - self.win_len + 1
      # check whether win_len greater than date's num of samples
      if date_num < 0:
        if ind_steps:
          raise ValueError(
            str(idx) + ' has %d samples, ' % count +
            'less than window length (ind_steps + lag_steps-1 + pred_steps) %d'
            % self.ind_win_len)
        else:
          raise ValueError(
              str(idx) + ' has %d samples, ' % count +
              'less than window length (lag_steps + pred_steps) %d' % self.win_len
          )
      # main function: generate oldest lag_step's row index in df
      idx_arr = np.arange(be, be + date_num)
      idx_list.append(idx_arr)

    be_index_arr = np.concatenate(idx_list)
    return be_index_arr

  def get_sample_by_index(self, idx):
    be_idx = self.be_index_arr[idx]
    return self.df.iloc[be_idx:be_idx + self.win_len]

  def get_std_sample_by_index(self, idx):
    be_idx = self.be_index_arr[idx]
    raw_df = self.df.iloc[be_idx:be_idx + self.ind_win_len]
    std = raw_df['c'].rolling(self.ind_steps).std()
    norm_std = (std - std.mean())/std.std()
    ind_df = raw_df.assign(std=norm_std)
    return ind_df.iloc[self.ind_steps - 1:]


if __name__ == '__main__':
  debug = False

  proced_csi_dir = [
      '/project/chli/dataset_csi300/train_norm/',
      '/project/chli/dataset_csi300/valid_norm/',
      '/project/chli/dataset_csi300/test_norm/'
  ]

  path = proced_csi_dir[0]

  df = pd.read_csv(
      path + 'csi300_norm.csv',
      nrows=1e7 if debug else None,
      parse_dates=['datetime'],
      index_col=['con_code', 'datetime'])

  # ta_df = pd.read_csv(
  #     path + 'csi300_ta_norm.csv',
  #     nrows=1e7 if debug else None,
  #     parse_dates=['datetime'],
  #     index_col=['con_code', 'datetime'])

  lag_steps = 20
  pred_steps = 5
  ind_steps = 10

  # df.to_csv(norm_csi_dir + 'csi300norm.csv.sorted')
  df_roller = DateRolling(df, lag_steps, pred_steps)
  ind_df_roller = DateRolling(df, lag_steps, pred_steps, ind_steps)
  print(df_roller.get_sample_by_index(0))
  print(ind_df_roller.get_std_sample_by_index(0))
  print(ind_df_roller.get_std_sample_by_index(206))
  print(ind_df_roller.get_std_sample_by_index(207))

  # # # Dev Multi Indicators
  # # stock_date_g  = df.groupby(
  # #     [df.index.get_level_values(0),
  # #      df.index.get_level_values(1).date])
  # # for key, df in stock_date_g:
  # #   break

  # # Dev Multi Indicators
  # index_df = pd.read_csv('test_multi_indicator.csv')
  # index_df.columns.values[1] = 'date'
  # index_df.set_index(['con_code', 'date'], inplace=True)

  # win_len = lag_steps + pred_steps

  # def df_apply(x):

  #   def date_fn(x):
  #     count = x['count']
  #     be = x['date_be']
  #     # todo: add self
  #     date_num = count - win_len + 1
  #     # check whether win_len greater than date's num of samples
  #     if date_num < 0:
  #       raise ValueError(
  #           str(x.name) + ' has %d samples, ' % count +
  #           'less than window length (lag_steps + pred_steps) %d' % self.win_len
  #       )
  #     # main function: generate oldest lag_step's row index in df
  #     idx_arr = np.arange(be, be + date_num)
  #     return pd.Series(idx_arr)

  #   # date_df is MultiIndexed by (con_code, date, integer)
  #   date_df = x.apply(date_fn, axis=1).stack()
  #   # remove con_code (added automatically by outer apply fn)
  #   # and integer index
  #   date_df.index = date_df.index.droplevel([0, 2])
  #   return date_df

  # # multi-index (con_code, date): be_index
  # be_index_arr = index_df.groupby(level=0).apply(df_apply)

  # idx_list = list()
  # for idx, (count, be) in index_df.iterrows():
  #   date_num = count - win_len + 1
  #   # check whether win_len greater than date's num of samples
  #   if date_num < 0:
  #     raise ValueError(
  #         str(idx) + ' has %d samples, ' % count +
  #         'less than window length (lag_steps + pred_steps) %d' % self.win_len)
  #   # main function: generate oldest lag_step's row index in df
  #   idx_arr = np.arange(be, be + date_num)
  #   idx_list.append(idx_arr)
