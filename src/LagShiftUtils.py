# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


class DateRolling:

  def __init__(self, df, lag_steps, pred_steps):
    '''
    df: multi-level index dataframe
      level0: con_code
      level1: datetime
      values: feature column
    '''
    self.df = df
    self.lag_steps = lag_steps
    self.pred_steps = pred_steps
    self.win_len = lag_steps + pred_steps
    self.be_index_arr = self.get_be_index_bydate()
    self.num_samples = self.be_index_arr.shape[0]

  def get_be_index_bydate(self):
    '''
    Return:
    np.array: array of oldest lag_step's row index in df
    for each rolling window
    '''

    # per stock per date: how many minutes it contains
    # multiindex (con_code, date)
    stock_date_count_ser = self.df.groupby(
        [df.index.get_level_values(0),
         df.index.get_level_values(1).date]).size()
    stock_date_count_ser.name = 'count'

    # per stock per date: the first minute's index in `df`
    # multiindex (con_code, date)
    date_begin_idx_ser = stock_date_count_ser.cumsum() - stock_date_count_ser
    date_begin_idx_ser.name = 'date_be'

    # multiindex (con_code, date): count, date_be
    index_df = pd.concat([stock_date_count_ser, date_begin_idx_ser], axis=1)
    index_df.index.set_names('date', level=1, inplace=True)

    def calc_index_arr(index_df):
      idx_list = list()
      for idx, (count, be) in index_df.iterrows():
        date_num = count - self.win_len + 1
        # check whether win_len greater than date's num of samples
        if date_num < 0:
          raise ValueError(
              str(idx) + ' has %d samples, ' % count +
              'less than window length (lag_steps + pred_steps) %d' %
              self.win_len)
        # main function: generate oldest lag_step's row index in df
        idx_arr = np.arange(be, be + date_num)
        idx_list.append(idx_arr)
      return np.concatenate(idx_list)

    be_index_arr = calc_index_arr(index_df)
    return be_index_arr

  def get_sample_by_index(self, idx):
    be_idx = self.be_index_arr[idx]
    return {
        'X': self.df.iloc[be_idx:be_idx + self.lag_steps],
        'Y': self.df.iloc[be_idx + self.win_len - 1]
    }


if __name__ == '__main__':
  norm_csi_dir = '/project/chli/scp/CSI300_NORM/'
  debug = True

  df = pd.read_csv(
      norm_csi_dir + 'csi300norm.csv',
      nrows=1e7 if debug else None,
      parse_dates=['datetime'],
      index_col=['con_code', 'datetime'])

  lag_steps = 10
  pred_steps = 5

  # df.to_csv(norm_csi_dir + 'csi300norm.csv.sorted')
  df_roller = DateRolling(df, lag_steps, pred_steps)
