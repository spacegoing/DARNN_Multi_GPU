# -*- coding: utf-8 -*-
'''
Generate MinMaxScaler normalized (-1,1) CSI300 Dataset (2015-2017).

CSI300 constitution stock and weight are according to csi300_20150130.csv

'''
import pandas as pd
import numpy as np
import ta

wcsi_path = '../../csi300/csi300_20150130.csv'
year_list = [2015, 2016, 2017]
raw_csi_dir = [
    '/project/chli/scp/CSI300/Stk_1F_2015/',
    '/project/chli/scp/CSI300/Stk_1F_2016/',
    '/project/chli/scp/CSI300/Stk_1F_2017/'
]
proced_csi_dir = [
    '/project/chli/dataset_csi300/train/',
    '/project/chli/dataset_csi300/train_norm/',
    '/project/chli/dataset_csi300/valid/',
    '/project/chli/dataset_csi300/valid_norm/',
    '/project/chli/dataset_csi300/test/',
    '/project/chli/dataset_csi300/test_norm/'
]

dataset_split_ratio = [0.8, 0.1, 0.1]


# Calculate TA
def calc_ta(date_price_df):
  '''
  All hyper-parameters are default in ta package
  Longest lag_steps: 34

  date_price_df = pd.read_csv(
      './1date_df.csv', index_col='datetime', parse_dates=['datetime'])
  '''
  high = date_price_df['h']
  low = date_price_df['l']
  close = date_price_df['c']
  volume = date_price_df['v']
  date_price_ta_df = pd.concat([
      date_price_df,
      ta.momentum.ao(high, low, s=5, l=34, fillna=False),
      ta.momentum.money_flow_index(
          high, low, close, volume, n=14, fillna=False),
      ta.volume.chaikin_money_flow(
          high, low, close, volume, n=20, fillna=False),
      ta.volume.on_balance_volume_mean(close, volume, n=10, fillna=False),
      ta.volatility.bollinger_hband(close, n=20, ndev=2, fillna=False),
      ta.volatility.bollinger_lband(close, n=20, ndev=2, fillna=False),
      ta.trend.adx(high, low, close, n=14, fillna=False),
      ta.trend.macd(close, n_fast=12, n_slow=26, fillna=False)
  ],
                               axis=1)
  with pd.option_context('mode.use_inf_as_null', True):
    date_price_ta_df.dropna(inplace=True)
  return date_price_ta_df


def split_norm_save_perstock_df(df, con_code):
  '''
  df must not contain any NaNs
  df = pd.read_csv('000156.csv', parse_dates=['datetime'], index_col='datetime')
  con_code = '000156'
  '''
  split_index_ser = df.iloc[:, 0].groupby(df.index.date).count().cumsum()
  stocks_num = split_index_ser.shape[0]
  train_stocks_num = int(stocks_num * dataset_split_ratio[0])
  train_index = split_index_ser.iloc[train_stocks_num - 1]
  valid_stocks_num = int(stocks_num * sum(dataset_split_ratio[:2]))
  valid_index = split_index_ser.iloc[valid_stocks_num - 1]
  train_df = df.iloc[:train_index]
  valid_df = df.iloc[train_index:valid_index]
  test_df = df.iloc[valid_index:]

  def norm(train_df, valid_df, test_df):
    mean_ser = train_df.mean()
    std_ser = train_df.std()
    train_norm_df = (train_df - mean_ser) / std_ser
    valid_norm_df = (valid_df - mean_ser) / std_ser
    test_norm_df = (test_df - mean_ser) / std_ser
    mean_ser.name = 'mean'
    std_ser.name = 'std'
    stock_norm_meta_df = pd.concat([mean_ser, std_ser], axis=1)
    return train_norm_df, valid_norm_df, test_norm_df, stock_norm_meta_df

  train_norm_df, valid_norm_df, test_norm_df, stock_norm_meta_df = norm(
      train_df, valid_df, test_df)

  train_ta_df = train_df.groupby(
      train_df.index.date, group_keys=False).apply(calc_ta)
  valid_ta_df = valid_df.groupby(
      valid_df.index.date, group_keys=False).apply(calc_ta)
  test_ta_df = test_df.groupby(
      test_df.index.date, group_keys=False).apply(calc_ta)
  train_ta_norm_df, valid_ta_norm_df, test_ta_norm_df, stock_ta_norm_meta_df = norm(
      train_ta_df, valid_ta_df, test_ta_df)

  def save_proced_df(raw_df, norm_df, ta_df, ta_norm_df, stock_norm_meta_df,
                     stock_ta_norm_meta_df, con_code, mode):
    '''
    mode = train valid test
    '''
    dir_id = 0
    if mode == 'valid':
      dir_id = 2
    if mode == 'test':
      dir_id = 4
    raw_df.to_csv(proced_csi_dir[dir_id] + con_code + '.csv')
    ta_df.to_csv(proced_csi_dir[dir_id] + con_code + '_ta.csv')
    norm_df.to_csv(proced_csi_dir[dir_id + 1] + con_code + '_norm.csv')
    ta_norm_df.to_csv(proced_csi_dir[dir_id + 1] + con_code + '_ta_norm.csv')

    if mode == 'train':
      stock_norm_meta_df.to_csv(proced_csi_dir[dir_id + 1] + con_code +
                                '_norm_meta.csv')
      stock_ta_norm_meta_df.to_csv(proced_csi_dir[dir_id + 1] + con_code +
                                   '_ta_norm_meta.csv')

  save_proced_df(train_df, train_norm_df, train_ta_df, train_ta_norm_df,
                 stock_norm_meta_df, stock_ta_norm_meta_df, con_code, 'train')
  save_proced_df(valid_df, valid_norm_df, valid_ta_df, valid_ta_norm_df,
                 stock_norm_meta_df, stock_ta_norm_meta_df, con_code, 'valid')
  save_proced_df(test_df, test_norm_df, test_ta_df, test_ta_norm_df,
                 stock_norm_meta_df, stock_ta_norm_meta_df, con_code, 'test')


def norm_stock(ser, debug=False):
  """
    Args:
        input_path (str): directory to nasdaq dataset.

    Returns:
        X (DataFrame): features.
        y (DataFrame): ground truth.
  debug=False
  """
  date_time_parser = lambda x: pd.to_datetime(x, format="%Y%m%d %H:%M")

  df_list = []
  con_code = ser.loc['con_code']
  print('start ' + con_code)
  for fdir in raw_csi_dir:
    tdf = pd.read_csv(
        fdir + ser.loc['filename'],
        header=None,
        names=['date', 'time', 'o', 'h', 'l', 'c', 'v', 'a'],
        index_col=0,
        parse_dates=[['date', 'time']],
        date_parser=date_time_parser,
        nrows=240 * 5 if debug else None)
    df_list.append(tdf)
  df = pd.concat(df_list)
  df.index.name = 'datetime'
  split_norm_save_perstock_df(df, con_code)
  print('end ' + con_code)


# Generate CSI300 Normalized Dataset
def gen_csi_norm():
  wcsi_df = pd.read_csv(
      wcsi_path,
      dtype={
          'date': np.str,
          'con_code': np.str,
          'weight': np.float,
          'con_code': np.str,
          'mkt': np.str,
          'filename': np.str
      })
  wcsi_df.apply(norm_stock, axis=1)



if __name__ == '__main__':

  gen_csi_norm()
  # check data
  con_code = '603993.SH'
  dir_id = 0

  raw_df = pd.read_csv(proced_csi_dir[dir_id] + con_code + '.csv')
  ta_df = pd.read_csv(proced_csi_dir[dir_id] + con_code + '_ta.csv')
  norm_df = pd.read_csv(proced_csi_dir[dir_id + 1] + con_code + '_norm.csv')
  ta_norm_df = pd.read_csv(proced_csi_dir[dir_id + 1] + con_code +
                          '_ta_norm.csv')
  df_list = [raw_df, ta_df, norm_df, ta_norm_df]
  for i in df_list:
    print(i.shape)
    with pd.option_context('mode.use_inf_as_null', True):
      print(i.isna().sum())

  if dir_id == 0:
    norm_meta_df = pd.read_csv(proced_csi_dir[dir_id + 1] + con_code +
                              '_norm_meta.csv')
    ta_norm_meta_df = pd.read_csv(proced_csi_dir[dir_id + 1] + con_code +
                                  '_ta_norm_meta.csv')
