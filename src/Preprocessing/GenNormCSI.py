# -*- coding: utf-8 -*-
'''
Generate MinMaxScaler normalized (-1,1) CSI300 Dataset (2015-2017).

CSI300 constitution stock and weight are according to csi300_20150130.csv

'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

wcsi_path = '../../csi300/csi300_20150130.csv'
wcsi_df = pd.read_csv(
    wcsi_path,
    dtype={
        'date': np.str,
        'con_code': np.str,
        'weight': np.float,
        'stock_code': np.str,
        'mkt': np.str,
        'filename': np.str
    })
wcsi_df.index = wcsi_df['con_code']
year_list = [2015, 2016, 2017]
csi_dir = [
    '/project/chli/scp/CSI300/Stk_1F_2015/',
    '/project/chli/scp/CSI300/Stk_1F_2016/',
    '/project/chli/scp/CSI300/Stk_1F_2017/'
]

norm_csi_dir = '/project/chli/scp/CSI300_NORM/'


def date_norm(x):
  date_scaler = MinMaxScaler((-1, 1))
  date_scaler.fit(x)
  x[:] = date_scaler.transform(x)
  return x


def norm_stock(stock_code, debug=False):
  """
    Args:
        input_path (str): directory to nasdaq dataset.

    Returns:
        X (DataFrame): features.
        y (DataFrame): ground truth.
  stock_code = wcsi_df.index[-1]
  debug=False
  """
  df_list = []
  for fdir in csi_dir:
    tdf = pd.read_csv(
        fdir + wcsi_df.loc[stock_code, 'filename'],
        header=None,
        index_col=None,
        nrows=240 * 5 if debug else None)
    tdf.columns = ['date', 'time', 'o', 'h', 'l', 'c', 'v', 'a']
    tdf.index = pd.to_datetime(
        tdf['date'] + ' ' + tdf['time'], format="%Y%m%d %H:%M")
    tdf = tdf.iloc[:, 2:]
    tdf_norm = tdf.groupby(tdf.index.date).apply(date_norm)
    df_list.append(tdf_norm)
  df = pd.concat(df_list)
  df.index.name = 'datetime'
  df.to_csv(norm_csi_dir + stock_code + '.csv')
  print(stock_code)
  return df


# Generate CSI300 Normalized Dataset
wcsi_df.con_code.apply(norm_stock)

# Aggregate stocks files to one giant file
df_list = []


def read_stock(con_code):
  tdf = pd.read_csv(csi_dir + con_code + '.csv', index_col='datetime')
  df_list.append(tdf)


wcsi_df['con_code'].apply(read_stock)
df = pd.concat(df_list, keys=wcsi_df['con_code'])
df.to_csv(norm_csi_dir + 'csi300norm.csv')


# Generate shifted lagging dataset (grouped by date)
def gen_lag_pred_norm_csi300(tdf, lag_steps, pred_steps):

  def gen_date_level(date_df, lag_steps=0, pred_steps=0):
    # let lag_steps=10. left to right: from oldest
    # (9:30 shift(0)) to latest (9:39 shift(9))
    shift_list = [date_df.shift(i) for i in range(lag_steps - 1, -1, -1)]
    # generate feat_mat
    shifted_df = pd.concat(shift_list, axis=1)
    # generate target_arr
    shifted_df.loc[:, 'c_gt'] = date_df.loc[:, 'c'].shift(-pred_steps)
    shifted_df.dropna(inplace=True)
    return shifted_df

  def gen_stock_level(df, lag_steps=0, pred_steps=0):
    stock_df = df.groupby(df.index.get_level_values(1).date).apply(
        gen_date_level, lag_steps=lag_steps, pred_steps=pred_steps)
    # drop con_code [0] and date [1] index assigned by goupby
    stock_df.index = stock_df.index.droplevel([0, 1])
    return stock_df

  train_df = tdf.groupby(level=0).apply(
      gen_stock_level, lag_steps=lag_steps, pred_steps=pred_steps)
  lag_columns = [
      i + '_%d' % j for j in range(1, lag_steps) for i in tdf.columns
  ]
  lag_columns = list(tdf.columns) + lag_columns + ['c_gt']
  train_df.to_csv(norm_csi_dir +
                  'csi300norm_lag%d_pred%d.csv' % (lag_steps, pred_steps))


if __name__ == "__main__":
  lag_steps = 10
  pred_steps = 2
  tdf = pd.read_csv(
      norm_csi_dir + 'csi300norm.csv',
      parse_dates=['datetime']).set_index(['con_code', 'datetime'])
  # pdf = tdf.loc[['000156.SZ', '601318.SH']]
  # gen_lag_pred_norm_csi300(pdf, lag_steps, pred_steps)
  gen_lag_pred_norm_csi300(tdf, lag_steps, pred_steps)
