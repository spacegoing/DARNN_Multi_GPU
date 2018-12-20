# -*- coding: utf-8 -*-
'''
Generate MinMaxScaler normalized (-1,1) CSI300 Dataset (2015-2017).

CSI300 constitution stock and weight are according to csi300_20150130.csv

'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client['CSI300']
col = db['CSI300_Meta']
wcsi_path = '../../csi300/csi300_20150130.csv'
year_list = [2015, 2016, 2017]
csi_dir = [
    '/project/chli/scp/CSI300/Stk_1F_2015/',
    '/project/chli/scp/CSI300/Stk_1F_2016/',
    '/project/chli/scp/CSI300/Stk_1F_2017/'
]
norm_csi_dir = '/project/chli/scp/CSI300_NORM/'


def date_norm(x, stock_code, date_scaler_list):
  # for o,l,h,c use the same scaler
  price_arr = x.iloc[:, :4].values.reshape(-1, 1)
  price_scaler = MinMaxScaler((-1, 1))
  price_scaler.fit(price_arr)
  x.iloc[:, :4] = price_scaler.transform(price_arr).reshape(-1, 4)

  # for v,a use the same scaler
  va_scaler = MinMaxScaler((-1, 1))
  va_scaler.fit(x.iloc[:, -2:])
  x.iloc[:, -2:] = va_scaler.transform(x.iloc[:, -2:])
  date_scaler_list.append({
      'stock_code': stock_code,
      'date': pd.to_datetime(x.index[0].date()),
      'price_min': price_scaler.min_.item(),
      'price_scale': price_scaler.scale_.item(),
      'va_min': va_scaler.min_.tolist(),
      'va_scale': va_scaler.scale_.tolist(),
  })
  return x


def norm_stock(stock_code, wcsi_df, debug=False):
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
  date_scaler_list = list()
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
    tdf_norm = tdf.groupby(tdf.index.date).apply(
        date_norm, stock_code=stock_code, date_scaler_list=date_scaler_list)
    df_list.append(tdf_norm)
  df = pd.concat(df_list)
  df.index.name = 'datetime'
  col.insert_many(date_scaler_list)
  df.to_csv(norm_csi_dir + stock_code + '.csv')
  print(stock_code)
  return df


# Generate CSI300 Normalized Dataset
def gen_csi_norm():
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
  wcsi_df.con_code.apply(norm_stock, wcsi_df=wcsi_df)


# Aggregate stocks files to one giant file
def concat_stockcsvfiles_to_onecsi():
  df_list = []
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

  def read_stock(con_code):
    tdf = pd.read_csv(norm_csi_dir + con_code + '.csv', index_col='datetime')
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
    print(df.name, lag_steps, pred_steps)
    return stock_df

  train_df = tdf.groupby(level=0).apply(
      gen_stock_level, lag_steps=lag_steps, pred_steps=pred_steps)
  lag_columns = [
      i + '_%d' % j for j in reversed(range(lag_steps)) for i in tdf.columns
  ]
  lag_columns = lag_columns + ['c_gt']
  train_df.columns = lag_columns
  train_df.to_csv(norm_csi_dir +
                  'csi300norm_lag%d_pred%d.csv' % (lag_steps, pred_steps))


if __name__ == "__main__":
  # Generate Training Set
  # lag_pred = [(10, 1), (10, 2), (10, 5), (15, 1), (15, 2), (15, 5), (20, 1),
  #             (20, 2), (20, 5), (20, 10)]
  # lag_steps, pred_steps = lag_pred[0]
  # tdf = pd.read_csv(
  #     norm_csi_dir + 'csi300norm.csv',
  #     parse_dates=['datetime']).set_index(['con_code', 'datetime'])
  # # pdf = tdf.loc[['000156.SZ', '601318.SH']]
  # # gen_lag_pred_norm_csi300(pdf, lag_steps, pred_steps)
  # gen_lag_pred_norm_csi300(tdf, lag_steps, pred_steps)
  gen_csi_norm()
