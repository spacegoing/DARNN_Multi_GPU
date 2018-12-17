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
