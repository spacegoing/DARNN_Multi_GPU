# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import multiprocessing as mp

if __name__ == "__main__":
  mp.freeze_support()

  wcsi_path = '../../csi300/csi300_20150130.csv'
  proced_csi_dir = [
      '/project/chli/dataset_csi300/train_norm/',
      '/project/chli/dataset_csi300/valid_norm/',
      '/project/chli/dataset_csi300/test_norm/'
  ]
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

  def para_fn(args):
    path, con_code, suffix = args
    df = pd.read_csv(path + con_code + suffix, index_col='datetime')
    return df

  def concat_csv(path):

    pool = mp.Pool(10)

    # concat norm csv
    stock_list = list()
    for i, row in wcsi_df.iterrows():
      stock_list.append((path, row['con_code'], '_norm.csv'))

    all_df_list = pool.map(para_fn, stock_list)
    all_df = pd.concat(all_df_list, keys=wcsi_df['con_code'])
    all_df.to_csv(path + 'csi300_norm.csv')

    # check identity
    # path = proced_csi_dir[0]
    # all_df = pd.read_csv(path + 'csi300_norm.csv', index_col='con_code')
    # g = all_df.groupby(all_df.index)
    # for con_code in wcsi_df['con_code'].iloc[:9]:
    #   df = pd.read_csv(path + con_code + '_norm.csv')
    #   gdf = g.get_group(con_code)
    #   res = df.values - gdf.values
    #   print(res.sum().sum())
    #   print(res.shape[0] * res.shape[1])

    # concat ta norm csv
    stock_list = list()
    for i, row in wcsi_df.iterrows():
      stock_list.append((path, row['con_code'], '_ta_norm.csv'))
    all_df_list = pool.map(para_fn, stock_list)
    all_df = pd.concat(all_df_list, keys=wcsi_df['con_code'])
    all_df.to_csv(path + 'csi300_ta_norm.csv')

  for path in proced_csi_dir:
    concat_csv(path)
