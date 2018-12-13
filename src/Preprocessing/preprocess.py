# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy.stats import zscore
import pandas as pd

raw_dir = '/project/chli/scp/CSI300/Stk_1F_%d/'
year_list = [2015, 2016, 2017]


def zf(x):
  ''' compute zscore by date by column
  '''
  return x.apply(zscore)


def check_nulldupoutlier(year, f_name):
  df = pd.read_csv(raw_dir % year + f_name, header=None, index_col=None)
  df.columns = ['date', 'time', 'o', 'h', 'l', 'c', 'v', 'a']
  df.index = pd.to_datetime(
      df['date'] + ' ' + df['time'], format="%Y%m%d %H:%M")
  df = df.dropna()
  df = df.drop_duplicates()

  # check daily trades amount equality
  daily_count = df.groupby('date').count()
  # all days should have same number of trades in a day
  mask = daily_count == daily_count.iloc[0, 0]
  assert mask.values.sum() == mask.shape[0] * mask.shape[1]

  # check singularity
  # v, a have too large volation, not check for now
  df = df.loc[:, ['o', 'h', 'l', 'c']]

  z_df = df.groupby(df.index.date).apply(zf)
  assert (z_df > 10).values.sum() == 0


if __name__ == "__main__":
  for year in year_list:
    f_name_list = os.listdir(raw_dir % year)
    for f in f_name_list:
      # year = year_list[0]
      # f_name = 'SH600000.csv'
      check_nulldupoutlier(year, f)
