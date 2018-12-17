# -*- coding: utf-8 -*-
import pandas as pd
import tushare as ts
import os

pro = ts.pro_api('2fd23cc108c16c878a67e2054dbd22fc069013f8f034224e56a5ac7a')
df = pro.index_weight(
    index_code='399300.SZ', start_date='20150101', end_date='20150201')
df.rename(columns={'trade_date': 'date'}, inplace=True)
df.index = pd.to_datetime(df['date'])

csi = df.loc['2015-01-30'].sort_values('weight')[['con_code', 'weight']]


def split_code_gen_filename(x):
  p_idx = x.find('.')
  code = x[:p_idx]
  mkt = x[p_idx + 1:]
  filename = mkt + code + '.csv'
  return pd.Series({'stock_code': code, 'mkt': mkt, 'filename': filename})


csi[['stock_code', 'mkt',
     'filename']] = csi['con_code'].apply(split_code_gen_filename)

csi_dir = [
    '/project/chli/scp/CSI300/Stk_1F_2015/',
    '/project/chli/scp/CSI300/Stk_1F_2016/',
    '/project/chli/scp/CSI300/Stk_1F_2017/'
]

mask = []
for c in csi['filename']:
  flag = True
  for p in csi_dir:
    if not os.path.isfile(p + c):
      flag = False
  mask.append(flag)
csi = csi[mask]
csi.to_csv('../../csi300/csi300_20150130.csv')
