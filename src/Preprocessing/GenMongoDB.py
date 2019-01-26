# -*- coding: utf-8 -*-
import pandas as pd
from pymongo import MongoClient

norm_csi_dir = '/project/chli/scp/CSI300_NORM/'
lag_pred = [(10, 1), (10, 2), (10, 5), (15, 1), (15, 2), (15, 5), (20, 1),
            (20, 2), (20, 5), (20, 10)]
lag_steps, pred_steps = lag_pred[2]

client = MongoClient('mongodb://localhost:27017/')
db = client['CSI300']
col = db['l10p5']

df = pd.read_csv(norm_csi_dir + 'csi300norm_lag%d_pred%d.csv' % (lag_steps, pred_steps), nrows=1e6)
df.columns = [i.replace('.', '_') if '.' in i else i for i in df.columns]
df.columns = ['con_code', 'datetime'] + [
    i + '_%d' % j
    for j in reversed(range(lag_steps))
    for i in ['o', 'h', 'l', 'c', 'v', 'a']
] + ['c_gt']

def insert(x):
  col.insert_many(x.to_dict(orient='record'))


df.groupby('con_code').apply(insert)
sdf = pd.DataFrame(list(col.find({}, {'_id': 0})))
# CPU times: user 34.7 s, sys: 4.59 s, total: 39.3 s
# Wall time: 43.7 s
