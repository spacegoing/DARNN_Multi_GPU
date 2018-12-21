# -*- coding: utf-8 -*-
class Version:

  def __init__(self):
    lag_pred = [(10, 1), (10, 2), (10, 5), (15, 1), (15, 2), (15, 5), (20, 1),
                (20, 2), (20, 5), (20, 10)]
    # cannot have hid_dim and (hid_dim_decoder, hid_dim_encoder)
    # at the same time
    self.version_dict = {
        1: {
            # (15, 5) 128
            'lag_steps': lag_pred[5][0],
            'pred_steps': lag_pred[5][1],
            'hid_dim': 128
        },
        2: {
            # (15, 5) 64
            'lag_steps': lag_pred[5][0],
            'pred_steps': lag_pred[5][1],
            'hid_dim': 64
        },
        3: {
            # (20, 5) 128
            'lag_steps': lag_pred[8][0],
            'pred_steps': lag_pred[8][1],
            'hid_dim': 128
        },
        4: {
            # (15, 2) 64
            'lag_steps': lag_pred[4][0],
            'pred_steps': lag_pred[4][1],
            'hid_dim': 64
        },
        5: {
            # (10, 1) 64
            'lag_steps': lag_pred[0][0],
            'pred_steps': lag_pred[0][1],
            'hid_dim': 64
        }
    }

  def set_ver_opt(self, idx, opt):
    '''
    idx = 0 : use default version
    '''
    opt_dict = vars(opt)
    if idx:
      param_dict = self.version_dict[idx]
      for k in param_dict:
        if k == 'hid_dim':
          opt_dict['hid_dim_decoder'] = param_dict[k]
          opt_dict['hid_dim_encoder'] = param_dict[k]
        opt_dict[k] = param_dict[k]
