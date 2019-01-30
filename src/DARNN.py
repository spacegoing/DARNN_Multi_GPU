# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F

DEBUG = False
test_dir = '/home/chli4934/UsydCodeLab/phd/MrfEvent/OldCode/src/test/'


class Encoder(nn.Module):
  """encoder in DARNN.
  All tensors are created by [Tensor].new_* , so new tensors
  are on same device as [Tensor]. No need for `device` to be
  passed
  """

  def __init__(self, timesteps, feat_dim, hid_dim):
    """Initialize an encoder in DA_RNN."""
    super(Encoder, self).__init__()
    self.hid_dim = hid_dim
    self.feat_dim = feat_dim
    # todo: timesteps = T-1 in zhen code
    self.timesteps = timesteps

    # Fig 1. Temporal Attention Mechanism: Encoder is LSTM
    self.lstm = nn.LSTM(
        input_size=self.feat_dim, hidden_size=self.hid_dim, batch_first=True)

    # Construct Input Attention Mechanism via deterministic attention model
    # Eq. 8: W_e[h_{t-1}; s_{t-1}] + U_e * x^k
    self.attn = nn.Sequential(
        nn.Linear(2 * hid_dim + timesteps, feat_dim), nn.Tanh(),
        nn.Linear(feat_dim, 1))
    self.count = 0

  def forward(self, X):
    """forward.

        Args:
            X: (batchsize, timesteps, feat_dim)

        """
    # Use zeros_like so encoder_out and X have same dtype, device and layout
    # (batchsize, timesteps, hid_dim)
    encoder_out = X.new_zeros(X.shape[0], X.shape[1], self.hid_dim)

    # Eq. 8, parameters not in nn.Linear but to be learnt
    # v_e = torch.nn.Parameter(data=torch.empty(
    #     self.feat_dim, self.timesteps).uniform_(0, 1), requires_grad=True)
    # U_e = torch.nn.Parameter(data=torch.empty(
    #     self.timesteps, self.timesteps).uniform_(0, 1), requires_grad=True)

    # hidden, cell: initial states with dimention hidden_size
    h = self._init_state(X)
    s = self._init_state(X)

    for t in range(self.timesteps):
      # (batch_size, feat_dim, (2*hidden_size + timesteps))
      # tensor.expand: do not copy data; -1 means no changes at that dim
      x = torch.cat((h.expand(self.feat_dim, -1, -1).permute(1, 0, 2),
                     s.expand(self.feat_dim, -1, -1).permute(1, 0, 2),
                     X.permute(0, 2, 1)),
                    dim=2)
      # (batch_size, feat_dim, 1)
      e = self.attn(x)

      # get weights by softmax
      # (batch_size, feat_dim)
      alpha = F.softmax(e.squeeze(), dim=1)

      # todo debug
      self.count += 1
      if DEBUG and False and self.count == 3:
        # test equalty
        import ipdb
        ipdb.set_trace(context=7)
        np.save(test_dir + 'new_alpha', alpha.detach().numpy())

      # get new input for LSTM
      x_tilde = torch.mul(alpha, X[:, t, :])

      # encoder LSTM
      self.lstm.flatten_parameters()
      # self.lstm has batch_first=True flag
      # x_tilde -> (batchsize, 1, feat_dim)
      _, final_state = self.lstm(x_tilde.unsqueeze(1), (h, s))
      h = final_state[0]  # (1, batchsize, hidden)
      s = final_state[1]  # (1, batchsize, hidden)
      encoder_out[:, t, :] = h

    return encoder_out

  def _init_state(self, X):
    batchsize = X.shape[0]
    # same dtype, device as X
    init_state = X.new_zeros([1, batchsize, self.hid_dim])
    return init_state


class Decoder(nn.Module):
  """decoder in DA_RNN."""

  def __init__(self, timesteps, feat_dim, hid_dim):
    """Initialize a decoder in DA_RNN.
    feat_dim: encoder hidden state dim
    """
    super(Decoder, self).__init__()
    self.hid_dim = hid_dim
    # todo: timesteps = T-1 in zhen code
    self.timesteps = timesteps

    self.attn = nn.Sequential(
        nn.Linear(2 * hid_dim + feat_dim, feat_dim), nn.Tanh(),
        nn.Linear(feat_dim, 1))
    self.lstm = nn.LSTM(input_size=1, hidden_size=hid_dim, batch_first=True)
    self.fc = nn.Linear(feat_dim + 1, 1)
    self.fc_final = nn.Linear(hid_dim + feat_dim, 1)

    # todo remove
    self.count = 0

  def forward(self, H, Y):
    """forward."""
    d_n = self._init_state(H)
    c_n = self._init_state(H)

    # todo debug
    self.count += 1
    for t in range(self.timesteps):

      # (batchsize, timesteps, 2*timesteps + feat_dim)
      x = torch.cat((d_n.expand(self.timesteps, -1, -1).permute(1, 0, 2),
                     c_n.expand(self.timesteps, -1, -1).permute(1, 0, 2), H),
                    dim=2)

      # (batchsize, timesteps)
      beta = F.softmax(self.attn(x).squeeze(), dim=1)
      # Eqn. 14: compute context vector
      # (batchsize, feat_dim)
      context = torch.bmm(beta.unsqueeze(1), H).squeeze()
      # Eqn. 15
      # batch_size * 1
      y_tilde = self.fc(torch.cat((context, Y[:, t].unsqueeze(1)), dim=1))

      # todo debug
      if DEBUG and True and self.count == 3:
        # test equalty
        np.save(test_dir + 'new_ytilde', y_tilde.detach().numpy())
        import ipdb
        ipdb.set_trace(context=7)

      # Eqn. 16: LSTM
      self.lstm.flatten_parameters()
      _, final_states = self.lstm(y_tilde.unsqueeze(1), (d_n, c_n))
      # 1 * batch_size * hid_dim
      d_n = final_states[0]
      # 1 * batch_size * hid_dim
      c_n = final_states[1]
    # Eqn. 22: final output
    # todo: two linear functions
    y_pred = self.fc_final(torch.cat((d_n[0], context), dim=1))

    return y_pred

  def _init_state(self, X):
    """Initialize all 0 hidden states and cell states for encoder.

        Args:
            X
        Returns:
            initial_hidden_states

        """
    # hidden state and cell state [num_layers*num_directions, batch_size, hidden_size]
    # https://pytorch.org/docs/master/nn.html?#lstm
    initial_state = X.new_zeros([1, X.shape[0], self.hid_dim])
    return initial_state


class ClassDecoder(nn.Module):
  """decoder in DA_RNN."""

  def __init__(self, timesteps, feat_dim, hid_dim):
    """Initialize a decoder in DA_RNN.
    feat_dim: encoder hidden state dim
    """
    super(ClassDecoder, self).__init__()
    self.hid_dim = hid_dim
    # todo: timesteps = T-1 in zhen code
    self.timesteps = timesteps

    self.attn = nn.Sequential(
        nn.Linear(2 * hid_dim + feat_dim, feat_dim), nn.Tanh(),
        nn.Linear(feat_dim, 1))
    self.lstm = nn.LSTM(input_size=1, hidden_size=hid_dim, batch_first=True)
    self.fc = nn.Linear(feat_dim + 1, 1)
    self.fc_final = nn.Linear(hid_dim + feat_dim, 2)

    # todo remove
    self.count = 0

  def forward(self, H, Y):
    """forward."""
    d_n = self._init_state(H)
    c_n = self._init_state(H)

    # todo debug
    self.count += 1
    for t in range(self.timesteps):

      # (batchsize, timesteps, 2*timesteps + feat_dim)
      x = torch.cat((d_n.expand(self.timesteps, -1, -1).permute(1, 0, 2),
                     c_n.expand(self.timesteps, -1, -1).permute(1, 0, 2), H),
                    dim=2)

      # (batchsize, timesteps)
      beta = F.softmax(self.attn(x).squeeze(), dim=1)
      # Eqn. 14: compute context vector
      # (batchsize, feat_dim)
      context = torch.bmm(beta.unsqueeze(1), H).squeeze()
      # Eqn. 15
      # batch_size * 1
      y_tilde = self.fc(torch.cat((context, Y[:, t].unsqueeze(1)), dim=1))

      # todo debug
      if DEBUG and True and self.count == 3:
        # test equalty
        np.save(test_dir + 'new_ytilde', y_tilde.detach().numpy())
        import ipdb
        ipdb.set_trace(context=7)

      # Eqn. 16: LSTM
      self.lstm.flatten_parameters()
      _, final_states = self.lstm(y_tilde.unsqueeze(1), (d_n, c_n))
      # 1 * batch_size * hid_dim
      d_n = final_states[0]
      # 1 * batch_size * hid_dim
      c_n = final_states[1]
    # Eqn. 22: final output
    # todo: two linear functions
    logits = self.fc_final(torch.cat((d_n[0], context), dim=1))
    output = F.log_softmax(logits, dim=1)

    return output

  def _init_state(self, X):
    """Initialize all 0 hidden states and cell states for encoder.

        Args:
            X
        Returns:
            initial_hidden_states

        """
    # hidden state and cell state [num_layers*num_directions, batch_size, hidden_size]
    # https://pytorch.org/docs/master/nn.html?#lstm
    initial_state = X.new_zeros([1, X.shape[0], self.hid_dim])
    return initial_state


if __name__ == "__main__":
  # from importlib import reload
  from main import opt
  from PrepareData import Nas100Dataset
  dataset = Nas100Dataset()
  pre_dataset, train_loader = dataset.get_data_loader(opt)
  feat_dim = 81

  encoder = Encoder(opt.timesteps, feat_dim, opt.hid_dim_encoder)
  decoder = Decoder(opt.timesteps, opt.hid_dim_encoder, opt.hid_dim_decoder)
  criterion = nn.MSELoss()
  encoder_optimizer = optim.Adam(encoder.parameters(), lr=opt.lr)
  decoder_optimizer = optim.Adam(decoder.parameters(), lr=opt.lr)

  n_batches_count = 1
  epoch_batch_loss_list = list()
  for epoch in range(opt.epochs):
    batch_loss_list = list()
    for data_dict in train_loader:
      X = data_dict['X'].type(torch.FloatTensor)
      Y = data_dict['Y'].type(torch.FloatTensor).squeeze()
      Ygt = data_dict['Ygt'].type(torch.FloatTensor)
      H = encoder(X)
      Ypred = decoder(H, Y)
      loss = criterion(Ypred.squeeze(), Ygt)

      encoder_optimizer.zero_grad()
      decoder_optimizer.zero_grad()
      loss.backward()
      encoder_optimizer.step()
      decoder_optimizer.step()

      batch_loss_list.append(loss.item())
      if n_batches_count % 50000 == 0:
        for p in encoder_optimizer.param_groups:
          p['lr'] *= 0.9
        for p in decoder_optimizer.param_groups:
          p['lr'] *= 0.9
      n_batches_count += 1

    epoch_batch_loss_list.append(batch_loss_list)
