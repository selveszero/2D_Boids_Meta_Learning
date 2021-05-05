from torch.utils.data import DataLoader, TensorDataset
import torch
import os
import numpy as np

def get_train_data(dir_path, data_length, NOISE_VAR, skip_begin_step=10, sample_rate=1):
  data_set = []
  for f_name in os.listdir(dir_path):
    file = open(os.path.join(dir_path, f_name), 'r')
    Lines = file.readlines()[:data_length]
    assert len(Lines) == data_length
    Xt_list = []
    for line in Lines:
      raw_data_list = line.split(')(')
      # remove redandent information
      raw_data_list[0] = raw_data_list[0][1:]
      raw_data_list[-1] = raw_data_list[-1][:-2]
      # create position array and orientation array (the following code may change, if the provide feature changes)
      position_list = []
      orientation_list = []
      for index, element in enumerate(raw_data_list):
        if index % 2 == 0:
          position_list.append(eval('np.array(['+element+'])'))
        elif index % 2 == 1:
          orientation_list.append(eval('np.array(['+element+'])'))
      # concatinate into matrix
      position = np.array(position_list)
      orientation = np.array(orientation_list)
      # combined into X matrix at time t
      Xt = np.hstack((position, orientation))
      # add one extra dimension to the matrix
      Xt_add_dim = np.expand_dims(Xt, axis=0)
      Xt_list.append(Xt_add_dim)
    # combine Xts into final X tensor
    X = np.concatenate(Xt_list, axis=0)
    X = torch.tensor(X, dtype=torch.float)
    X = X[skip_begin_step::sample_rate, :, :]
    # add NOISE
    if NOISE_VAR != 0:
      X = X + torch.randn_like(X)*NOISE_VAR
    data_set.append(X)

  return data_set

class Dataset_builder(torch.utils.data.Dataset):
    def __init__(self, dataset):
        # readin a list of data
        self.dataset = dataset
    def __getitem__(self, index):
        data = torch.tensor(self.dataset[index], dtype=torch.float)
        return data
    def __len__(self):
        return len(self.dataset)