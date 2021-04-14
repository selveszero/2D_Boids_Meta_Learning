from dataset import get_train_data, Dataset_builder
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
from model import BaselineModel


def plot_2D(position, orientation, obs_time=None, title_name=None):
    n_agent = position.shape[1]
    fig = plt.figure()
    # use position data to plot the 3D trajectory
    for obs_index in range(n_agent):
        if obs_time != None:
            x = position[:obs_time, obs_index, 0]
            y = position[:obs_time, obs_index, 1]
        else:
            x = position[:, obs_index, 0]
            y = position[:, obs_index, 1]
        plt.plot(x, y, label='motion' + str(obs_index))
    plt.legend()
    plt.title(title_name)
    plt.show()

def train_model(EPOCHS, LR, device, model, train_loader, val_loader):
    # initial the optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # initial running loss
    running_train_loss = 0
    running_val_loss = 0

    for epoch_index in range(EPOCHS):
        for iter_index, data in enumerate(train_loader, 0):
            data = data.to(device)
            bs, t, n_agent, s_dim = data.shape
            forward_t = t-1

            pred = torch.zeros(*data.shape).to(device)
            pred[:, 0] = data[:, 0]  # given initial state
            # init the hidden state
            init_s = data[:,0].reshape(-1,s_dim)  # intput the first state into model
            h = model.initHidden(n_agent)
            # forward
            for t_index in range(forward_t):
                s, h = model(init_s, h)
                pred[:, t_index+1] = s

            # compute loss function
            loss = F.mse_loss(pred, data)
            optimizer.zero_grad()
            loss.backward()  # backpropogation
            optimizer.step()
            # add up run loss for each batch
            running_train_loss += loss.item()

        # log
        avg_train_loss = running_train_loss/len(train_loader)
        running_train_loss = 0
        print('Loss: {:.4f}'.format(avg_train_loss))
    pass



if __name__ == '__main__':
    # hyper-para setting
    EPOCHS = 1000
    LR = 0.01
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get the training data
    train_data_path = './data/train'
    val_data_path = './data/val'
    test_data_path = './data/test'
    visual_data_path = './data/visual'
    # get the dataset
    data_length = 1000  # specify the length we want
    train_set = get_train_data(train_data_path, data_length=data_length, NOISE_VAR=0)
    val_set = get_train_data(val_data_path, data_length=data_length, NOISE_VAR=0)
    test_set = get_train_data(test_data_path, data_length=data_length, NOISE_VAR=0)
    visual_set = get_train_data(visual_data_path, data_length=data_length, NOISE_VAR=0)

    # build pytorch dataset
    train_data = Dataset_builder(train_set)
    val_data = Dataset_builder(val_set)
    test_data = Dataset_builder(test_set)
    # build pytorch dataloader
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

    ## plot trajectory
    position = train_set[0][:, :, :2]
    orientation = train_set[0][:, :, 2:]
    # plot traj
    plot_2D(position, orientation, obs_time=3000)

    ## Training
    # initial the model
    model = BaselineModel(device)
    model = model.to(device)
    train_model(EPOCHS, LR, device, model, train_loader, val_loader)