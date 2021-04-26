from dataset import get_train_data, Dataset_builder
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
from model import BaselineModel
from metrics import *
import tqdm
import copy

import numpy as np
from matplotlib import animation, rc
from IPython.display import HTML, Image # For GIF

torch.autograd.set_detect_anomaly(True)

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
    
def plot_animation(position, orientation, obs_time=50, interval=20, title_name='animation', save_gif=False):
    # lower interval --> faster animation as displayed on colab
    # higher fps --> faster animation in the GIF
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', (0.5, 0.0, 0.5)]
    fig, ax = plt.subplots()
    n_agent = position.shape[1]
    duration = position.shape[0]
    lines = []
    for obs_index in range(n_agent):
        line, = ax.plot(position[0,obs_index,0], position[0,obs_index,1], "o-")
        lines.append(line)

    def connect(i):
        start=max((i-obs_time,0))
        for lnum,line in enumerate(lines):
            line.set_data(position[start:i,lnum,0],position[start:i,lnum,1])
            line.set_color(colors[lnum])
            line.set_alpha(0.7)
        return lines

    ax.set_xlim(0,500)
    ax.set_ylim(0,500)
    anim = animation.FuncAnimation(fig, connect, np.arange(1, duration), interval=interval)
    rc('animation', html='jshtml')
    if save_gif:
      anim.save(title_name + '.gif', writer='pillow', fps=50)
      HTML(anim.to_html5_video())
    return anim

def St_compute(s_pos, perception_range=50):
    with torch.no_grad():
        offset_mat = s_pos.unsqueeze(-2) - s_pos.unsqueeze(-3)  # current - neighbor
        dist_mat = torch.sqrt(torch.sum(offset_mat ** 2, dim=-1))  # compute dist mat along time, (t, n_agent, n_agent)
        St = (dist_mat < perception_range).type(torch.float)
        # normalization
        St_sum = torch.sum(St, dim=2)
        St_norm = St / (St_sum.unsqueeze(-1))
    return St, St_norm, dist_mat


def topk_neighbor(s, k=3):
    bs, n_agent, state_dim = s.shape
    St, _, dist_mat = St_compute(s[:, :, :2])
    St_unsqz = St.unsqueeze(-1)  # (bs,n,n,1)
    s_unsqz = s.unsqueeze(1).clone()  # (bs,1,n,state_dim)
    # zero out the non-neighbor, filter
    dec_s = St_unsqz * s_unsqz  # (bs,n,n,state_dim), two n: for each agent, corresponding to all other n agents
    # choose the topk close agent (except self node)
    close_k_index = torch.topk(dist_mat, k + 1, dim=2, largest=False, sorted=True)[1][:, :, 1:]  # return closest, except self
    close_k_index_expand = close_k_index.unsqueeze(-1).expand(*close_k_index.shape, state_dim)

    s_neigh = torch.gather(dec_s, 2, close_k_index_expand)
    return s, s_neigh

def train_model(EPOCHS, LR, device, model, train_loader, val_loader, vis_init):
    # Meta Learning
    meta_train = True
    # init setting
    log_freq = 10
    # initial the optimizer
    # optimizer = optim.Adam(model.parameters(), lr=LR)
    # optimizer_meta = optimizer(model)
    # initial running loss
    running_train_loss = 0
    running_val_loss = 0
    # init loss arr
    train_loss_arr = []
    val_loss_arr = []

    for epoch_index in tqdm.tqdm(range(EPOCHS)):
        # if meta_train:
        model_epoch = copy.deepcopy(model)
        meta_grad = []
        meta_optimizer = optim.Adam(model_epoch.parameters(), lr=LR)
        for iter_index, data in enumerate(train_loader, 0):
            # if meta_train:
            model = copy.deepcopy(model_epoch)
            optimizer = optim.Adam(model.parameters(), lr=LR)
            data = data.to(device)
            bs, t, n_agent, s_dim = data.shape
            forward_t = t-1
            x = data[:, :-1]
            y = data[:, 1:]
            y_delta = y-x
            # init 0 pred
            pred_delta = torch.zeros(bs, t-1, n_agent, s_dim).to(device)
            # init the hidden state
            h = model.initHidden(n_agent, bs)
            s = x[:, 0].reshape(-1, s_dim).unsqueeze(0) # intput the first state into model
            for t_index in range(forward_t):
                s, s_neighbor = topk_neighbor(s)
                s_input = torch.cat([s.unsqueeze(-2), s_neighbor], dim=2).flatten(start_dim=2, end_dim=3)
                s_residual, h = model(s_input, h)
                s = s+s_residual
                pred_delta[:, t_index] = s_residual.squeeze(0).reshape(bs, n_agent, s_dim)

            # compute loss function
            loss = F.mse_loss(pred_delta, y_delta)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)  # backpropogation
            optimizer.step()
            # record train loss
            train_loss_arr.append(loss.item())
            # add up run loss for each batch
            running_train_loss += loss.item()

            ## meta sub-train
            if meta_train:
                # data = data.to(device)
                # bs, t, n_agent, s_dim = data.shape
                # forward_t = t - 1
                # x = data[:, :-1]
                # y = data[:, 1:]
                # y_delta = y - x
                # init 0 pred
                pred_delta = torch.zeros(bs, t - 1, n_agent, s_dim).to(device)
                # init the hidden state
                h = model.initHidden(n_agent, bs)
                s = x[:, 0].reshape(-1, s_dim).unsqueeze(0)  # intput the first state into model
                for t_index in range(forward_t):
                    s, s_neighbor = topk_neighbor(s)
                    s_input = torch.cat([s.unsqueeze(-2), s_neighbor], dim=2).flatten(start_dim=2, end_dim=3)
                    s_residual, h = model(s_input, h)
                    s = s + s_residual
                    pred_delta[:, t_index] = s_residual.squeeze(0).reshape(bs, n_agent, s_dim)

                # compute loss function
                optimizer.zero_grad()
                loss = F.mse_loss(pred_delta, y_delta)
                loss.backward()
                meta_grad.append(model.parameters())
        ## Meta update on each epoch
        if meta_train:
            # gradient list for all
            # add up the gradient
            for m_index, m_para in enumerate(meta_grad):
                if m_index == 0:
                    grad_list = []
                    for m_index, m in enumerate(m_para):
                        temp = m.grad.data
                        grad_list.append(temp)
                        if m_index == 19:
                            print(temp)
                else:
                    for m_index, m in enumerate(m_para):
                        temp = m.grad.data
                        grad_list[m_index] += temp
                        if m_index == 19:
                            print(temp)
            print()

        # log
        if epoch_index % log_freq == 0:
            # log the training loss function
            avg_train_loss = running_train_loss/len(train_loader)
            running_train_loss = 0

            # eval
            with torch.no_grad():
                for iter_index, data in enumerate(val_loader, 0):
                    data = data.to(device)
                    bs, t, n_agent, s_dim = data.shape
                    forward_t = t - 1
                    x = data[:, :-1]
                    y = data[:, 1:]
                    y_delta = y - x
                    # init 0 pred
                    pred_delta = torch.zeros(bs, t - 1, n_agent, s_dim).to(device)
                    # init the hidden state
                    h = model.initHidden(n_agent, bs)
                    s = x[:, 0].reshape(-1, s_dim).unsqueeze(0)  # intput the first state into model
                    for t_index in range(forward_t):
                        s, s_neighbor = topk_neighbor(s)
                        s_input = torch.cat([s.unsqueeze(-2), s_neighbor], dim=2).flatten(start_dim=2, end_dim=3)
                        s_residual, h = model(s_input, h)
                        s = s + s_residual
                        pred_delta[:, t_index] = s_residual.squeeze(0).reshape(bs, n_agent, s_dim)

                    # compute loss function
                    loss = F.mse_loss(pred_delta, y_delta)
                    # add up run loss for each batch
                    running_val_loss += loss.item()
                # record avg eval loss
                val_loss_avg = running_val_loss/len(val_loader)
                val_loss_arr.append(val_loss_avg)

            print('Epoch: {}\tTrain Loss: {:.4f}\tVal loss: {:.4f}'.format(epoch_index, avg_train_loss, val_loss_avg))


            ## Visualization
            h = model.initHidden(n_agent, bs)
            vis_traj = torch.zeros(t, n_agent, s_dim).to(device)
            vis_traj[0] = vis_init.squeeze(0)
            s = vis_init
            for t_index in range(forward_t):
                s, s_neighbor = topk_neighbor(s)
                s_input = torch.cat([s.unsqueeze(-2), s_neighbor], dim=2).flatten(start_dim=2, end_dim=3)
                s_residual, h = model(s_input, h)
                s = s + s_residual
                # s_end = s_end.squeeze(0).reshape(bs, n_agent, s_dim)
                vis_traj[t_index+1] = s.squeeze(0)
            # pred and plot the traj
            vis_traj = vis_traj.to('cpu').detach().numpy()
            position = vis_traj[:, :, :2]
            orientation = vis_traj[:, :, 2:]
            plot_2D(position, orientation, obs_time=3000)

            ## Save model
            torch.save(model, 'trained_model_{}.pt'.format(epoch_index))
    pass



if __name__ == '__main__':
    # hyper-para setting
    EPOCHS = 1000
    LR = 0.0005
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get the training data
    train_data_path = './data/train'
    val_data_path = './data/val'
    test_data_path = './data/test'
    visual_data_path = './data/visual'
    # get the dataset
    data_length = 500  # specify the length we want
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
    anim = plot_animation(position, orientation, save_gif=False)
    print(anim)

    ## Training
    vis_init = visual_set[0][0].unsqueeze(0).to(device)
    # initial the model
    model = BaselineModel(device)
    model = model.to(device)
    train_model(EPOCHS, LR, device, model, train_loader, val_loader, vis_init)
