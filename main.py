from dataset import get_train_data, Dataset_builder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


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



if __name__ == '__main__':
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