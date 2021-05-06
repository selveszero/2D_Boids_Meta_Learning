import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


def position_consensus(position,title_name='None'):
    fig = plt.figure()
    fig.set_size_inches(10, 5)


    pos_diff_matrix = np.sqrt(np.sum((np.expand_dims(position, axis=-2) - np.expand_dims(position, axis=-3)) ** 2, axis=-1))
    pos_consensus = np.sum(pos_diff_matrix, axis=(1, 2)) / 2
    plt.plot(pos_consensus, label='pos_consensus')
    plt.title('Position Consensus')
    plt.xlabel('Time')
    plt.ylabel('Position Vector Difference Norm Sum')
    # plt.xlim([0, 400])
    # plt.ylim([125, 280])
    plt.legend()
    plt.title(title_name)
    plt.savefig('./images/'+title_name+'.jpg')
    print('mean position consensus of last 20 time stamps', np.mean(pos_consensus[-20:]))
    plt.show()