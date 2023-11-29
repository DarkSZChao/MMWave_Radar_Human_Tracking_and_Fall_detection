import pickle

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator

with open('C:/SZC/PhD/MMWave_Radar/ID/Data/FAUSTSim/faust_pred_pointcloud_10000_dis_randspeed', 'rb') as file:
    data, _ = pickle.load(file)

idx = [78, 1188, 2288]


def plot(cur_dataset):
    # setup for matplotlib plot
    matplotlib.use('TkAgg')  # set matplotlib backend
    # plt.rcParams['toolbar'] = 'None'  # disable the toolbar
    # create a figure
    fig = plt.figure(figsize=(9, 9))
    ax1 = fig.add_subplot(111, projection='3d')
    # adjust figure position
    mngr = plt.get_current_fig_manager()
    mngr.window.wm_geometry('+300+30')

    ax1.xaxis.set_major_locator(LinearLocator(3))  # set axis scale
    ax1.yaxis.set_major_locator(LinearLocator(3))
    ax1.zaxis.set_major_locator(LinearLocator(3))
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    ax1.plot3D(cur_dataset[:, 0], cur_dataset[:, 1], cur_dataset[:, 2], marker='o', linestyle='None')


for i in idx:
    plot(data[i])
plt.show()
