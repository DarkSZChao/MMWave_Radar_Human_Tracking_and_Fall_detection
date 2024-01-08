import pickle

import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator

from library.utils import *

RP_colormap = ['C5', 'C7', 'C8']  # the colormap for radar raw points
# ES_colormap = ['lavender', 'thistle', 'violet', 'darkorchid', 'indigo']  # the colormap for radar energy strength
ES_colormap = ['thistle', 'violet', 'violet', 'darkorchid', 'indigo']  # the colormap for radar energy strength
OS_colormap = ['grey', 'green', 'gold', 'red']  # the colormap for object status


def data_concatenate(_data):
    _data_points = np.ndarray([0, 5])
    for d in _data:
        for d_np in list(d.values()):
            _data_points = np.concatenate([_data_points, d_np])

    # # remove points with low energy strength
    # data_points, noise = np_filter(data_points, axis=4, range_lim=(0, None))
    # # identify the noise with speed
    # if len(noise) > 0:
    #     noise, noise_with_speed = np_filter(noise, axis=3, range_lim=0)
    #     data_points = np.concatenate([data_points, noise_with_speed])
    return _data_points


def plot_2D(_data_points, axes=(0, 1), xlim=(-1.5, 1.5), ylim=(0, 3)):
    fig_2D = plt.figure(figsize=(9, 9))
    ax_2D = fig_2D.add_subplot(111)
    ax_2D.set_xlim(xlim[0], xlim[1])
    ax_2D.set_ylim(ylim[0], ylim[1])
    ax_2D.xaxis.set_major_locator(LinearLocator(5))  # set axis scale
    ax_2D.yaxis.set_major_locator(LinearLocator(5))
    labels = ('X (m)', 'Y (m)', 'Height (m)')
    ax_2D.set_xlabel(labels[axes[0]])
    ax_2D.set_ylabel(labels[axes[1]])

    # draw signal energy strength
    for i in range(len(ES_colormap)):
        val_data_allradar_ES, _ = np_filter(_data_points, axis=4, range_lim=(i * 100, (i + 1) * 100))
        ax_2D.plot(val_data_allradar_ES[:, axes[0]], val_data_allradar_ES[:, axes[1]], marker='.', linestyle='None', color=ES_colormap[i])


def plot_3D(_data_points):
    fig_2D = plt.figure(figsize=(9, 9))
    ax_2D = fig_2D.add_subplot(111, projection='3d')
    ax_2D.set_xlim(-2, 2)
    ax_2D.set_ylim(0, 3)
    ax_2D.set_zlim(0, 2)
    ax_2D.xaxis.set_major_locator(LinearLocator(5))  # set axis scale
    ax_2D.yaxis.set_major_locator(LinearLocator(5))
    ax_2D.zaxis.set_major_locator(LinearLocator(5))
    ax_2D.set_xlabel('X (m)')
    ax_2D.set_ylabel('Y (m)')
    ax_2D.set_zlabel('Height (m)')

    # draw signal energy strength
    for i in range(len(ES_colormap)):
        val_data_allradar_ES, _ = np_filter(_data_points, axis=4, range_lim=(i * 100, (i + 1) * 100))
        ax_2D.plot3D(val_data_allradar_ES[:, 0], val_data_allradar_ES[:, 1], val_data_allradar_ES[:, 2], marker='.', linestyle='None', color=ES_colormap[i])


def plot_hist(_data):
    fig_hist = plt.figure()
    ax_hist = fig_hist.add_subplot(111)
    ax_hist.set_xlabel('Points')
    ax_hist.set_ylabel('Height (m)')
    ax_hist.set_ylim(0, 2)
    ax_hist.hist(_data, bins=20, range=(0, 2), color='blue', alpha=0.7, rwidth=0.85, orientation='horizontal')  # Plotting the histogram vertically


if __name__ == '__main__':
    """radar placement evaluation"""
    # with open('../data/Radar_placement/55deg_2', 'rb') as file:
    #     data = pickle.load(file)[400:]
    # data_points = data_concatenate(data)
    #
    # # remove out-ranged points
    # data_points, _ = np_filter(data_points, axis=0, range_lim=(-1, 1))
    # data_points, _ = np_filter(data_points, axis=1, range_lim=(0, 2))
    # data_points, _ = np_filter(data_points, axis=2, range_lim=(0, 2))
    # # custom filter
    # left, right = np_filter(data_points, axis=0, range_lim=(None, 0.4))
    # right_down, right_up = np_filter(right, axis=2, range_lim=(None, 1.5))
    # # inverse selection
    # data_points = np_2D_set_operations(data_points, right_up, ops='subtract')
    #
    # # plot points in 2D
    # plot_2D(data_points, axes=(0, 2))
    # # plot points in hist
    # plot_hist(data_points[:, 2])
    # # Display the plot
    # plt.show()

    """human fall posture evaluation"""
    with open('../data/Fall_posture/1', 'rb') as file:
        data = pickle.load(file)
    data_points = data_concatenate(data)

    # remove out-ranged points
    data_points, _ = np_filter(data_points, axis=0, range_lim=(-1, 1))
    data_points, _ = np_filter(data_points, axis=1, range_lim=(0, 2.5))
    data_points, _ = np_filter(data_points, axis=2, range_lim=(0, 0.5))
    # custom filter
    left, right = np_filter(data_points, axis=0, range_lim=(None, -0.32))
    left_front, left_back = np_filter(left, axis=1, range_lim=(None, 2.25))
    # inverse selection
    data_points = np_2D_set_operations(data_points, left_back, ops='subtract')

    # plot points in 2D/3D
    plot_2D(data_points, axes=(0, 1))
    plot_2D(data_points, axes=(1, 2), xlim=(0, 3), ylim=(0, 2))
    plot_3D(data_points)

    # Display the plot
    plt.show()

    # with open('../data/Fall_posture/2', 'rb') as file:
    #     data = pickle.load(file)
    # data_points = data_concatenate(data)
    #
    # # remove out-ranged points
    # data_points, _ = np_filter(data_points, axis=0, range_lim=(-1, 0.4))
    # data_points, _ = np_filter(data_points, axis=1, range_lim=(0.5, 2.5))
    # data_points, _ = np_filter(data_points, axis=2, range_lim=(0, 0.6))
    #
    # # plot points in 2D/3D
    # plot_2D(data_points, axes=(0, 1))
    # plot_2D(data_points, axes=(1, 2), xlim=(0, 3), ylim=(0, 2))
    # plot_3D(data_points)
    #
    # # Display the plot
    # plt.show()

    # with open('../data/Fall_posture/3', 'rb') as file:
    #     data = pickle.load(file)
    # data_points = data_concatenate(data)
    #
    # # remove out-ranged points
    # data_points, _ = np_filter(data_points, axis=0, range_lim=(-1, 0.6))
    # data_points, _ = np_filter(data_points, axis=1, range_lim=(0, 2.5))
    # data_points, _ = np_filter(data_points, axis=2, range_lim=(0, 1.2))
    # # custom filter
    # front, back = np_filter(data_points, axis=1, range_lim=(None, 1.2))
    # front_bottom, front_top = np_filter(front, axis=2, range_lim=(None, 0.35))
    # # inverse selection
    # data_points = np_2D_set_operations(data_points, front_top, ops='subtract')
    #
    # # plot points in 2D/3D
    # plot_2D(data_points, axes=(0, 1))
    # plot_2D(data_points, axes=(1, 2), xlim=(0, 3), ylim=(0, 2))
    # plot_3D(data_points)
    #
    # # Display the plot
    # plt.show()

