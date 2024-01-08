import pickle

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator

with open('./obj_path', 'rb') as file:
    data = pickle.load(file)

OS_colormap = ['grey', 'green', 'gold', 'red']  # the colormap for object status
OS_colormap2 = ['green', 'purple', 'deepskyblue']  # the colormap for object status


def vis_thread():
    # setup for matplotlib plot
    matplotlib.use('TkAgg')  # set matplotlib backend
    # plt.rcParams['toolbar'] = 'None'  # disable the toolbar
    # create a figure
    fig = plt.figure(figsize=(9, 9))
    ax1 = fig.add_subplot(111)
    # adjust figure position
    mngr = plt.get_current_fig_manager()
    mngr.window.wm_geometry('+300+30')
    # draws a completely frameless window
    win = plt.gcf().canvas.manager.window
    win.overrideredirect(1)

    ax1.set_xlim(-2, 2)
    ax1.set_ylim(0, 4)
    ax1.xaxis.set_major_locator(LinearLocator(5))  # set axis scale
    ax1.yaxis.set_major_locator(LinearLocator(5))
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Object Path')

    for d in data:
        # # print position & status
        ax1.plot(float(d[0]), float(d[1]), marker='.', linestyle='None', color=OS_colormap[int(d[-2])])
        # ax1.plot(-float(d[0]), 4-float(d[1]), marker='.', linestyle='None', color=OS_colormap[int(d[-2])])
        # print position & objects
        # ax1.plot(-float(d[0]), 4-float(d[1]), marker='.', linestyle='None', color=OS_colormap2[int(d[-1].split('_')[-1])])
        # if d[-1] == 'obj_0':
        #     ax1.plot(-float(d[0]), 4-float(d[1]), marker='.', linestyle='None', color=OS_colormap2[int(d[-1].split('_')[-1])])

    plt.show()


vis_thread()
