import math
import pickle

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator

from library import FramePProcessor, folder_clean_recreate

RP_colormap = ['C5', 'C7', 'C8']  # the colormap for radar raw points
ES_colormap = ['lavender', 'thistle', 'violet', 'darkorchid', 'indigo']  # the colormap for radar energy strength
OS_colormap = ['grey', 'green', 'gold', 'red']  # the colormap for object status


class Visualizer:
    def __init__(self, cur_dataset, interval=0.1, image_output_enable=False, **kwargs_CFG):
        """
        pass config static parameters
        """
        """ module own config """
        VIS_CFG = kwargs_CFG['VISUALIZER_CFG']
        self.dimension = VIS_CFG['dimension']
        self.VIS_xlim = VIS_CFG['VIS_xlim']
        self.VIS_ylim = VIS_CFG['VIS_ylim']
        self.VIS_zlim = VIS_CFG['VIS_zlim']

        """ other configs """
        self.RDR_CFG_LIST = kwargs_CFG['RADAR_CFG_LIST']

        """self content"""
        self.cur_dataset = cur_dataset
        self.interval = interval
        self.obj_path_saved = np.ndarray([0, 5])
        self.image_output_enable = image_output_enable
        if self.image_output_enable:
            self.image_dir = './temp/'
            folder_clean_recreate(self.image_dir)
            self.image_dpi = 150

        self.fpp = FramePProcessor(**kwargs_CFG)  # call other class

        # setup for matplotlib plot
        matplotlib.use('TkAgg')  # set matplotlib backend
        # plt.rcParams['toolbar'] = 'None'  # disable the toolbar
        # create a figure
        self.fig = plt.figure(figsize=(9, 9))
        # adjust figure position
        mngr = plt.get_current_fig_manager()
        mngr.window.wm_geometry('+150+30')
        # draws a completely frameless window
        win = plt.gcf().canvas.manager.window
        win.overrideredirect(1)
        # interactive mode on, no need plt.show()
        plt.ion()

    def run(self):
        if self.dimension == '2D':
            # create a plot
            ax1 = self.fig.add_subplot(111)

            for idx, cur_data in enumerate(self.cur_dataset):
                # clear and reset
                plt.cla()
                ax1.set_xlim(self.VIS_xlim[0], self.VIS_xlim[1])
                ax1.set_ylim(self.VIS_ylim[0], self.VIS_ylim[1])
                ax1.xaxis.set_major_locator(LinearLocator(5))  # set axis scale
                ax1.yaxis.set_major_locator(LinearLocator(5))
                ax1.set_xlabel('x')
                ax1.set_ylabel('y')
                ax1.set_title('Radar')
                # update the canvas
                self._update_canvas(ax1, cur_data)
                print(idx)
                if self.image_output_enable:
                    self.fig.savefig(self.image_dir + f'{idx:03d}.png', dpi=self.image_dpi)

        elif self.dimension == '3D':
            # create a plot
            ax1 = self.fig.add_subplot(111, projection='3d')

            spin = 0
            for idx, cur_data in enumerate(self.cur_dataset):
                # clear and reset
                plt.cla()
                ax1.set_xlim(self.VIS_xlim[0], self.VIS_xlim[1])
                ax1.set_ylim(self.VIS_ylim[0], self.VIS_ylim[1])
                ax1.set_zlim(self.VIS_zlim[0], self.VIS_zlim[1])
                ax1.xaxis.set_major_locator(LinearLocator(3))  # set axis scale
                ax1.yaxis.set_major_locator(LinearLocator(3))
                ax1.zaxis.set_major_locator(LinearLocator(3))
                ax1.set_xlabel('x')
                ax1.set_ylabel('y')
                ax1.set_zlabel('z')
                ax1.set_title('Radar')
                spin += 0.04
                ax1.view_init(ax1.elev - 0.5 * math.sin(spin), ax1.azim - 0.3 * math.sin(0.2 * spin))  # spin the view angle
                # update the canvas
                self._update_canvas(ax1, cur_data)
                print(idx)
                if self.image_output_enable:
                    self.fig.savefig(self.image_dir + f'{idx:03d}.png', dpi=self.image_dpi)

    def _update_canvas(self, ax1, cur_data):
        # draw radar point
        for RDR_CFG in self.RDR_CFG_LIST:
            self._plot(ax1, [RDR_CFG['pos_offset'][0]], [RDR_CFG['pos_offset'][1]], [RDR_CFG['pos_offset'][2]], marker='o', color='DarkRed')

        # get values from queues of all radars
        val_data_allradar = np.ndarray([0, 5], dtype=np.float16)
        ES_noise_allradar = np.ndarray([0, 5], dtype=np.float16)
        for i, RDR_CFG in enumerate(self.RDR_CFG_LIST):
            data_1radar = cur_data[RDR_CFG['name']]

            # apply ES and speed filter for each radar channel
            val_data, ES_noise = self.fpp.DP_ES_Speed_filter(data_1radar, RDR_CFG['ES_threshold'])
            val_data_allradar = np.concatenate([val_data_allradar, val_data])
            ES_noise_allradar = np.concatenate([ES_noise_allradar, ES_noise])

        # apply global boundary filter
        val_data_allradar = self.fpp.FPP_boundary_filter(val_data_allradar)
        ES_noise_allradar = self.fpp.FPP_boundary_filter(ES_noise_allradar)
        # apply global energy strength filter
        val_data_allradar, global_ES_noise = self.fpp.FPP_ES_Speed_filter(val_data_allradar)
        # apply background noise filter
        val_data_allradar = self.fpp.BGN_filter(val_data_allradar)

        # val_data_allradar[:, 0] = -val_data_allradar[:, 0]
        # val_data_allradar[:, 1] = 4 - val_data_allradar[:, 1]

        # draw signal energy strength
        for i in range(len(ES_colormap)):
            val_data_allradar_ES, _ = self.fpp.DP_np_filter(val_data_allradar, axis=4, range_lim=(i * 100, (i + 1) * 100))
            self._plot(ax1, val_data_allradar_ES[:, 0], val_data_allradar_ES[:, 1], val_data_allradar_ES[:, 2], marker='.', color=ES_colormap[i])

        # draw valid point, DBSCAN envelope
        # vertices_list, valid_points_list, _, DBS_noise = self.fpp.DBS(val_data_allradar)
        vertices_list, valid_points_list, _, DBS_noise = self.fpp.DBS_dynamic_ES(val_data_allradar)
        for vertices in vertices_list:
            self._plot(ax1, vertices[:, 0], vertices[:, 1], vertices[:, 2], linestyle='-', color='salmon')

        # background noise filter
        if self.fpp.BGN_enable:
            # update the background noise
            if len(vertices_list) > 0:
                self.fpp.BGN_update(np.concatenate([ES_noise_allradar, global_ES_noise, DBS_noise]))
            else:
                self.fpp.BGN_update(np.concatenate([ES_noise_allradar, global_ES_noise]))
            # draw BGN area
            BGN_block_list = self.fpp.BGN_get_filter_area()
            for bgn in BGN_block_list:
                self._plot(ax1, bgn[:, 0], bgn[:, 1], bgn[:, 2], marker='.', linestyle='-', color='g')

        # tracking system
        if self.fpp.TRK_enable:
            self.fpp.TRK_update_poss_matrix(valid_points_list)
            # draw object central points
            for person in self.fpp.TRK_people_list:
                obj_cp, obj_status = person.get_info()
                self._plot(ax1, obj_cp[:, 0], obj_cp[:, 1], obj_cp[:, 2], marker='o', color=OS_colormap[obj_status])
                # if obj_status == 3:  # warning when object falls
                #     winsound.Beep(1000, 20)

                # if obj_cp.size > 0 and person.name == 'obj_0':
                if obj_cp.size > 0:
                    self.obj_path_saved = np.concatenate([self.obj_path_saved, np.concatenate([obj_cp, [[obj_status]], [[person.name]]], axis=1)])

        # wait at the end of each loop
        plt.pause(self.interval)

    def _plot(self, ax, x, y, z, fmt='', **kwargs):
        """
        :param ax: the current canvas
        :param x: data in x-axis
        :param y: data in y-axis
        :param z: data in z-axis
        :param fmt: plot and plot3D fmt
        :param kwargs: plot and plot3D marker, linestyle, color
        :return: None
        """
        if len(fmt) > 0:  # if fmt is using
            if self.dimension == '2D':
                # ax.plot(x, y, fmt)
                ax.plot(-np.array(x), 4-np.array(y), fmt)
            elif self.dimension == '3D':
                ax.plot3D(x, y, z, fmt)
        else:  # if para is using
            for i in ['marker', 'linestyle', 'color']:
                if not (i in kwargs):
                    kwargs[i] = 'None'
            if self.dimension == '2D':
                # ax.plot(x, y, marker=kwargs['marker'], linestyle=kwargs['linestyle'], color=kwargs['color'])
                ax.plot(-np.array(x), 4-np.array(y), marker=kwargs['marker'], linestyle=kwargs['linestyle'], color=kwargs['color'])
            elif self.dimension == '3D':
                ax.plot3D(x, y, z, marker=kwargs['marker'], linestyle=kwargs['linestyle'], color=kwargs['color'])


if __name__ == '__main__':
    # from cfg.config_maggs303 import *
    # kwargs_CFG = {'VISUALIZER_CFG'          : VISUALIZER_CFG,
    #               'RADAR_CFG_LIST'          : RADAR_CFG_LIST,
    #               'FRAME_POST_PROCESSOR_CFG': FRAME_POST_PROCESSOR_CFG,
    #               'DBSCAN_GENERATOR_CFG'    : DBSCAN_GENERATOR_CFG,
    #               'BGNOISE_FILTER_CFG'      : BGNOISE_FILTER_CFG,
    #               'HUMAN_TRACKING_CFG'      : HUMAN_TRACKING_CFG,
    #               'HUMAN_OBJECT_CFG'        : HUMAN_OBJECT_CFG}
    #
    # # Load database
    # with open('../data/Maggs_303/2perple_AllRadar_Jan-25-16-50-55', 'rb') as file:
    #     data = pickle.load(file)
    #
    # vis = Visualizer(data[80:], interval=0.05, image_output_enable=False, **kwargs_CFG)
    # vis.run()
    #
    # with open(f'obj_path', 'wb') as file:
    #     pickle.dump(vis.obj_path_saved, file)

    # from cfg.config_maggs307 import *
    # kwargs_CFG = {'VISUALIZER_CFG'          : VISUALIZER_CFG,
    #               'RADAR_CFG_LIST'          : RADAR_CFG_LIST,
    #               'FRAME_POST_PROCESSOR_CFG': FRAME_POST_PROCESSOR_CFG,
    #               'DBSCAN_GENERATOR_CFG'    : DBSCAN_GENERATOR_CFG,
    #               'BGNOISE_FILTER_CFG'      : BGNOISE_FILTER_CFG,
    #               'HUMAN_TRACKING_CFG'      : HUMAN_TRACKING_CFG,
    #               'HUMAN_OBJECT_CFG'        : HUMAN_OBJECT_CFG}
    #
    # # Load database
    # with open('../data/Maggs_307/SZC/SZC_auto_RadarSeq_Jun-04-16-59-54', 'rb') as file:
    #     data = pickle.load(file)
    #
    # vis = Visualizer(data[0:], interval=0.05, image_output_enable=False, **kwargs_CFG)
    # vis.run()

    # from cfg.config_mvb340 import *
    # kwargs_CFG = {'VISUALIZER_CFG'          : VISUALIZER_CFG,
    #               'RADAR_CFG_LIST'          : RADAR_CFG_LIST,
    #               'FRAME_POST_PROCESSOR_CFG': FRAME_POST_PROCESSOR_CFG,
    #               'DBSCAN_GENERATOR_CFG'    : DBSCAN_GENERATOR_CFG,
    #               'BGNOISE_FILTER_CFG'      : BGNOISE_FILTER_CFG,
    #               'HUMAN_TRACKING_CFG'      : HUMAN_TRACKING_CFG,
    #               'HUMAN_OBJECT_CFG'        : HUMAN_OBJECT_CFG}
    #
    # # Load database
    # with open('../data/radar_placement/60deg_2', 'rb') as file:
    #     data = pickle.load(file)
    #
    # vis = Visualizer(data[400:], interval=0.05, image_output_enable=False, **kwargs_CFG)
    # vis.run()

    # from cfg.config_mvb340_3R import *
    # kwargs_CFG = {'VISUALIZER_CFG'          : VISUALIZER_CFG,
    #               'RADAR_CFG_LIST'          : RADAR_CFG_LIST,
    #               'FRAME_POST_PROCESSOR_CFG': FRAME_POST_PROCESSOR_CFG,
    #               'DBSCAN_GENERATOR_CFG'    : DBSCAN_GENERATOR_CFG,
    #               'BGNOISE_FILTER_CFG'      : BGNOISE_FILTER_CFG,
    #               'HUMAN_TRACKING_CFG'      : HUMAN_TRACKING_CFG,
    #               'HUMAN_OBJECT_CFG'        : HUMAN_OBJECT_CFG}
    #
    # # Load database
    # with open('../data/fall_posture/4', 'rb') as file:
    #     data = pickle.load(file)
    #
    # vis = Visualizer(data[:], interval=0.05, image_output_enable=False, **kwargs_CFG)
    # vis.run()

    # from data.fall_detection.config_maggs307 import *
    # kwargs_CFG = {'VISUALIZER_CFG'          : VISUALIZER_CFG,
    #               'RADAR_CFG_LIST'          : RADAR_CFG_LIST,
    #               'FRAME_POST_PROCESSOR_CFG': FRAME_POST_PROCESSOR_CFG,
    #               'DBSCAN_GENERATOR_CFG'    : DBSCAN_GENERATOR_CFG,
    #               'BGNOISE_FILTER_CFG'      : BGNOISE_FILTER_CFG,
    #               'HUMAN_TRACKING_CFG'      : HUMAN_TRACKING_CFG,
    #               'HUMAN_OBJECT_CFG'        : HUMAN_OBJECT_CFG}
    #
    # # Load database
    # with open('../data/fall_detection/test_auto_RadarSeq_Jun-10-15-48-31', 'rb') as file:
    #     data = pickle.load(file)
    #
    # vis = Visualizer(data[:], interval=0.1, image_output_enable=False, **kwargs_CFG)
    # vis.run()
    #
    # with open(f'./obj_path', 'wb') as file:
    #     pickle.dump(vis.obj_path_saved, file)

    # from data.multiple_tracking.Tpeople.config_maggs307 import *
    # kwargs_CFG = {'VISUALIZER_CFG'          : VISUALIZER_CFG,
    #               'RADAR_CFG_LIST'          : RADAR_CFG_LIST,
    #               'FRAME_POST_PROCESSOR_CFG': FRAME_POST_PROCESSOR_CFG,
    #               'DBSCAN_GENERATOR_CFG'    : DBSCAN_GENERATOR_CFG,
    #               'BGNOISE_FILTER_CFG'      : BGNOISE_FILTER_CFG,
    #               'HUMAN_TRACKING_CFG'      : HUMAN_TRACKING_CFG,
    #               'HUMAN_OBJECT_CFG'        : HUMAN_OBJECT_CFG}
    #
    # # Load database
    # with open('../data/multiple_tracking/Tpeople/test_auto_RadarSeq_Mar-31-16-25-48', 'rb') as file:
    #     data = pickle.load(file)
    #
    # vis = Visualizer(data[2000:], interval=0.01, image_output_enable=False, **kwargs_CFG)
    # vis.run()
    #
    # with open(f'./obj_path', 'wb') as file:
    #     pickle.dump(vis.obj_path_saved, file)

    from cfg.config_mvb501 import *
    kwargs_CFG = {'VISUALIZER_CFG'          : VISUALIZER_CFG,
                  'RADAR_CFG_LIST'          : RADAR_CFG_LIST,
                  'FRAME_POST_PROCESSOR_CFG': FRAME_POST_PROCESSOR_CFG,
                  'DBSCAN_GENERATOR_CFG'    : DBSCAN_GENERATOR_CFG,
                  'BGNOISE_FILTER_CFG'      : BGNOISE_FILTER_CFG,
                  'HUMAN_TRACKING_CFG'      : HUMAN_TRACKING_CFG,
                  'HUMAN_OBJECT_CFG'        : HUMAN_OBJECT_CFG}

    # Load database
    with open('../data/MVB_501/2023_Nov/test_manual_RadarSeq_Nov-18-16-54-36', 'rb') as file:
        data = pickle.load(file)

    vis = Visualizer(data[:], interval=0.01, image_output_enable=False, **kwargs_CFG)
    vis.run()


