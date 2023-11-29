"""
Designed for data visualization, abbr. VIS
"""
import math
import queue
import time
from datetime import datetime
from multiprocessing import Manager

import matplotlib
import numpy as np
import winsound
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator

from Library.frame_post_processor import FramePProcessor

RP_colormap = ['C5', 'C7', 'C8']  # the colormap for radar raw points
ES_colormap = ['lavender', 'thistle', 'violet', 'darkorchid', 'indigo']  # the colormap for radar energy strength
OS_colormap = ['grey', 'green', 'gold', 'red']  # the colormap for object status


class Visualizer:
    def __init__(self, run_flag, radar_rd_queue_list, shared_param_dict, **kwargs_CFG):
        """
        get shared values and queues
        """
        self.run_flag = run_flag
        # radar rawdata queue list
        self.radar_rd_queue_list = radar_rd_queue_list
        # shared params
        try:
            self.save_queue = shared_param_dict['save_queue']
        except:
            self.save_queue = Manager().Queue(maxsize=0)
        self.mansave_flag = shared_param_dict['mansave_flag']
        self.autosave_flag = shared_param_dict['autosave_flag']

        """
        pass config static parameters
        """
        """ module own config """
        VIS_CFG = kwargs_CFG['VISUALIZER_CFG']
        self.dimension = VIS_CFG['dimension']
        self.VIS_xlim = VIS_CFG['VIS_xlim']
        self.VIS_ylim = VIS_CFG['VIS_ylim']
        self.VIS_zlim = VIS_CFG['VIS_zlim']

        self.auto_inactive_skip_frame = VIS_CFG['auto_inactive_skip_frame']

        """ other configs """
        self.MANSAVE_ENABLE = kwargs_CFG['MANSAVE_ENABLE']
        self.AUTOSAVE_ENABLE = kwargs_CFG['AUTOSAVE_ENABLE']
        self.RDR_CFG_LIST = kwargs_CFG['RADAR_CFG_LIST']

        """
        self content
        """
        self.fpp = FramePProcessor(**kwargs_CFG)  # call other class

        # setup for matplotlib plot
        matplotlib.use('TkAgg')  # set matplotlib backend
        plt.rcParams['toolbar'] = 'None'  # disable the toolbar
        # create a figure
        self.fig = plt.figure()
        # adjust figure position
        mngr = plt.get_current_fig_manager()
        mngr.window.wm_geometry('+30+30')
        # draws a completely frameless window
        win = plt.gcf().canvas.manager.window
        win.overrideredirect(1)
        # interactive mode on, no need plt.show()
        plt.ion()

        self._log('Start...')

    def run(self):
        if self.dimension == '2D':
            # create a plot
            ax1 = self.fig.add_subplot(111)

            while self.run_flag.value:
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
                self._update_canvas(ax1)

        elif self.dimension == '3D':
            # create a plot
            ax1 = self.fig.add_subplot(111, projection='3d')

            spin = 0
            while self.run_flag.value:
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
                self._update_canvas(ax1)
        else:
            while self.run_flag.value:
                for q in self.radar_rd_queue_list:
                    _ = q.get(block=True, timeout=5)

    def _update_canvas(self, ax1):
        # draw radar point
        for RDR_CFG in self.RDR_CFG_LIST:
            self._plot(ax1, [RDR_CFG['pos_offset'][0]], [RDR_CFG['pos_offset'][1]], [RDR_CFG['pos_offset'][2]], marker='o', color='DarkRed')

        # get values from queues of all radars
        val_data_allradar = np.ndarray([0, 5], dtype=np.float16)
        ES_noise_allradar = np.ndarray([0, 5], dtype=np.float16)
        save_data_frame = {}
        try:
            # adaptive short skip when no object is detected
            if self.AUTOSAVE_ENABLE and not self.autosave_flag.value:
                for _ in range(self.auto_inactive_skip_frame):
                    for q in self.radar_rd_queue_list:
                        _ = q.get(block=True, timeout=5)

            for i, RDR_CFG in enumerate(self.RDR_CFG_LIST):
                data_1radar = self.radar_rd_queue_list[i].get(block=True, timeout=5)
                # apply ES and speed filter for each radar channel
                val_data, ES_noise = self.fpp.DP_ES_Speed_filter(data_1radar, RDR_CFG['ES_threshold'])
                val_data_allradar = np.concatenate([val_data_allradar, val_data])
                ES_noise_allradar = np.concatenate([ES_noise_allradar, ES_noise])
                # # draw raw point cloud
                # self._plot(ax1, val_data[:, 0], val_data[:, 1], val_data[:, 2], marker='.', linestyle='None', color=RP_colormap[i])

                # save the frames
                save_data_frame[RDR_CFG['name']] = data_1radar

        except queue.Empty:
            self._log('Raw Data Queue Empty.')
            self.run_flag.value = False

        # put frame and time into queue
        self.save_queue.put({'source'   : 'radar',
                             'data'     : save_data_frame,
                             'timestamp': time.time(),
                             })

        # apply global boundary filter
        val_data_allradar = self.fpp.FPP_boundary_filter(val_data_allradar)
        ES_noise_allradar = self.fpp.FPP_boundary_filter(ES_noise_allradar)
        # apply global energy strength filter
        val_data_allradar, global_ES_noise = self.fpp.FPP_ES_Speed_filter(val_data_allradar)
        # apply background noise filter
        val_data_allradar = self.fpp.BGN_filter(val_data_allradar)

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
            obj_status_list = []
            for person in self.fpp.TRK_people_list:
                obj_cp, obj_status = person.get_info()
                obj_status_list.append(obj_status)
                self._plot(ax1, obj_cp[:, 0], obj_cp[:, 1], obj_cp[:, 2], marker='o', color=OS_colormap[obj_status])
                if obj_status == 3:  # warning when object falls
                    winsound.Beep(1000, 20)

            # auto save based on object detection
            if self.AUTOSAVE_ENABLE:
                if max(obj_status_list) >= 0:  # object detected
                    # activate flag
                    self.autosave_flag.value = True
                else:
                    # deactivate flag
                    self.autosave_flag.value = False

        # wait at the end of each loop
        # plt.pause(0.001)
        self._detect_key_press(0.001)

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
                ax.plot(x, y, fmt)
            elif self.dimension == '3D':
                ax.plot3D(x, y, z, fmt)
        else:  # if para is using
            for i in ['marker', 'linestyle', 'color']:
                if not (i in kwargs):
                    kwargs[i] = 'None'
            if self.dimension == '2D':
                ax.plot(x, y, marker=kwargs['marker'], linestyle=kwargs['linestyle'], color=kwargs['color'])
            elif self.dimension == '3D':
                ax.plot3D(x, y, z, marker=kwargs['marker'], linestyle=kwargs['linestyle'], color=kwargs['color'])

    def _detect_key_press(self, timeout):  # error caused if the key is pressed at very beginning (first loop)
        keyPressed = plt.waitforbuttonpress(timeout=timeout)  # detect whether key is pressed or not
        plt.gcf().canvas.mpl_connect('key_press_event', self._press)  # detect which key is pressed
        if keyPressed:
            if self.the_key == 'escape':
                self.run_flag.value = False
            # manual save trigger
            if self.MANSAVE_ENABLE:
                if self.the_key == '+':
                    # activate flag
                    self.mansave_flag.value = 'image'
                elif self.the_key == '0':
                    # activate flag
                    self.mansave_flag.value = 'video'

    def _press(self, event):
        self.the_key = event.key

    def _log(self, txt):  # print with device name
        print(f'[{self.__class__.__name__}]\t{txt}')

    def __del__(self):
        plt.close(self.fig)
        self._log(f"Closed. Timestamp: {datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
        self.run_flag.value = False
