"""
Human object, abbr. OBJ
"""
import time
from collections import deque
from math import hypot

import numpy as np
from scipy import stats


class HumanObject:
    def __init__(self, name, **kwargs_CFG):
        """
        pass config static parameters
        """
        """ module own config """
        OBJ_CFG = kwargs_CFG['HUMAN_OBJECT_CFG']
        # store the previous data
        self.obj_cp_deque = deque([], OBJ_CFG['obj_deque_length'])
        self.obj_size_deque = deque([], OBJ_CFG['obj_deque_length'])
        self.obj_status_deque = deque(np.ones(OBJ_CFG['obj_deque_length'], dtype=int), OBJ_CFG['obj_deque_length'])  # set as standing as default
        self.obj_speed_deque = deque([], OBJ_CFG['obj_deque_length'])
        self.obj_timestamp_deque = deque([], OBJ_CFG['obj_deque_length'])
        self.obj_height_dict = {'height': 0, 'count': 0, }  # fixed property

        # the threshold for data update
        self.dis_diff_threshold = OBJ_CFG['dis_diff_threshold']['threshold']
        self.dis_diff_threshold_dr = OBJ_CFG['dis_diff_threshold']['dynamic_ratio']
        self.size_diff_threshold = OBJ_CFG['size_diff_threshold']
        self.expect_pos = OBJ_CFG['expect_pos']
        self.expect_shape = OBJ_CFG['expect_shape']

        self.sub_possibility_proportion = OBJ_CFG['sub_possibility_proportion']
        self.inactive_timeout = OBJ_CFG['inactive_timeout']
        self.obj_delete_timeout = OBJ_CFG['obj_delete_timeout']

        # for object entering the scene
        self.fuzzy_boundary_enter = OBJ_CFG['fuzzy_boundary_enter']
        self.fuzzy_boundary_threshold = OBJ_CFG['fuzzy_boundary_threshold']
        self.scene_xlim = OBJ_CFG['scene_xlim']
        self.scene_ylim = OBJ_CFG['scene_ylim']
        self.scene_zlim = OBJ_CFG['scene_zlim']

        # for object status
        self.standing_sitting_threshold = OBJ_CFG['standing_sitting_threshold']
        self.sitting_lying_threshold = OBJ_CFG['sitting_lying_threshold']

        # get mean position and other info from last serval data
        self.get_fuzzy_pos_No = OBJ_CFG['get_fuzzy_pos_No']
        self.get_fuzzy_status_No = OBJ_CFG['get_fuzzy_status_No']

        """
        self content
        """
        self.name = 'obj_' + name

    def check_clus_possibility(self, obj_cp, obj_size):
        """
        :param obj_cp: (ndarray) channels(3), xyz, central point of cluster
        :param obj_size: (ndarray) channels(3), xyz, size of cluster
        :return: (float) the possibility of the point whether it should be taken
        """
        # if no previous points saved
        if len(self.obj_cp_deque) == 0:
            dis_possibility = 0
            size_possibility = 0
            # enable central point starts around scene boundaries
            if self.fuzzy_boundary_enter and not self._boundary_fuzzy_area(obj_cp):
                return 0
        # if there are previous points
        else:
            # # use z-score
            # if len(self.obj_cp_deque) > 10:
            #     obj_cp_np = np.array(self.obj_cp_deque).reshape([-1, 3])
            #     obj_cp_np = np.concatenate([obj_cp_np, obj_cp])
            #     x = obj_cp_np[:, 0]
            #     y = obj_cp_np[:, 1]
            #     z = obj_cp_np[:, 2]
            #     dis_possibility_x = np.exp(-abs(stats.zscore(x)[-1]))
            #     dis_possibility_y = np.exp(-abs(stats.zscore(y)[-1]))
            #     dis_possibility_z = np.exp(-abs(stats.zscore(z)[-1]))
            #     dis_possibility = sum(np.array([dis_possibility_x, dis_possibility_y, dis_possibility_z]) * [0.3, 0.3, 0.4])
            #
            #     obj_size_np = np.array(self.obj_size_deque).reshape([-1, 3])
            #     obj_size_np = np.concatenate([obj_size_np, obj_size])
            #     x = obj_size_np[:, 0]
            #     y = obj_size_np[:, 1]
            #     z = obj_size_np[:, 2]
            #     size_possibility_x = np.exp(-abs(stats.zscore(x)[-1]))
            #     size_possibility_y = np.exp(-abs(stats.zscore(y)[-1]))
            #     size_possibility_z = np.exp(-abs(stats.zscore(z)[-1]))
            #     size_possibility = sum(np.array([dis_possibility_x, dis_possibility_y, dis_possibility_z]) * [0.2, 0.2, 0.6])

            # based on distance between the current and the last
            diff = obj_cp - self.obj_cp_deque[-1]
            dis_diff = hypot(diff[0], diff[1], diff[2])
            # get dynamic distance threshold by adding object speed of the last frame
            dyn_dis_threshold = self.dis_diff_threshold + abs(self.obj_speed_deque[-1]) * self.dis_diff_threshold_dr
            # get distance possibility
            if dis_diff < dyn_dis_threshold:  # if the current point is close enough to previous one
                dis_possibility = (dyn_dis_threshold - dis_diff) / dyn_dis_threshold
            else:
                return 0

            # based on the object size difference between the current and the last
            size_diff = abs(np.prod(obj_size) - np.prod(self.obj_size_deque[-1]))
            # get size possibility
            if size_diff < self.size_diff_threshold:
                size_possibility = (self.size_diff_threshold - size_diff) / self.size_diff_threshold
            else:
                return 0

        # get self_possibility
        # status = ['default', 'standing', 'sitting', 'lying'][self.get_info()[1]]
        # pos_possibility, shape_possibility = self._get_self_possibility(obj_cp, obj_size, self.expect_pos[status], self.expect_shape[status])
        pos_possibility, shape_possibility = self._get_self_possibility(obj_cp, obj_size, self.expect_pos['default'], self.expect_shape['default'])

        # calculate total possibility
        point_taken_possibility = sum(np.array([dis_possibility, size_possibility, pos_possibility, shape_possibility]) * np.array(self.sub_possibility_proportion))
        return point_taken_possibility

    def update_info(self, obj, obj_cp, obj_size):
        """
        :param obj: (ndarray) data_numbers(n) * channels(5), cluster data points
        :param obj_cp: (ndarray) channels(3), xyz, central point of cluster
        :param obj_size: (ndarray) channels(3), xyz, size of cluster
        """
        # update the cluster info including central point, size as an object
        self.obj_cp_deque.append(obj_cp)
        self.obj_size_deque.append(obj_size)
        self._update_height(obj_cp, obj_size)
        self.obj_status_deque.append(self._get_status(obj_cp, obj_size))
        self.obj_speed_deque.append(self._get_speed(obj))
        self.obj_timestamp_deque.append(time.time())

        # # adjust data deque based on outlier detection
        # 'data_adjusted_range': None,
        # # based on the distribution between current point and all sequence
        # data = np.squeeze(np.array(list(self.obj_cp_deque)))
        # mean_x = np.mean(data[:, 0])
        # std = np.std(data[:, 0])

    # this function runs every frame loop
    def get_info(self):
        """
        :return: obj_cp (ndarray) data_numbers(1) * channels(3), the central point of the object (this shape is for plot when no points)
                 obj_size (ndarray) data_numbers(1) * channels(3)
                 obj_status (int) 0-losing target, 1-standing, 2-sitting, 3-lying
        """
        # default is empty
        obj_cp = np.ndarray([0, 3], dtype=np.float16)
        obj_size = np.ndarray([0, 3], dtype=np.float16)
        obj_status = -1

        # if there are previous points
        if len(self.obj_cp_deque) > 0:
            # check if long time no update
            if self._check_timeout(self.obj_delete_timeout):
                self.obj_cp_deque.clear()
                self.obj_size_deque.clear()
                self.obj_status_deque = deque(np.ones(self.obj_status_deque.maxlen, dtype=int), self.obj_status_deque.maxlen)  # set as standing as default
                self.obj_speed_deque.clear()
                self.obj_timestamp_deque.clear()

                self.obj_height_dict = {'height': 0, 'count': 0, }
            else:
                # get obj position
                if self.get_fuzzy_pos_No:  # get comprehensive info based on previous value sequence
                    # calculate the mean position from the last few data
                    obj_cp_np = np.array(list(self.obj_cp_deque))[-self.get_fuzzy_pos_No:]
                    obj_cp = np.array([[np.mean(obj_cp_np[:, 0]), np.mean(obj_cp_np[:, 1]), np.mean(obj_cp_np[:, 2])]])
                else:  # get latest info
                    obj_cp = self.obj_cp_deque[-1][np.newaxis, :]

                # get obj size
                if self.get_fuzzy_pos_No:  # get comprehensive info based on previous value sequence
                    obj_size_np = np.array(list(self.obj_size_deque))[-self.get_fuzzy_pos_No:]
                    obj_size = np.array([[np.mean(obj_size_np[:, 0]), np.mean(obj_size_np[:, 1]), np.mean(obj_size_np[:, 2])]])
                else:  # get latest info
                    obj_size = self.obj_size_deque[-1][np.newaxis, :]

                # get obj status
                if self.get_fuzzy_status_No:  # get comprehensive info based on previous value sequence
                    # sort the status labels, high to low
                    unique, counts = np.unique(list(self.obj_status_deque)[-self.get_fuzzy_status_No:], return_counts=True)
                    unique_sorted = [i[0] for i in sorted(tuple(zip(unique, counts)), key=lambda item: item[1], reverse=True)]
                    obj_status = unique_sorted[0]  # use the label with maximum number
                else:  # get latest info
                    obj_status = self.obj_status_deque[-1]

                # check if temporarily lose the target
                if self._check_timeout(self.inactive_timeout):
                    obj_status = 0

        return obj_cp, obj_size, obj_status

    def _get_status(self, obj_cp, obj_size):
        """
        :return: (int) 0-unknown, 1-standing, 2-sitting, 3-lying
        """
        # current_height = obj_cp[2] + obj_size[2] / 2
        # current_volume = np.prod(obj_size)

        status_possibility_list = []
        for s in ['standing', 'sitting', 'lying']:
            pos_possibility, shape_possibility = self._get_self_possibility(obj_cp, obj_size, self.expect_pos[s], self.expect_shape[s])
            status_possibility_list.append(sum(np.array([pos_possibility, shape_possibility]) * np.array(self.sub_possibility_proportion)[2:4]))
        status = status_possibility_list.index(max(status_possibility_list)) + 1

        # # for big cluster cube
        # if current_volume > 0.25 and obj_size[2] > 1:
        #     if obj_cp[2] >= self.standing_sitting_threshold and current_height >= self.obj_height_dict['height'] * 0.8:
        #         status = 1  # standing
        # # for big cluster cube
        # elif current_volume > 0.04 and obj_size[2] >= 0.2:
        #     if self.sitting_lying_threshold <= obj_cp[2] < self.standing_sitting_threshold or self.obj_height_dict['height'] * 0.25 <= current_height < self.obj_height_dict['height'] * 0.8:
        #         status = 2  # sitting
        # elif obj_cp[2] < self.sitting_lying_threshold or current_height < self.obj_height_dict['height'] * 0.25:
        #     status = 3  # lying
        return status

    def _get_speed(self, data_points):
        """
        :param data_points: (ndarray) data_numbers(n) * channels(5), cluster data points
        :return: (float) cluster average speed
        """
        # get cluster average speed from all points speed excluding 0s
        speed_np = data_points[:, 3]
        speed_np = speed_np[speed_np != 0]  # filter 0 values
        if speed_np.size == 0:
            speed = 0
        else:
            speed = float(np.mean(speed_np))
        return speed

    def _update_height(self, obj_cp, obj_size):
        current_height = obj_cp[2] + obj_size[2] / 2
        if current_height > self.obj_height_dict['height']:
            self.obj_height_dict['height'] = current_height

        # height_range = (1.7, 1.9)
        # current_height = obj_cp[2] + obj_size[2] / 2
        # current_count = self.obj_height_dict['count']
        #
        # if height_range[0] < current_height < height_range[1]:
        #     self.obj_height_dict['height'] = (self.obj_height_dict['height'] * current_count + current_height) / (current_count + 1)
        #     self.obj_height_dict['count'] = current_count + 1
        #     print(obj_cp, obj_size, current_height, current_count, self.obj_height_dict)

    def _get_self_possibility(self, obj_cp, obj_size, expect_pos, expect_shape):
        """
        :param obj_cp: (ndarray) channels(3), xyz, central point of cluster
        :param obj_size: (ndarray) channels(3), xyz, size of cluster
        :param expect_pos: (list), xyz, expect central point of cluster
        :param expect_shape: (list), xyz, expect size of cluster
        :return: pos_possibility: (float)
                 shape_possibility: (float)
        """
        # based on the object current position
        pos_diff_list = []
        for i in range(len(expect_pos)):
            if expect_pos[i]:
                pos_diff_axis = abs(obj_cp[i] - expect_pos[i])
            else:
                pos_diff_axis = 0
            pos_diff_list.append(pos_diff_axis)
        pos_diff = hypot(pos_diff_list[0], pos_diff_list[1], pos_diff_list[2])
        pos_possibility = np.exp(-pos_diff)

        # based on the object current shape when different status
        shape_diff_list = []
        for i in range(len(expect_shape)):
            if expect_shape[i]:
                shape_diff_axis = abs(obj_size[i] - expect_shape[i])
            else:
                shape_diff_axis = 0
            shape_diff_list.append(shape_diff_axis)
        shape_diff = np.array(shape_diff_list, dtype=np.float16)
        shape_possibility = sum(np.exp(-shape_diff) * [0.2, 0.2, 0.6])

        return pos_possibility, shape_possibility

    def _boundary_fuzzy_area(self, data_point):
        """
        :param data_point: (ndarray) channels(3), the central point of the object (this shape is for plot when no points)
        :return: (boolean) F-not in the boundary fuzzy area, 1-in the boundary fuzzy area
        """
        x_fuzzy1 = (self.scene_xlim[0] - self.fuzzy_boundary_threshold, self.scene_xlim[0] + self.fuzzy_boundary_threshold)
        x_fuzzy2 = (self.scene_xlim[1] - self.fuzzy_boundary_threshold, self.scene_xlim[1] + self.fuzzy_boundary_threshold)
        y_fuzzy1 = (self.scene_ylim[0] - self.fuzzy_boundary_threshold, self.scene_ylim[0] + self.fuzzy_boundary_threshold)
        y_fuzzy2 = (self.scene_ylim[1] - self.fuzzy_boundary_threshold, self.scene_ylim[1] + self.fuzzy_boundary_threshold)
        z_fuzzy1 = (self.scene_zlim[0] - self.fuzzy_boundary_threshold, self.scene_zlim[0] + self.fuzzy_boundary_threshold)
        z_fuzzy2 = (self.scene_zlim[1] - self.fuzzy_boundary_threshold, self.scene_zlim[1] + self.fuzzy_boundary_threshold)

        x = data_point[0]
        y = data_point[1]
        z = data_point[2]
        if (x_fuzzy1[0] <= x < x_fuzzy1[1]) or (x_fuzzy2[0] <= x < x_fuzzy2[1]) or \
                (y_fuzzy1[0] <= y < y_fuzzy1[1]) or (y_fuzzy2[0] <= y < y_fuzzy2[1]) or \
                (z_fuzzy1[0] <= z < z_fuzzy1[1]) or (z_fuzzy2[0] <= z < z_fuzzy2[1]):
            in_boundary_fuzzy_area = True
        else:
            in_boundary_fuzzy_area = False
        return in_boundary_fuzzy_area

    def _check_timeout(self, time_threshold):
        """
        :param time_threshold: (float/int) unit is second
        :return: (boolean) is timeout or not
        """
        if (time.time() - self.obj_timestamp_deque[-1]) >= time_threshold:
            timeout = True
        else:
            timeout = False
        return timeout
