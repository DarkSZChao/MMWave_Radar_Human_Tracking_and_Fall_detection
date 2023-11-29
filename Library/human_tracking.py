"""
Human tracking, abbr. TRK
"""
from collections import deque
from math import hypot

import numpy as np

from Library.data_processor import DataProcessor
from Library.human_object import HumanObject


class HumanTracking(DataProcessor):
    def __init__(self, **kwargs_CFG):
        """
        pass config static parameters
        """
        """ module own config """
        TRK_CFG = kwargs_CFG['HUMAN_TRACKING_CFG']
        self.TRK_enable = TRK_CFG['TRK_enable']

        # get TRK processing para
        self.TRK_people_list = []
        for i in range(TRK_CFG['TRK_obj_bin_number']):  # create objects based on the maximum number
            self.TRK_people_list.append(HumanObject(name=str(i), **kwargs_CFG))
        self.TRK_poss_clus_deque = deque([], TRK_CFG['TRK_poss_clus_deque_length'])
        self.TRK_redundant_clus_remove_cp_dis = TRK_CFG['TRK_redundant_clus_remove_cp_dis']

        """inherit father class __init__ para"""
        super().__init__()

    def TRK_update_poss_matrix(self, valid_points_list):
        # stack cluster valid points
        self.TRK_poss_clus_deque.append(valid_points_list)

        # get stacked cluster valid points and remove list nesting
        poss_clus_list = self.DP_list_nesting_remover(list(self.TRK_poss_clus_deque))

        # initial values
        obj_cp_total = np.ndarray([0, 3], dtype=np.float16)  # (cp_pos_x, cp_pos_y, cp_pos_z)
        obj_size_total = np.ndarray([0, 3], dtype=np.float16)  # (size_x, size_y, size_z)
        # get central point and size for all clusters
        for clus_valid_points in poss_clus_list:
            x, y, z = self.DP_boundary_calculator(clus_valid_points, axis=range(3))
            obj_cp = np.array([[sum(x) / 2, sum(y) / 2, sum(z) / 2]], dtype=np.float16)
            obj_size = np.concatenate([np.diff(x), np.diff(y), np.diff(z)])[np.newaxis, :]
            obj_cp_total = np.concatenate([obj_cp_total, obj_cp])
            obj_size_total = np.concatenate([obj_size_total, obj_size])

        # calculate possibility matrix for each cluster and each object bin
        point_taken_poss_matrix = np.zeros([len(poss_clus_list), len(self.TRK_people_list)], dtype=np.float16)
        for c in range(len(poss_clus_list)):  # for each cluster
            for p in range(len(self.TRK_people_list)):  # for each object bin
                point_taken_poss_matrix[c, p] = self.TRK_people_list[p].check_clus_possibility(obj_cp_total[c], obj_size_total[c])

        # keep finding the global maximum value of the possibility matrix until no values above 0
        while point_taken_poss_matrix.size > 0 and np.max(point_taken_poss_matrix) > 0:
            max_index = divmod(np.argmax(point_taken_poss_matrix), point_taken_poss_matrix.shape[1])
            c = max_index[0]
            p = max_index[1]
            # append the central point and size to the corresponding object
            self.TRK_people_list[p].update_info(poss_clus_list[c], obj_cp_total[c], obj_size_total[c])

            # by setting the poss_matrix raw & column to 0 to remove redundant clusters closed to the updated one including itself, for multiple obj bin purpose
            obj_cp_used = obj_cp_total[c]
            for i in range(len(obj_cp_total)):
                diff = obj_cp_total[i] - obj_cp_used
                dis_diff = hypot(diff[0], diff[1])
                if dis_diff < self.TRK_redundant_clus_remove_cp_dis:
                    point_taken_poss_matrix[i, :] = 0
            point_taken_poss_matrix[:, p] = 0
