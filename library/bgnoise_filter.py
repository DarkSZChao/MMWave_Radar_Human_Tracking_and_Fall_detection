"""
Background noise filter, abbr. BGN
"""
from collections import deque

import numpy as np
from sklearn.cluster import DBSCAN

from library.data_processor import DataProcessor


class BGNoiseFilter(DataProcessor):
    def __init__(self, **kwargs_CFG):
        """
        pass config static parameters
        """
        """ module own config """
        BGN_CFG = kwargs_CFG['BGNOISE_FILTER_CFG']
        self.BGN_enable = BGN_CFG['BGN_enable']

        # get BGN processing para
        self.BGN_deque = deque([], BGN_CFG['BGN_deque_length'])
        self.BGN_accept_ES_threshold = BGN_CFG['BGN_accept_ES_threshold']
        self.BGN_filter_ES_threshold = BGN_CFG['BGN_filter_ES_threshold']
        self.BGN_DBS_window_step = BGN_CFG['BGN_DBS_window_step']
        self.BGN_DBS_eps = BGN_CFG['BGN_DBS_eps']
        self.BGN_DBS_min_samples = BGN_CFG['BGN_DBS_min_samples']
        self.BGN_cluster_tf = BGN_CFG['BGN_cluster_tf']
        self.BGN_cluster_xextension = BGN_CFG['BGN_cluster_xextension']
        self.BGN_cluster_yextension = BGN_CFG['BGN_cluster_yextension']
        self.BGN_cluster_zextension = BGN_CFG['BGN_cluster_zextension']

        """
        self content
        """
        self.BGN_cluster_boundary = np.ndarray([0, 6], dtype=np.float16)  # (xmin, xmax, ymin, ymax, zmin, zmax)

        """
        inherit father class __init__ para
        """
        super().__init__(**kwargs_CFG)

    def BGN_filter(self, data_points):
        for c in self.BGN_cluster_boundary:
            xboundary = (c[0], c[1])
            yboundary = (c[2], c[3])
            zboundary = (c[4], c[5])
            # identify the noise inside each cluster
            noise, _ = self.DP_np_filter(data_points, axis=0, range_lim=xboundary, mode=0)
            noise, _ = self.DP_np_filter(noise, axis=1, range_lim=yboundary, mode=0)
            noise, _ = self.DP_np_filter(noise, axis=2, range_lim=zboundary, mode=0)
            # identify the noise with no speed and low ES
            noise, _ = self.DP_np_filter(noise, axis=3, range_lim=0)
            noise, _ = self.DP_np_filter(noise, axis=4, range_lim=self.BGN_filter_ES_threshold)
            # remove noise points specifically
            data_points = self.DP_np_2D_row_removal(data_points, noise)
        return data_points

    def BGN_get_filter_area(self):
        BGN_block_list = []
        for bb in self.BGN_cluster_boundary:
            BGN_block_list.append(self.DP_cubehull(None, (bb[0], bb[1]), (bb[2], bb[3]), (bb[4], bb[5])))
        return BGN_block_list

    def BGN_update(self, bg_noise):
        # identify the noise with no speed and low ES
        bg_noise, _ = self.DP_np_filter(bg_noise, axis=3, range_lim=0)
        bg_noise, _ = self.DP_np_filter(bg_noise, axis=4, range_lim=self.BGN_accept_ES_threshold)

        # append the BG points
        self.BGN_deque.append(bg_noise)
        BGN_points = np.concatenate(self.BGN_deque)

        # reach the window length
        if len(self.BGN_deque) == self.BGN_deque.maxlen and len(BGN_points) > 1000:
            labels = DBSCAN(eps=self.BGN_DBS_eps, min_samples=self.BGN_DBS_min_samples).fit_predict(BGN_points[:, 0:3])  # only feed xyz coords
            # filter out DBSCAN noise
            valid_points = BGN_points[labels != -1]
            valid_labels = labels[labels != -1]

            # select the clusters with point number more than threshold
            unique, counts = np.unique(valid_labels, return_counts=True)
            unique = unique[counts > len(BGN_points) * self.BGN_cluster_tf]
            for u in unique:
                cluster = valid_points[valid_labels == u]
                xyz = self.DP_boundary_calculator(cluster, axis=range(3))  # get boundary info of cluster
                xyz_extended = np.array([xyz[0][0] - self.BGN_cluster_xextension,
                                         xyz[0][1] + self.BGN_cluster_xextension,
                                         xyz[1][0] - self.BGN_cluster_yextension,
                                         xyz[1][1] + self.BGN_cluster_yextension,
                                         xyz[2][0] - self.BGN_cluster_zextension,
                                         xyz[2][1] + self.BGN_cluster_zextension], dtype=np.float16)

                # check whether this cluster should be saved by comparing with the database clusters' area
                self.BGN_cluster_boundary = self._check_area_overlap(self.BGN_cluster_boundary, xyz_extended)

            # delete previous points for sliding the window
            _ = [self.BGN_deque.popleft() for i in range(self.BGN_DBS_window_step)]

    def _check_area_overlap(self, database, data_point):
        """
        check whether incoming cluster cube is inside one of the stored cluster cubes,
        if it is inside then don't save,
        if it is cross then save,
        if it contains then save and delete the clusters which are contained
        :param database: (ndarray) data_numbers(n) * channels(6), (xmin, xmax, ymin, ymax, zmin, zmax) for each cluster stored
        :param data_point: (ndarray) channels(6), incoming cluster boundary info
        :return: database: (ndarray) data_numbers(n) * channels(6), updated database
        """
        # check if data_point is inside any of database
        idx0, _ = self.DP_get_idx_bool(database, axis=0, range_lim=(None, data_point[0]), mode=0)  # xmin
        idx1, _ = self.DP_get_idx_bool(database, axis=1, range_lim=(data_point[1], None), mode=0)  # xmax
        idx2, _ = self.DP_get_idx_bool(database, axis=2, range_lim=(None, data_point[2]), mode=0)  # ymin
        idx3, _ = self.DP_get_idx_bool(database, axis=3, range_lim=(data_point[3], None), mode=0)  # ymax
        idx4, _ = self.DP_get_idx_bool(database, axis=4, range_lim=(None, data_point[4]), mode=0)  # zmin
        idx5, _ = self.DP_get_idx_bool(database, axis=5, range_lim=(data_point[5], None), mode=0)  # zmax
        idx_inside = (idx0 & idx1 & idx2 & idx3 & idx4 & idx5).any()  # if any points found

        # check if data_point contains any of database
        idx0, _ = self.DP_get_idx_bool(database, axis=0, range_lim=(data_point[0], None), mode=0)  # xmin
        idx1, _ = self.DP_get_idx_bool(database, axis=1, range_lim=(None, data_point[1]), mode=0)  # xmax
        idx2, _ = self.DP_get_idx_bool(database, axis=2, range_lim=(data_point[2], None), mode=0)  # ymin
        idx3, _ = self.DP_get_idx_bool(database, axis=3, range_lim=(None, data_point[3]), mode=0)  # ymax
        idx4, _ = self.DP_get_idx_bool(database, axis=4, range_lim=(data_point[4], None), mode=0)  # zmin
        idx5, _ = self.DP_get_idx_bool(database, axis=5, range_lim=(None, data_point[5]), mode=0)  # zmax
        idx_contains_np = idx0 & idx1 & idx2 & idx3 & idx4 & idx5
        idx_contains = idx_contains_np.any()  # if any points found

        # update database, if both idx_inside and idx_contains are True, then data_point is identical with one of database, then pass
        if idx_inside:
            pass
        elif idx_contains:
            database = database[~idx_contains_np]
            database = np.concatenate([database, data_point[np.newaxis, :]])
        else:
            database = np.concatenate([database, data_point[np.newaxis, :]])
        return database


if __name__ == '__main__':
    bgn = BGNoiseFilter()
    datab = np.array([[0.34, 0.34, 3.53, 3.58, -0.40, -0.34],
                      [0.34, 0.34, 3.53, 3.56, -0.40, -0.04],
                      ])
    data1 = np.array([0.34, 0.34, 3.53, 3.58, -0.40, -0.34])
    data2 = np.array([0.34, 0.34, 3.53, 3.56, -0.40, -0.04])
    data3 = np.array([0.34, 0.34, 3.53, 3.56, -0.40, -0.14])
    data4 = np.array([0.34, 0.34, 3.53, 3.56, -0.40, -0.04])
    datab = bgn._check_area_overlap(datab, data1)
    print(datab)
    datab = bgn._check_area_overlap(datab, data2)
    print(datab)
    datab = bgn._check_area_overlap(datab, data3)
    print(datab)
    datab = bgn._check_area_overlap(datab, data4)
    print(datab)
    pass
