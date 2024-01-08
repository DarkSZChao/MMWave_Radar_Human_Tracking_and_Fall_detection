"""
DBSCAN generator, abbr. DBS
"""
import numpy as np
from sklearn.cluster import DBSCAN

from library.data_processor import DataProcessor


class DBSCANGenerator(DataProcessor):
    def __init__(self, **kwargs_CFG):
        """
        pass config static parameters
        """
        """ module own config """
        DBS_CFG = kwargs_CFG['DBSCAN_GENERATOR_CFG']
        self.DBS_CFG = DBS_CFG

        # get default DBSCAN para
        self.DBS_eps = DBS_CFG['Default']['DBS_eps']
        self.DBS_min_samples = DBS_CFG['Default']['DBS_min_samples']
        self.DBS_cp_pos_xlim = DBS_CFG['Default']['DBS_cp_pos_xlim']
        self.DBS_cp_pos_ylim = DBS_CFG['Default']['DBS_cp_pos_ylim']
        self.DBS_cp_pos_zlim = DBS_CFG['Default']['DBS_cp_pos_zlim']
        self.DBS_size_xlim = DBS_CFG['Default']['DBS_size_xlim']
        self.DBS_size_ylim = DBS_CFG['Default']['DBS_size_ylim']
        self.DBS_size_zlim = DBS_CFG['Default']['DBS_size_zlim']
        self.DBS_sort = DBS_CFG['Default']['DBS_sort']

        # get DBSCAN dynamic level listed in config
        self.DBS_dynamic_ES_list = []
        for para in DBS_CFG:
            if para.split('_')[0] == 'Dynamic':
                ES_level = int(para.split('_')[2])
                self.DBS_dynamic_ES_list.append(ES_level)

        """
        inherit father class __init__ para
        """
        super().__init__(**kwargs_CFG)

    def DBS_dynamic_ES(self, data_points):
        """
        :param data_points: (ndarray) data_numbers(n) * channels(5) for a dozen of data frames
        :return: vertices_list: (list-ndarray) list of data_numbers(n) * channels(3) for vertices of 3D hull
                 valid_points_list: (list-ndarray) list of data_numbers(n) * channels(c) for valid points
                 valid_points: (ndarray) data_numbers(n) * channels(c) for total valid points
                 noise: (ndarray) data_numbers(n) * channels(c) for total noise points
        """
        # initial values
        vertices_list_total = []
        valid_points_list_total = []
        valid_points_total = np.ndarray([0, 5], dtype=np.float16)
        noise_total = np.ndarray([0, 5], dtype=np.float16)

        if data_points.shape[0] > 0:
            # use ES_level_list to minimize the DBSCAN times and lower computation cost by avoiding calculate 2 times for 2 levels which have same data
            ES_level_list = []
            prev_ES_dp_No = 0
            for ES_level in self.DBS_dynamic_ES_list[::-1]:  # get reversed ES level list
                curr_ES_dp_No = len(self.DP_np_filter(data_points, axis=4, range_lim=(ES_level, None))[0])  # find how many points within this level
                if curr_ES_dp_No != prev_ES_dp_No:  # if the number of data points at curr level is diff with prev one
                    ES_level_list.append(ES_level)  # append this level
                    prev_ES_dp_No = curr_ES_dp_No  # update number for prev level

            # run DBSCAN multiple times with diff para
            # for ES_level in self.DBSCAN_dynamic_ES_list:
            for ES_level in ES_level_list:
                # set default DBSCAN para
                self.DBS_eps = self.DBS_CFG['Default']['DBS_eps']
                self.DBS_min_samples = self.DBS_CFG['Default']['DBS_min_samples']
                self.DBS_cp_pos_xlim = self.DBS_CFG['Default']['DBS_cp_pos_xlim']
                self.DBS_cp_pos_ylim = self.DBS_CFG['Default']['DBS_cp_pos_ylim']
                self.DBS_cp_pos_zlim = self.DBS_CFG['Default']['DBS_cp_pos_zlim']
                self.DBS_size_xlim = self.DBS_CFG['Default']['DBS_size_xlim']
                self.DBS_size_ylim = self.DBS_CFG['Default']['DBS_size_ylim']
                self.DBS_size_zlim = self.DBS_CFG['Default']['DBS_size_zlim']
                self.DBS_sort = self.DBS_CFG['Default']['DBS_sort']

                # set DBSCAN para for each subgroup
                index = f'Dynamic_ES_{ES_level}_above'
                for p in self.DBS_CFG[index]:
                    exec('self.%s = self.DBS_CFG[index][\'%s\']' % (p, p))

                # filter the points lower than energy strength level and feed into DBS
                data_points_sub, _ = self.DP_np_filter(data_points, axis=4, range_lim=(ES_level, None))
                vertices_list, valid_points_list, valid_points, noise = self.DBS(data_points_sub)
                vertices_list_total = vertices_list_total + vertices_list
                valid_points_list_total = valid_points_list_total + valid_points_list
                valid_points_total = np.concatenate([valid_points_total, valid_points])
                noise_total = np.concatenate([noise_total, noise])
        return vertices_list_total, valid_points_list_total, valid_points_total, noise_total

    def DBS(self, data_points):
        """
        :param data_points: (ndarray) data_numbers(n) * channels(c>=3) for a dozen of data frames
        :return: vertices_list: (list-ndarray) list of data_numbers(n) * channels(3) for vertices of 3D hull
                 valid_points_list: (list-ndarray) list of data_numbers(n) * channels(c) for valid points
                 valid_points: (ndarray) data_numbers(n) * channels(c) for total valid points
                 noise: (ndarray) data_numbers(n) * channels(c) for total noise points
        """
        # initial values
        vertices_list = []
        valid_points_list = []
        valid_points = np.ndarray([0, 5], dtype=np.float16)
        noise = np.ndarray([0, 5], dtype=np.float16)

        if data_points.shape[0] >= int(self.DBS_min_samples * 1):  # guarantee enough points and speed up when factor>1
            # DBSCAN find clusters
            labels = DBSCAN(eps=self.DBS_eps, min_samples=self.DBS_min_samples).fit_predict(data_points[:, 0:3])  # only feed xyz coords
            # filter DBSCAN noise
            noise = data_points[labels == -1]
            valid_points = data_points[labels != -1]
            valid_labels = labels[labels != -1]

            # get info for each cluster including central point position, size and label
            cluster_info_total = np.ndarray([0, 7], dtype=np.float16)  # (cp_pos_x, cp_pos_y, cp_pos_z, size_x, size_y, size_z, label)
            valid_labels_unique = np.unique(valid_labels)
            for j in range(len(valid_labels_unique)):
                label = valid_labels_unique[j]
                points = valid_points[valid_labels == label]
                x, y, z = self.DP_boundary_calculator(points, axis=range(3))
                cp_pos = np.array([sum(x) / 2, sum(y) / 2, sum(z) / 2], dtype=np.float16)
                size = np.concatenate([np.diff(x), np.diff(y), np.diff(z)])
                cluster_info = np.concatenate([cp_pos, size, np.array([label], dtype=np.float16)])[np.newaxis, :]
                cluster_info_total = np.concatenate([cluster_info_total, cluster_info])
            # apply filters
            cluster_info_total, _ = self.DP_np_filter(cluster_info_total, axis=0, range_lim=self.DBS_cp_pos_xlim)
            cluster_info_total, _ = self.DP_np_filter(cluster_info_total, axis=1, range_lim=self.DBS_cp_pos_ylim)
            cluster_info_total, _ = self.DP_np_filter(cluster_info_total, axis=2, range_lim=self.DBS_cp_pos_zlim)
            cluster_info_total, _ = self.DP_np_filter(cluster_info_total, axis=3, range_lim=self.DBS_size_xlim)
            cluster_info_total, _ = self.DP_np_filter(cluster_info_total, axis=4, range_lim=self.DBS_size_ylim)
            cluster_info_total, _ = self.DP_np_filter(cluster_info_total, axis=5, range_lim=self.DBS_size_zlim)
            # get index of cluster points that passed filters
            index = np.zeros(len(valid_labels), dtype=bool)
            for info in cluster_info_total:
                cluster_index = valid_labels == info[-1]
                index = np.logical_or(index, cluster_index)
            # update the valid points and labels
            valid_points = valid_points[index]
            valid_labels = valid_labels[index]

            # DBSCAN sort process
            if self.DBS_sort:
                # sort the DBSCAN labels based on the point number of cluster, high to low
                unique, counts = np.unique(valid_labels, return_counts=True)
                unique_sorted = [i[0] for i in sorted(tuple(zip(unique, counts)), key=lambda item: item[1], reverse=True)]
                # find the envelope of the biggest several clusters
                for i in range(len(unique_sorted)):
                    if i < self.DBS_sort:  # only choose the biggest several clusters
                        cluster = valid_points[valid_labels == unique_sorted[i]]
                        # vertices_list.append(self._convexhull(cluster))  # give cluster convexhull vertices
                        vertices_list.append(self.DP_cubehull(cluster))  # give cluster cubehull vertices
                        valid_points_list.append(cluster)
                    else:
                        break
            else:
                valid_labels_unique = np.unique(valid_labels)
                for i in range(len(valid_labels_unique)):
                    cluster = valid_points[valid_labels == valid_labels_unique[i]]
                    # vertices_list.append(self._convexhull(cluster))  # give cluster convexhull vertices
                    vertices_list.append(self.DP_cubehull(cluster))  # give cluster cubehull vertices
                    valid_points_list.append(cluster)

        return vertices_list, valid_points_list, valid_points, noise
