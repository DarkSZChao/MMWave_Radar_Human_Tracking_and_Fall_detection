"""
Designed for processing the data, abbr. DP
data(ndarray) = data_numbers(n) * channels(x, y, z, v, SNR)
"""

import numpy as np
from scipy.spatial import ConvexHull


class DataProcessor:
    def DP_list_nesting_remover(self, input_list, output_list=None):
        """
        to extract each element inside a list with deep nesting level
        :param input_list: (list/element) a list with multiple nesting level
        :param output_list: (list) a cumulated list during iteration
        :return: output_list: (list) a non-nesting list
        """
        # list_nesting_remover = lambda list_in: [list_out for i in list_in for list_out in list_nesting_remover(i)] if type(list_in) is list else [list_in]

        if output_list is None:
            output_list = []
        if type(input_list) is list:
            for i in input_list:
                output_list = self.DP_list_nesting_remover(i, output_list)
        else:
            output_list.append(input_list)
        return output_list

    def DP_ES_Speed_filter(self, data_points, ES_threshold):
        """
        :param data_points: (ndarray) data_numbers(n) * channels(c=5)
        :param ES_threshold: (dict) the ES threshold
        :return: data_points: (ndarray) data_numbers(n) * channels(c=5)
                 noise: (ndarray) data_numbers(n) * channels(c=5)
        """
        # remove points with low energy strength
        data_points, noise = self.DP_np_filter(data_points, axis=4, range_lim=ES_threshold['range'])

        # identify the noise with speed
        if len(noise) > 0 and ES_threshold['speed_none_0_exception']:
            noise, noise_with_speed = self.DP_np_filter(noise, axis=3, range_lim=0)
            data_points = np.concatenate([data_points, noise_with_speed])
        return data_points, noise

    def DP_np_filter(self, data_points, axis, range_lim, mode=1):
        """
        only one axis can be processed in one call
        :param data_points: (ndarray) data_numbers(n) * channels(c)
        :param axis: (int) the axis number
        :param range_lim: (tuple/int/float) (bottom_lim, upper_lim) element can be None, the range for preserved data
        :param mode: (int) 0-[min, max], 1-[min, max), 2-(min, max], 3-(min, max), include boundary or not
        :return: data_preserved: (ndarray) data_numbers(n) * channels(c)
                 data_removed: (ndarray) data_numbers(n) * channels(c)
        """
        preserved_index, removed_index = self.DP_get_idx_bool(data_points, axis, range_lim, mode)
        # get data and noise
        data_preserved = data_points[preserved_index]
        data_removed = data_points[removed_index]
        return data_preserved, data_removed

    def DP_get_idx_bool(self, data_points, axis, range_lim, mode=1):
        """
        only one axis can be processed in one call
        :param data_points: (ndarray) data_numbers(n) * channels(c)
        :param axis: (int) the axis number
        :param range_lim: (tuple/int/float) (bottom_lim, upper_lim) element can be None, the range for preserved data
        :param mode: (int) 0-[min, max], 1-[min, max), 2-(min, max], 3-(min, max), include boundary or not
        :return: preserved_index: (ndarray-bool) data_numbers(n)
                 removed_index: (ndarray-bool) data_numbers(n)
        """
        # initialize the index array
        preserved_index = np.ones(len(data_points), dtype=bool)
        removed_index = np.zeros(len(data_points), dtype=bool)

        if range_lim is not None:
            if type(range_lim) is tuple or type(range_lim) is list:  # expect list type
                if range_lim[0] is not None:
                    if mode == 0 or mode == 1:
                        index = data_points[:, axis] >= range_lim[0]
                    else:
                        index = data_points[:, axis] > range_lim[0]
                    # update the index
                    preserved_index = preserved_index & index
                    removed_index = removed_index | ~index
                if range_lim[1] is not None:
                    if mode == 0 or mode == 2:
                        index = data_points[:, axis] <= range_lim[1]
                    else:
                        index = data_points[:, axis] < range_lim[1]
                    # update the index
                    preserved_index = preserved_index & index
                    removed_index = removed_index | ~index
            else:  # expect int/float type
                index = data_points[:, axis] == range_lim
                # update the index
                preserved_index = preserved_index & index
                removed_index = removed_index | ~index
        return preserved_index, removed_index

    def DP_np_2D_row_removal(self, database, data_remove):
        """
        remove those data rows in the data_remove from the database
        :param database: (ndarray) data_numbers(n) * channels(c), original database
        :param data_remove: (ndarray) data_numbers(n) * channels(c), need to be removed from database
        :return: database: (ndarray) data_numbers(n) * channels(c), updated database
        """
        # locate the data_remove in database first and get boolean index
        database_index = np.zeros(len(database), dtype=bool)
        for d in data_remove:
            # locate one data row which needs to be removed in database
            row_index = np.ones(len(database), dtype=bool)
            for i in range(data_remove.shape[1]):
                idx, _ = self.DP_get_idx_bool(database, axis=i, range_lim=d[i])
                row_index = row_index & idx
            database_index = database_index | row_index
        # reverse the index to remove data_remove and output database
        return database[~database_index]

    def DP_boundary_calculator(self, data_points, axis):
        """
        multiple axis can be processed in one call
        :param data_points: (ndarray) data_numbers(n>0) * channels(c)
        :param axis: (int/tuple/list/range) the axis number
        :return: result: (tuple-tuple) boundaries, (min, max)
        """
        result = []
        axis_list = [axis] if type(axis) == int else axis
        for axis in axis_list:
            dmin, dmax = data_points[:, axis].min(), data_points[:, axis].max()
            result.append((dmin, dmax))
        return tuple(result) if len(result) > 1 else result[0]

    # def repeated_points_removal(self, data_points):
    #     """
    #     :param data_points: (ndarray) data_numbers(n) * channels(c)
    #     :return: (ndarray) data_numbers(n) * channels(c)
    #     """
    #     # lower the precision to speed up
    #     data_points = data_points.astype(np.float16)
    #     data_points = np.around(data_points, decimals=2)  # not working when the number is too big more than 500
    #     # remove repeated points
    #     data_points = np.unique(data_points, axis=0)
    #     return data_points

    def DP_convexhull(self, cluster):
        """
        :param cluster: (ndarray) data_numbers(n) * channels(c>=3) for cluster data
        :return: (ndarray) data_numbers(16) * channels(3) for hull vertices
        """
        try:  # just in case of insufficient points or all points in single line
            vertices_index = ConvexHull(cluster[:, 0:3]).vertices
            vertices_index = np.concatenate([vertices_index, vertices_index[0:1]])  # connect end to end for drawing closed shape
        except:
            vertices_index = []
        return cluster[vertices_index]

    def DP_cubehull(self, cluster, *args):
        """
        :param cluster: (ndarray) data_numbers(n) * channels(c>=3) for cluster data
        :return: (ndarray) data_numbers(16) * channels(3) for hull vertices
        """
        try:
            xboundary, yboundary, zboundary = args
        except:
            xboundary, yboundary, zboundary = self.DP_boundary_calculator(cluster, axis=range(3))
        xmin, xmax = xboundary
        ymin, ymax = yboundary
        zmin, zmax = zboundary
        x = [xmin, xmax, xmax, xmin, xmin, xmin, xmax, xmax, xmin, xmin, xmax, xmax, xmax, xmax, xmin, xmin]
        y = [ymin, ymin, ymax, ymax, ymin, ymin, ymin, ymax, ymax, ymin, ymin, ymin, ymax, ymax, ymax, ymax]
        z = [zmin, zmin, zmin, zmin, zmin, zmax, zmax, zmax, zmax, zmax, zmax, zmin, zmin, zmax, zmax, zmin]
        return np.array([x, y, z]).T


if __name__ == '__main__':
    # frame_group = np.zeros([5, 3]).astype(np.float16)
    # frame_group[0, :] = [2, 1, 1]
    # frame_group[1, :] = [1, -1, 1]
    # frame_group[2, :] = [1, 0, 0]
    # frame_group[-1, :] = [0, 1, 10]
    #
    # dp = DataProcessor()
    # a, b, c = dp.boundary_calculator(frame_group, axis=range(3))

    a = [1, [12, [1, [1, 23, 4, [1, 2, [3]]]]], [3, [1, [2, 645, [3, 5, [456, [4, [45, 7], [2, [[5]]]]]]]], 4]]
    b = [[1, [12, 1, [1, 23, 4]]]]
    dp = DataProcessor()
    print(dp.DP_list_nesting_remover(a))
    # print(dp.list_nesting_remover(b))

    pass
