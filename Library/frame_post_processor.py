"""
Designed for post-processing of data frame merged from all radars, abbr. FPP
data(ndarray) = data_numbers(n) * channels(x, y, z, v, SNR)
"""

from Library.DBSCAN_generator import DBSCANGenerator
from Library.bgnoise_filter import BGNoiseFilter
from Library.human_tracking import HumanTracking


class FramePProcessor(DBSCANGenerator, BGNoiseFilter, HumanTracking):
    def __init__(self, **kwargs_CFG):
        """
        pass config static parameters
        """
        """ module own config """
        FPP_CFG = kwargs_CFG['FRAME_POST_PROCESSOR_CFG']
        self.FPP_global_xlim = FPP_CFG['FPP_global_xlim']
        self.FPP_global_ylim = FPP_CFG['FPP_global_ylim']
        self.FPP_global_zlim = FPP_CFG['FPP_global_zlim']
        self.FPP_ES_threshold = FPP_CFG['FPP_ES_threshold']

        """
        inherit father class __init__ para
        """
        super().__init__(**kwargs_CFG)

    def FPP_boundary_filter(self, data_points):
        """
        :param data_points: (ndarray) data_numbers(n) * channels(c>3)
        :return: data_points: (ndarray) data_numbers(n) * channels(c>3)
        """
        # remove out-ranged points
        data_points, _ = self.DP_np_filter(data_points, axis=0, range_lim=self.FPP_global_xlim)
        data_points, _ = self.DP_np_filter(data_points, axis=1, range_lim=self.FPP_global_ylim)
        data_points, _ = self.DP_np_filter(data_points, axis=2, range_lim=self.FPP_global_zlim)
        return data_points

    def FPP_ES_Speed_filter(self, data_points):
        """
        :param data_points: (ndarray) data_numbers(n) * channels(c=5)
        :return: data_points: (ndarray) data_numbers(n) * channels(c=5)
                 noise: (ndarray) data_numbers(n) * channels(c=5)
        """
        data_points, noise = self.DP_ES_Speed_filter(data_points, self.FPP_ES_threshold)
        return data_points, noise
