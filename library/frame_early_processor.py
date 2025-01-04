"""
Designed for data frame from each radar, abbr. FEP
data(ndarray) = data_numbers(n) * channels(x, y, z, v, SNR)
"""

from collections import deque

import numpy as np
from matplotlib import pyplot as plt
from numpy import sin, cos

from library.data_processor import DataProcessor


class FrameEProcessor(DataProcessor):  # early processing for frame of each radar before merging
    def __init__(self, **kwargs_CFG):
        """
        pass config static parameters
        """
        """ module own config """
        FEP_CFG = kwargs_CFG['FRAME_EARLY_PROCESSOR_CFG']
        self.FEP_frame_group_deque = deque([], FEP_CFG['FEP_frame_deque_length'])

        """ other configs """
        RDR_CFG = kwargs_CFG['RADAR_CFG']
        self.name = RDR_CFG['name']
        self.xlim = RDR_CFG['xlim']
        self.ylim = RDR_CFG['ylim']
        self.zlim = RDR_CFG['zlim']
        self.pos_offset = RDR_CFG['pos_offset']
        self.facing_angle = RDR_CFG['facing_angle']

        """
        inherit father class __init__ para
        """
        super().__init__()

    def FEP_accumulate_update(self, frame):  # ndarray(points, channels=5) of 1 frame
        # append frames
        self.FEP_frame_group_deque.append(frame)
        frame_group = np.concatenate(self.FEP_frame_group_deque).astype(np.float16)  # concatenate the deque list

        # apply boundary filter
        frame_group = self.FEP_boundary_filter(frame_group)

        # apply angle shift and position updates
        frame_group = np.concatenate([self.FEP_trans_rotation_3D(frame_group[:, 0:3]), frame_group[:, 3:5]], axis=1)
        frame_group = np.concatenate([self.FEP_trans_position_3D(frame_group[:, 0:3]), frame_group[:, 3:5]], axis=1)

        return frame_group  # ndarray(points, channels) of a dozen of frames

    def FEP_boundary_filter(self, data_points):
        """
        :param data_points: (ndarray) data_numbers(n) * channels(c>3)
        :return: data_points: (ndarray) data_numbers(n) * channels(c>3)
        """
        # remove out-ranged points
        data_points, _ = self.DP_np_filter(data_points, axis=0, range_lim=self.xlim)
        data_points, _ = self.DP_np_filter(data_points, axis=1, range_lim=self.ylim)
        data_points, _ = self.DP_np_filter(data_points, axis=2, range_lim=self.zlim)
        return data_points

    def FEP_trans_rotation_3D(self, data_points):
        """
        update the data from radar based on its facing angle
        based on right-hand coord-sys (global coord-sys is used, means the xyz axes are frozen during rotation transformation),
        3 rotation matrices for xyz axes are used, default rotation sequence: dot matrix of z-axis first, followed by y and x-axis

        :param data_points: (ndarray) data_numbers(n) * channels(3)
        :return: (ndarray) data_numbers(n) * channels(3)
        """
        ref_point = (0, 0, 0)  # reference point, (0, 0, 0) by default not using
        rpx, rpy, rpz = ref_point
        alpha, beta, gamma = np.deg2rad(self.facing_angle['angle'])  # tuple of degrees for xyz axes

        # define rotation matrices for each axis
        Rx = np.array([[1, 0, 0, 0],
                       [0, cos(alpha), -sin(alpha), rpy * (1 - cos(alpha)) + rpz * sin(alpha)],
                       [0, sin(alpha), cos(alpha), rpz * (1 - cos(alpha)) - rpy * sin(alpha)],
                       [0, 0, 0, 1]],
                      dtype=data_points.dtype)
        Ry = np.array([[cos(beta), 0, sin(beta), rpx * (1 - cos(beta)) - rpz * sin(beta)],
                       [0, 1, 0, 0],
                       [-sin(beta), 0, cos(beta), rpz * (1 - cos(beta)) + rpx * sin(beta)],
                       [0, 0, 0, 1]],
                      dtype=data_points.dtype)
        Rz = np.array([[cos(gamma), -sin(gamma), 0, rpx * (1 - cos(gamma)) + rpy * sin(gamma)],
                       [sin(gamma), cos(gamma), 0, rpy * (1 - cos(gamma)) - rpx * sin(gamma)],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]],
                      dtype=data_points.dtype)

        # add one row of 1 to be compatible with matrices
        data_points = np.concatenate([data_points, np.ones([data_points.shape[0], 1], dtype=data_points.dtype)], axis=1).T

        # choose rotation sequence
        if self.facing_angle['sequence'] == 'xyz':
            data_points_transformed = np.dot(np.dot(np.dot(Rz, Ry), Rx), data_points)
        elif self.facing_angle['sequence'] == 'xzy':
            data_points_transformed = np.dot(np.dot(np.dot(Ry, Rz), Rx), data_points)
        elif self.facing_angle['sequence'] == 'yxz':
            data_points_transformed = np.dot(np.dot(np.dot(Rz, Rx), Ry), data_points)
        elif self.facing_angle['sequence'] == 'yzx':
            data_points_transformed = np.dot(np.dot(np.dot(Rx, Rz), Ry), data_points)
        elif self.facing_angle['sequence'] == 'zxy':
            data_points_transformed = np.dot(np.dot(np.dot(Ry, Rx), Rz), data_points)
        else:  # default sequence is zyx
            data_points_transformed = np.dot(np.dot(np.dot(Rx, Ry), Rz), data_points)
        return data_points_transformed.T[:, 0:3]

    def FEP_trans_position_3D(self, data_points):
        """
        update the data from radar based on its position offset
        :param data_points: (ndarray) data_numbers(n) * channels(3)
        :return: (ndarray) data_numbers(n) * channels(3)
        """
        dx, dy, dz = self.pos_offset  # tuple of distance for xyz axes

        # define distance matrix for each axis
        T = np.array([[1, 0, 0, dx],
                      [0, 1, 0, dy],
                      [0, 0, 1, dz],
                      [0, 0, 0, 1]],
                     dtype=data_points.dtype)

        # add one row of 1 to be compatible with matrices
        data_points = np.concatenate([data_points, np.ones([data_points.shape[0], 1], dtype=data_points.dtype)], axis=1).T
        data_points_transformed = np.dot(T, data_points)
        return data_points_transformed.T[:, 0:3]


if __name__ == '__main__':
    RADAR_CFG = {'name'          : 'test',
                 'cfg_port_name' : 'COM3',
                 'data_port_name': 'COM4',
                 'cfg_file_name' : './cfg/IWR1843_3D.cfg',  # always use 3D data as input
                 'xlim'          : None,  # the x-direction limit for cloud points from this single radar, set as [a, b), from radar view
                 'ylim'          : (0.25, 4),
                 'zlim'          : None,
                 'pos_offset'    : (0, 0, 0.8),  # default pos_offset is (0, 0, 0)
                 'facing_angle'  : {'angle': (0, -60, 90), 'sequence': None},  # right-hand global coord-sys, (x, y, z): [-180, 180] positive counted anti-clockwise when facing from axis vertex towards origin, default rotation sequence: zyx
                 'ES_threshold'  : {'range': (200, None), 'speed_none_0_exception': True},  # if speed_none_0_exception is True, then the data with low ES but with speed will be reserved
                 }
    FRAME_EARLY_PROCESSOR_CFG = {  # early process config
        'FEP_frame_deque_length': 10,  # the number of frame stacked
    }
    kwargs_CFG = {'RADAR_CFG'                : RADAR_CFG,
                  'FRAME_EARLY_PROCESSOR_CFG': FRAME_EARLY_PROCESSOR_CFG}

    # frame = np.array(([], [], []))
    f = FrameEProcessor(**kwargs_CFG)

    """
    This example shows the surface which can be treated as the radar surface facing towards the center. 
    Because if the radar placed facing down, the received point cloud needs to be lift to compensate the angle. 
    Following the instruction (global coord-sys, positive counted anti-clockwise), it will be easy to rotate.
    """
    # data
    points = 1000
    x = np.random.uniform(-2, 2, points)
    y = np.zeros(points) - 1
    z = np.random.uniform(-2, 2, points)
    x = np.concatenate([x, np.linspace(-2, 2, points)])
    y = np.concatenate([y, np.zeros(points) - 1])
    z = np.concatenate([z, np.linspace(-2, 2, points)])
    data = f.FEP_trans_rotation_3D(np.concatenate([x[:, np.newaxis], y[:, np.newaxis], z[:, np.newaxis]], axis=1))

    # create a figure
    fig = plt.figure(figsize=(9, 9))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_zlim(-2, 2)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    # draw axis
    x_axis = np.arange(-2, 2, 0.01)
    y_axis = np.zeros(400)
    z_axis = np.zeros(400)
    x_axis = np.concatenate([x_axis, np.zeros(400)])
    y_axis = np.concatenate([y_axis, np.arange(-2, 2, 0.01)])
    z_axis = np.concatenate([z_axis, np.zeros(400)])
    x_axis = np.concatenate([x_axis, np.zeros(400)])
    y_axis = np.concatenate([y_axis, np.zeros(400)])
    z_axis = np.concatenate([z_axis, np.arange(-2, 2, 0.01)])
    ax1.plot3D(x_axis, y_axis, z_axis, marker='o', linestyle='None', color='r')

    # draw the surface
    ax1.plot3D(data[:, 0], data[:, 1], data[:, 2], marker='.', linestyle='None', color='g')

    plt.show()
