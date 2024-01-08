import glob
import math
import multiprocessing
import os
import pickle
from collections import deque

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Conv3D, MaxPooling3D, Embedding, LSTM, Bidirectional, Reshape, Concatenate


from library.frame_post_processor import FramePProcessor
from cfg.config_maggs307 import *


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 0 for GPU, -1 for CPU


def model_structure_5D(input_shape, output_shape):
    _model_input = Input(shape=input_shape)
    _model = Conv2D(32, (3, 3), padding='same', activation='relu')(_model_input)
    _model = MaxPooling2D(pool_size=(2, 2))(_model)
    _model = Reshape(target_shape=(27*2, 32))(_model)
    _model = Bidirectional(LSTM(64))(_model)
    _model = Flatten()(_model)
    _model = Dense(output_shape)(_model)
    _model = Model(_model_input, _model)
    return _model


def model_structure_matrix(input_shape, output_shape):
    def _submodel_structure(_submodel_input):
        _submodel = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(_submodel_input)
        _submodel = MaxPooling3D(pool_size=(2, 2, 2))(_submodel)
        _submodel = Reshape(target_shape=(16 * 9, 32))(_submodel)
        _submodel = Bidirectional(LSTM(64))(_submodel)
        _submodel = Flatten()(_submodel)
        return _submodel

    _submodel_input1 = Input(shape=input_shape[0])
    _submodel_input2 = Input(shape=input_shape[1])
    _submodel1 = _submodel_structure(_submodel_input1)
    _submodel2 = _submodel_structure(_submodel_input2)

    _model = Concatenate()([_submodel1, _submodel2])
    _model = Dense(output_shape)(_model)
    _model = Model([_submodel_input1, _submodel_input2], _model)
    return _model


def load_data():
    data = []
    dir_list = glob.glob('./data/Maggs_307/[a-d, f-z]*')
    for data_dir in dir_list:
        path_list = glob.glob(os.path.join(data_dir, '*RadarSeq*'))
        for data_path in path_list:
            # Load database
            with open(data_path, 'rb') as file:
                data = pickle.load(file)
                for d in data:

                    # get values from queues of all radars
                    data_allradar = np.ndarray([0, 5], dtype=np.float16)
                    for i, RADAR_CFG in enumerate(RADAR_CFG_LIST):
                        data_1radar = data[RADAR_CFG['name']]
                        # apply ES and speed filter for each radar channel
                        val_data, ES_noise = fpp.DP_ES_Speed_filter(data_1radar, RADAR_CFG['ES_threshold'])
                        val_data_allradar = np.concatenate([val_data_allradar, val_data])
                        ES_noise_allradar = np.concatenate([ES_noise_allradar, ES_noise])
                    pass





    _alldata_list = []
    _data_vertices_list = []  # for all data frames, each frame has only one set of vertices
    _labels_list = []

    # load database of each file
    for n in range(len(name_list)):
        # load file
        with open(path + name_list[n], 'rb') as file:
            dataset = pickle.load(file)

        fp = FramePProcessor(process_cfg=PROCESS_CFG)
        for data in dataset:
            data, _ = fp.ES_filter(data)  # filter points with low ES
            # select the frame passed DBSCAN
            vertices_list, valid_points_list, _, _ = fp.DBSCAN(data)
            # extract valuable data recognized by DBSCAN
            if len(vertices_list) != 0:
                _alldata_list.append(valid_points_list[0])  # only 1 set of valid points can be passed
                # _data_vertices_list.append(vertices_list[0])  # only 1 set of vertices can be passed
                # create labels
                _labels_list.append(n)

    return _alldata_list, np.array(_labels_list)  # , _data_vertices_list  # return data as a list of variable-length nparray frame, label as a nparray


def matrix_generator(size=None, resolution=None, coords=None, value=None):  # only support 3D cube matrix
    """
        size is a tuple/list of meters,
        resolution can be a float number,
        coords is a 2D nparray containing xyz numbers
        value is a 1D nparray containing the value of that position
    """
    # create an empty matrix of zeros based on the size and resolution
    _matrix = np.zeros([int(i/resolution) for i in size])
    # set values based on coords
    for c, v in zip(coords, value):
        x = int(c[0] / resolution)
        y = int(c[1] / resolution)
        z = int(c[2] / resolution)
        if _matrix[x, y, z] == 0:
            _matrix[x, y, z] = v
        else:
            _matrix[x, y, z] = (_matrix[x, y, z] + v) / 2  # get average if 2 or more points set into one mesh

    return _matrix


def dataset_length_fixer(method, _alldata_list):
    # post-process data format to ensure equal length inputs
    if method == '2D_padding_0':
        # find maximum length of dataset
        max_length = 0
        for data in _alldata_list:
            if max_length < data.shape[0]:
                max_length = data.shape[0]
        # padding 0
        for d in range(len(_alldata_list)):
            diff = max_length - _alldata_list[d].shape[0]
            data = np.concatenate([_alldata_list[d], np.zeros([diff, _alldata_list[d].shape[1]])])
            _alldata_list[d] = data
        _alldata_np = np.array(_alldata_list)  # convert list to nparray

        return _alldata_np[:, :, :, np.newaxis]

    elif method == '3D_matrix_cube':
        matrix_size = (2, 4, 3)
        matrix_resolution = 0.1
        matrix_mesh_No = [int(i / matrix_resolution) for i in matrix_size]
        _matrix_vel_np = np.empty([0] + matrix_mesh_No)
        _matrix_SNR_np = np.empty([0] + matrix_mesh_No)
        i = 0
        for frame in _alldata_list:
            i += 1
            print('Processing Frame No:', i)

            # split coords and info
            coords = frame[:, :3]
            velocity = frame[:, 3]
            SNR = frame[:, 4]

            # shift the coords, keep positive
            coords_mapped = ((coords + (1, 0, 1)) * (1, 1, 1)).astype(np.float16)

            # generate sparse matrix
            matrix_vel = matrix_generator(size=matrix_size, resolution=matrix_resolution, coords=coords_mapped, value=velocity)[np.newaxis, :]
            matrix_SNR = matrix_generator(size=matrix_size, resolution=matrix_resolution, coords=coords_mapped, value=SNR)[np.newaxis, :]
            # merge all matrix
            _matrix_vel_np = np.concatenate([_matrix_vel_np, matrix_vel], axis=0)
            _matrix_SNR_np = np.concatenate([_matrix_SNR_np, matrix_SNR], axis=0)

        return _matrix_vel_np[:, :, :, :, np.newaxis], _matrix_SNR_np[:, :, :, :, np.newaxis]

    elif method == '3D_matrix_cube_small':  # coords start with (0, 0, 0) by default
        matrix_size = (0.8, 0.8, 1.8)
        matrix_central_point = [i / 2 for i in matrix_size]
        matrix_resolution = 0.1
        matrix_mesh_No = [int(i / matrix_resolution) for i in matrix_size]
        _matrix_vel_np = np.empty([0] + matrix_mesh_No)
        _matrix_SNR_np = np.empty([0] + matrix_mesh_No)
        count = 0
        for frame in _alldata_list:
            count += 1
            print('Processing Frame No:', count)

            # split coords and info
            coords = frame[:, :3]
            velocity = frame[:, 3]
            SNR = frame[:, 4]

            # find current frame weight central point
            point_No = frame.shape[0]
            x = sum(coords[:, 0]) / point_No
            y = sum(coords[:, 1]) / point_No
            z = sum(coords[:, 2]) / point_No
            data_central_point = [x, y, z]
            # compare with matrix central point and map to it
            shift_diff = [m - d for m, d in zip(matrix_central_point, data_central_point)]
            coords_mapped = (coords + shift_diff).astype(np.float16)

            # remove points outside
            coords_mapped = coords_mapped[(coords_mapped[:, 0] >= 0) & (coords_mapped[:, 0] < matrix_size[0])]
            coords_mapped = coords_mapped[(coords_mapped[:, 1] >= 0) & (coords_mapped[:, 1] < matrix_size[1])]
            coords_mapped = coords_mapped[(coords_mapped[:, 2] >= 0) & (coords_mapped[:, 2] < matrix_size[2])]

            # generate sparse matrix
            matrix_vel = matrix_generator(size=matrix_size, resolution=matrix_resolution, coords=coords_mapped, value=velocity)[np.newaxis, :]
            matrix_SNR = matrix_generator(size=matrix_size, resolution=matrix_resolution, coords=coords_mapped, value=SNR)[np.newaxis, :]
            # merge all matrix
            _matrix_vel_np = np.concatenate([_matrix_vel_np, matrix_vel], axis=0)
            _matrix_SNR_np = np.concatenate([_matrix_SNR_np, matrix_SNR], axis=0)

        return _matrix_vel_np[:, :, :, :, np.newaxis], _matrix_SNR_np[:, :, :, :, np.newaxis]


if __name__ == '__main__':
    kwargs_CFG = {'FRAME_POST_PROCESSOR_CFG': FRAME_POST_PROCESSOR_CFG,
                  'DBSCAN_GENERATOR_CFG'    : DBSCAN_GENERATOR_CFG,
                  'BGNOISE_FILTER_CFG'      : BGNOISE_FILTER_CFG,
                  'HUMAN_TRACKING_CFG'      : HUMAN_TRACKING_CFG,
                  'HUMAN_OBJECT_CFG'        : HUMAN_OBJECT_CFG}

    fpp = FramePProcessor(**kwargs_CFG)

    # load data from stored dataset
    alldata_list, labels_np = load_data()

    # # choose method to make data length equal
    # alldata_np = dataset_length_fixer('2D_padding_0', alldata_list)
    # model = model_structure_5D(alldata_np.shape[1:], max(labels_np) + 1)
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()
    # history = model.fit(alldata_np, labels_np, 
    #                     validation_data=(alldata_np, labels_np), 
    #                     epochs=10)

    # # choose method to make data length equal
    # matrix_vel_np, matrix_SNR_np = dataset_length_fixer('3D_matrix_cube_small', alldata_list)
    # model = model_structure_matrix((matrix_vel_np.shape[1:], matrix_SNR_np.shape[1:]), max(labels_np) + 1)
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()
    # history = model.fit([matrix_vel_np, matrix_SNR_np], labels_np,
    #                     validation_data=([matrix_vel_np, matrix_SNR_np], labels_np),
    #                     epochs=10)
    #
    # pass
    #
    # # Load database
    # with open('../data/Maggs_307/SZC/SZC_auto_RadarSeq_Jun-04-16-59-54', 'rb') as file:
    #     data = pickle.load(file)
