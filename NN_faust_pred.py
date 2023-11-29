import math
import os
import pickle

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation, Add
from tensorflow.keras.utils import to_categorical

from Library.utils import random_split

# enable GPU for calculation
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 0 for GPU, -1 for CPU
print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))


def scheduler(epoch, lr):
    if epoch < 10:
        lr = 0.0002
    # elif epoch == 50:
    #     lr = 0.002
    else:
        # lr = lr * 0.95
        lr = lr * math.exp(-lr * 5)
    return lr


lr_callback = tf.keras.callbacks.LearningRateScheduler(schedule=scheduler, verbose=1)


class NN:
    @staticmethod
    def Conv1D_BN_Relu(filter_No, kernel_size, strides, layer):
        layer = Conv1D(filter_No, kernel_size, strides=strides, padding='same')(layer)
        # layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        return layer

    def Res_block(self, filters, node_0, first_block=False):
        if first_block:
            node_0 = self.Conv1D_BN_Relu(filters, 3, 1, node_0)

        layer = self.Conv1D_BN_Relu(filters, 3, 1, node_0)
        layer = self.Conv1D_BN_Relu(filters, 3, 1, layer)
        node_1 = Add()([layer, node_0])
        return node_1

    def CNN_ResNet(self, input_shape, output_shape):
        model_input = Input(shape=input_shape)

        layer = self.Res_block(256, model_input, first_block=True)
        layer = self.Res_block(256, layer)
        layer = self.Res_block(256, layer)
        layer = MaxPooling1D(pool_size=2)(layer)

        layer = self.Res_block(128, model_input, first_block=True)
        layer = self.Res_block(128, layer)
        layer = self.Res_block(128, layer)
        layer = MaxPooling1D(pool_size=2)(layer)

        layer = self.Res_block(64, layer, first_block=True)
        layer = self.Res_block(64, layer)
        layer = self.Res_block(64, layer)
        layer = MaxPooling1D(pool_size=2)(layer)

        layer = self.Res_block(32, layer, first_block=True)
        layer = self.Res_block(32, layer)
        layer = self.Res_block(32, layer)
        layer = MaxPooling1D(pool_size=2)(layer)

        layer = self.Res_block(16, layer, first_block=True)
        layer = self.Res_block(16, layer)
        layer = self.Res_block(16, layer)
        layer = MaxPooling1D(pool_size=2)(layer)

        layer = Flatten()(layer)
        layer = Dense(128)(layer)
        layer = Dense(64)(layer)
        layer = Dense(output_shape, activation='softmax')(layer)
        return Model(model_input, layer)

    def CNN1D(self, input_shape, output_shape):
        _model_input = Input(shape=input_shape)

        layer = Conv1D(64, 3, strides=1, padding='same', activation='relu')(_model_input)
        layer = MaxPooling1D(pool_size=2)(layer)

        layer = Conv1D(64, 3, strides=1, padding='same', activation='relu')(layer)
        layer = MaxPooling1D(pool_size=2)(layer)

        layer = Conv1D(128, 3, strides=1, padding='same', activation='relu')(layer)
        layer = MaxPooling1D(pool_size=2)(layer)

        layer = Conv1D(128, 3, strides=1, padding='same', activation='relu')(layer)
        layer = MaxPooling1D(pool_size=2)(layer)

        layer = Conv1D(256, 3, strides=1, padding='same', activation='relu')(layer)
        layer = MaxPooling1D(pool_size=2)(layer)

        layer = Conv1D(256, 3, strides=1, padding='same', activation='relu')(layer)
        layer = MaxPooling1D(pool_size=2)(layer)

        layer = Conv1D(512, 3, strides=1, padding='same', activation='relu')(layer)
        layer = MaxPooling1D(pool_size=2)(layer)

        layer = Conv1D(512, 3, strides=1, padding='same', activation='relu')(layer)
        layer = MaxPooling1D(pool_size=2)(layer)

        layer = Flatten()(layer)
        layer = Dense(512)(layer)
        layer = Dense(128)(layer)
        layer = Dense(64)(layer)
        layer = Dense(output_shape, activation='softmax')(layer)
        return Model(_model_input, layer)

    @staticmethod
    def CNN2D(input_shape, output_shape):
        _model_input = Input(shape=input_shape)
        _model = Conv2D(480, (3, 3), activation='relu', padding='same')(_model_input)
        _model = MaxPooling2D(pool_size=(2, 1))(_model)
        _model = Conv2D(320, (3, 3), activation='relu', padding='same')(_model)
        _model = MaxPooling2D(pool_size=(2, 1))(_model)
        _model = Conv2D(240, (3, 3), activation='relu', padding='same')(_model)
        _model = MaxPooling2D(pool_size=(2, 1))(_model)
        _model = Conv2D(120, (3, 3), activation='relu', padding='same')(_model)
        _model = MaxPooling2D(pool_size=(2, 1))(_model)
        _model = Flatten()(_model)
        _model = Dense(128)(_model)
        _model = Dense(output_shape, activation='softmax')(_model)
        return Model(_model_input, _model)


def load_data(label_categorical=False):
    """
    Faust point cloud data only
    """
    # with open('C:/SZC/PhD/MMWave_Radar/ID/Data/FAUSTSim/faust_pred_pointcloud_10000_dis_randspeed', 'rb') as file:
    #     datapoints_list, labels_np = pickle.load(file)
    #
    # np_length = 900
    # datapoints_np = np.empty([0, np_length, 3])
    # for i, d in enumerate(datapoints_list):
    #     zeros = np.zeros([np_length - len(d), 3])
    #     datapoints = np.concatenate([d, zeros])[np.newaxis]
    #     datapoints_np = np.concatenate([datapoints_np, datapoints])
    #     print(f'Padding: {i}')

    with open('C:/SZC/PhD/MMWave_Radar/ID/Data/FAUSTSim/faust_pred_pointcloud_10000_dis_randspeed_padding0_1', 'rb') as file:
        datapoints_np1, labels_np1 = pickle.load(file)
    with open('C:/SZC/PhD/MMWave_Radar/ID/Data/FAUSTSim/faust_pred_pointcloud_10000_dis_randspeed_padding0_2', 'rb') as file:
        datapoints_np2, labels_np2 = pickle.load(file)
    with open('C:/SZC/PhD/MMWave_Radar/ID/Data/FAUSTSim/faust_pred_pointcloud_10000_dis_randspeed_padding0_3', 'rb') as file:
        datapoints_np3, labels_np3 = pickle.load(file)
    datapoints_np, labels_np = np.concatenate([datapoints_np1, datapoints_np2, datapoints_np3]), np.concatenate([labels_np1, labels_np2, labels_np3])

    # define dataset params
    people = 10
    train_ratio = 0.8
    train_data_np = np.empty([0] + list(datapoints_np.shape[1:]))
    train_label_np = np.empty([0] + list(labels_np.shape[1:]))
    val_data_np = np.empty([0] + list(datapoints_np.shape[1:]))
    val_label_np = np.empty([0] + list(labels_np.shape[1:]))
    for person in range(people):
        # random split whole dataset into train and valid in posture dimension
        idx = np.squeeze(np.argwhere(labels_np[:, 0] == person))
        train_idx, val_idx = random_split(idx, split_ratio=train_ratio)
        train_data_np = np.concatenate([train_data_np, datapoints_np[train_idx]])
        train_label_np = np.concatenate([train_label_np, labels_np[train_idx]])
        val_data_np = np.concatenate([val_data_np, datapoints_np[val_idx]])
        val_label_np = np.concatenate([val_label_np, labels_np[val_idx]])

    if label_categorical:
        return train_data_np, to_categorical(train_label_np), val_data_np, to_categorical(val_label_np)
    else:
        return train_data_np, train_label_np, val_data_np, val_label_np


if __name__ == '__main__':
    # load data from stored dataset
    train_data_np, train_label_np, val_data_np, val_label_np = load_data(label_categorical=True)

    model = NN().CNN1D(train_data_np.shape[1:], 10)
    # train_data_np = train_data_np[:, :, :, np.newaxis]
    # val_data_np = val_data_np[:, :, :, np.newaxis]
    # model = NN.CNN2D(train_data_np.shape[1:], 10)
    # model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(train_data_np, train_label_np,
                        validation_data=(val_data_np, val_label_np),
                        shuffle=True,
                        # batch_size=256,
                        callbacks=[lr_callback],
                        epochs=500,
                        verbose=1)

    print(f"Epoch {np.argmax(history.history['val_accuracy']) + 1} with maximum val_accuracy: {history.history['val_accuracy'][np.argmax(history.history['val_accuracy'])] * 100:.2f}%")

    plt.figure()
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    plt.show()
