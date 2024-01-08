"""
Designed to collect the data from modules and save them, abbr. SVC
"""
import os.path
import pickle
import queue
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import winsound

from library.utils import folder_create_with_curmonth


class SaveCenter:
    def __init__(self, run_flag, shared_param_dict, **_kwargs_CFG):
        """
        get shared values and queues
        """
        self.run_flag = run_flag
        # shared params
        self.save_queue = shared_param_dict['save_queue']
        self.mansave_flag = shared_param_dict['mansave_flag']
        self.autosave_flag = shared_param_dict['autosave_flag']
        self.compress_video_file = shared_param_dict['compress_video_file']
        self.email_image = shared_param_dict['email_image']

        """
        pass config static parameters
        """
        """ module own config """
        SVC_CFG = _kwargs_CFG['SAVE_CENTER_CFG']
        self.file_save_dir = SVC_CFG['file_save_dir']
        self.experiment_name = SVC_CFG['experiment_name']

        # manual save
        self.mansave_period = SVC_CFG['mansave_period']
        # for radar
        self.mansave_rdr_data_deque = deque([], SVC_CFG['mansave_rdr_frame_max'])
        self.mansave_rdr_timestamp_deque = deque([], SVC_CFG['mansave_rdr_frame_max'])
        # for camera
        self.mansave_cam_data_deque = deque([], SVC_CFG['mansave_cam_frame_max'])
        self.mansave_cam_timestamp_deque = deque([], SVC_CFG['mansave_cam_frame_max'])

        # auto save
        # for radar
        self.autosave_rdr_data_deque = deque([], SVC_CFG['autosave_rdr_frame_max'])  # radar auto save data deque
        self.autosave_rdr_timestamp_deque = deque([], SVC_CFG['autosave_rdr_frame_max'])
        self.autosave_end_remove_period = SVC_CFG['autosave_end_remove_period']
        # for camera
        self.autosave_cam_buffer_deque = deque([], SVC_CFG['autosave_cam_buffer'])
        self.autosave_cam_timestamp_deque = deque([], SVC_CFG['autosave_cam_buffer'])

        """
        self content
        """
        # create a VideoWriter object to save the video
        self.videowriter = cv2.VideoWriter()

        # for camera auto save
        self.videowriter_status = False
        self.buffer_status = False
        self.asave_info = {
            'file_label' : 'auto_CameraVideo',
            'file_time'  : time.strftime('%b-%d-%H-%M-%S', time.localtime()),
            'file_format': '.mp4',
            'file_path'  : folder_create_with_curmonth(self.file_save_dir) + f"{self.experiment_name}_{'auto_CameraVideo'}_{time.strftime('%b-%d-%H-%M-%S', time.localtime())}{'.mp4'}",  # whole path includes name
            'fps'        : 30,
        }

        self._log('Start...')

    def run(self):
        while self.run_flag.value:
            # get saved data and classify
            try:
                packet = self.save_queue.get(block=True, timeout=10)
            except queue.Empty:
                self._log('Save Center Queue Empty.')
                break

            """manual save"""
            # save data for manual save
            if packet['source'] == 'radar':
                self.mansave_rdr_data_deque.append(packet['data'])
                self.mansave_rdr_timestamp_deque.append(packet['timestamp'])
            elif packet['source'] == 'camera':
                self.mansave_cam_data_deque.append(packet['data'])
                self.mansave_cam_timestamp_deque.append(packet['timestamp'])
            # manual save, only triggered at the end of recording
            if self.mansave_flag.value == 'image':
                msave_time = time.time()
                # for radar
                if len(self.mansave_rdr_data_deque) > 0:
                    self._pickle_save([self.mansave_rdr_data_deque[-1]],
                                      file_label='manual_RadarSnapshot',
                                      file_time=time.strftime('%b-%d-%H-%M-%S', time.localtime(msave_time)),
                                      file_dir=folder_create_with_curmonth(self.file_save_dir))
                # for camera
                if len(self.mansave_cam_data_deque) > 0:
                    self._opencv_save(self.mansave_cam_data_deque[-1],
                                      None,
                                      file_label='manual_CameraImage',
                                      file_time=time.strftime('%b-%d-%H-%M-%S', time.localtime(msave_time)),
                                      file_format='.jpg',
                                      file_dir=folder_create_with_curmonth(self.file_save_dir))
                # deactivate flag
                self.mansave_flag.value = None
            elif self.mansave_flag.value == 'video':
                # set start and end save time for asynchronous modules
                msave_end_time = time.time()
                msave_start_time = msave_end_time - self.mansave_period
                # for radar
                if len(self.mansave_rdr_data_deque) > 0:
                    start_index = abs(np.array(self.mansave_rdr_timestamp_deque) - msave_start_time).argmin()
                    self._pickle_save(list(self.mansave_rdr_data_deque)[start_index:],
                                      file_label='manual_RadarSeq',
                                      file_time=time.strftime('%b-%d-%H-%M-%S', time.localtime(msave_end_time)),
                                      file_dir=folder_create_with_curmonth(self.file_save_dir))
                    # clear the save deque
                    self.mansave_rdr_data_deque.clear()
                    self.mansave_rdr_timestamp_deque.clear()
                # for camera
                if len(self.mansave_cam_data_deque) > 0:
                    start_index = abs(np.array(self.mansave_cam_timestamp_deque) - msave_start_time).argmin()
                    self._opencv_save(list(self.mansave_cam_data_deque)[start_index:],
                                      list(self.mansave_cam_timestamp_deque)[start_index:],
                                      file_label='manual_CameraVideo',
                                      file_time=time.strftime('%b-%d-%H-%M-%S', time.localtime(msave_end_time)),
                                      file_format='.mp4',
                                      file_dir=folder_create_with_curmonth(self.file_save_dir))
                    # clear the save deque
                    self.mansave_cam_data_deque.clear()
                    self.mansave_cam_timestamp_deque.clear()
                # deactivate flag
                self.mansave_flag.value = None

            """auto save"""
            # auto save, constantly high from the beginning to the end of recording
            if self.autosave_flag.value:
                # for email notification
                if packet['source'] == 'camera':
                    self.email_image.value = packet['data']

                # save data for auto save
                if packet['source'] == 'radar':
                    self.autosave_rdr_data_deque.append(packet['data'])
                    self.autosave_rdr_timestamp_deque.append(packet['timestamp'])
                elif packet['source'] == 'camera':
                    # take some beginning buffer frames for video recording info
                    if not self.buffer_status:
                        self.autosave_cam_buffer_deque.append(packet['data'])
                        self.autosave_cam_timestamp_deque.append(packet['timestamp'])

                    # open VideoWriter for camera after buffer deque is full
                    if len(self.autosave_cam_buffer_deque) == self.autosave_cam_buffer_deque.maxlen and not self.videowriter_status:
                        self.asave_info['file_time'] = time.strftime('%b-%d-%H-%M-%S', time.localtime())
                        self.asave_info['fps'] = len(self.autosave_cam_timestamp_deque) / (self.autosave_cam_timestamp_deque[-1] - self.autosave_cam_timestamp_deque[0])
                        self.asave_info['file_path'] = folder_create_with_curmonth(self.file_save_dir) + f"{self.experiment_name}_{self.asave_info['file_label']}_{self.asave_info['file_time']}{self.asave_info['file_format']}"
                        width = self.autosave_cam_buffer_deque[0].shape[1]
                        height = self.autosave_cam_buffer_deque[0].shape[0]
                        self.videowriter.open(self.asave_info['file_path'], cv2.VideoWriter_fourcc(*'mp4v'), self.asave_info['fps'], (width, height))
                        # set status as done
                        self.videowriter_status = True
                        self.buffer_status = True

                    if self.videowriter_status:
                        # pop all buffer frames to VideoWriter
                        for _ in range(len(self.autosave_cam_buffer_deque)):
                            self.videowriter.write(self.autosave_cam_buffer_deque.popleft())
                        # write current frame as usual
                        self.videowriter.write(packet['data'])
            else:
                # for radar
                if len(self.autosave_rdr_data_deque) > 0:
                    # set start and end save time for asynchronous modules
                    asave_end_time = time.time() - self.autosave_end_remove_period
                    end_index = abs(np.array(self.autosave_rdr_timestamp_deque) - asave_end_time).argmin() + 1
                    self._pickle_save(list(self.autosave_rdr_data_deque)[:end_index],
                                      file_label='auto_RadarSeq',
                                      file_time=self.asave_info['file_time'],
                                      file_dir=folder_create_with_curmonth(self.file_save_dir))
                    # clear the save deque
                    self.autosave_rdr_data_deque.clear()
                    self.autosave_rdr_timestamp_deque.clear()
                # for camera
                if self.videowriter_status:
                    self.videowriter.release()
                    self._log(f"{os.path.basename(self.asave_info['file_path'])} saved with FPS: {round(self.asave_info['fps'], 2)}.")
                    self.videowriter_status = False
                    self.buffer_status = False
                    # send file name for compression
                    self.compress_video_file.value = self.asave_info['file_path']
                    winsound.Beep(800, 100)
                    winsound.Beep(500, 100)

    # used for radar data save of all cases
    def _pickle_save(self, data, file_label, file_time, file_dir='./'):
        file_name = f'{self.experiment_name}_{file_label}_{file_time}'
        with open(file_dir + file_name, 'wb') as file:
            pickle.dump(data, file)
        self._log(f'{file_name} saved.')
        winsound.Beep(500, 100)
        winsound.Beep(800, 100)

    # used for camera data save of manual case only
    def _opencv_save(self, data, timestamp, file_label, file_time, file_format, file_dir='./'):
        file_name = f'{self.experiment_name}_{file_label}_{file_time}{file_format}'
        # saved as image
        if file_format == '.jpg' or file_format == '.png':
            cv2.imwrite(file_dir + file_name, data)
            self._log(f'{file_name} saved.')
        # saved as video
        elif file_format == '.mp4':
            try:  # get video fps
                fps = len(timestamp) / (timestamp[-1] - timestamp[0])
            except:
                fps = 30  # for case of time_length = 0 by fast saving

            # open VideoWriter and save video
            self.videowriter.open(file_dir + file_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (data[0].shape[1], data[0].shape[0]))
            for f in data:
                self.videowriter.write(f)
            self.videowriter.release()
            self._log(f'{file_name} saved with FPS: {round(fps, 2)}.')
            self.compress_video_file.value = file_dir + file_name
        else:
            self._log(f'{file_name} save failed! Unsupported file type.')
        winsound.Beep(800, 100)
        winsound.Beep(500, 100)

    def _log(self, txt):  # print with device name
        print(f'[{self.__class__.__name__}]\t{txt}')

    def __del__(self):
        self.videowriter.release()
        self._log(f"Closed. Timestamp: {datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
