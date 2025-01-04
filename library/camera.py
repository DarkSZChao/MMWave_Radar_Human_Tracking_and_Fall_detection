"""
Designed to capture the image by using web camera, abbr. CAM
"""

import time
from datetime import datetime
from multiprocessing import Process, Manager

import cv2


class Camera:
    def __init__(self, run_flag, shared_param_dict, **kwargs_CFG):
        """
        get shared values and queues
        """
        self.run_flag = run_flag
        # shared params
        self.save_queue = shared_param_dict['save_queue']
        self.autosave_flag = shared_param_dict['autosave_flag']
        self.status = shared_param_dict['proc_status_dict']
        self.status['Module_CAM'] = True
        """
        pass config static parameters
        """
        """ module own config """
        CAM_CFG = kwargs_CFG['CAMERA_CFG']
        self.name = CAM_CFG['name']
        self.capture = cv2.VideoCapture(CAM_CFG['camera_index'], cv2.CAP_DSHOW)  # create a camera capture
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_CFG['capture_resolution'][0])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_CFG['capture_resolution'][1])
        self.window_enable = CAM_CFG['window_enable']

        self.auto_inactive_skip_enable = CAM_CFG['auto_inactive_skip_enable']

        """ other configs """
        self.AUTOSAVE_ENABLE = kwargs_CFG['AUTOSAVE_ENABLE']

        """
        self content
        """
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # set video coding format
        # self.capture.set(cv2.CAP_PROP_FPS, 2)  # not work for real-time capture
        self.w = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)

        self._log(f'Start...\tWidth: {self.w} Height: {self.h} FPS: {self.fps}')

    # module entrance
    def run(self):
        if self.window_enable:
            # create window and set as top window
            cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.name, self.w, self.h)
            cv2.setWindowProperty(self.name, cv2.WND_PROP_TOPMOST, cv2.WINDOW_FULLSCREEN)

        # read and show frames
        while self.run_flag.value:
            # adaptive long skip when no object is detected
            while self.auto_inactive_skip_enable:
                if self.run_flag.value and self.AUTOSAVE_ENABLE and not self.autosave_flag.value:
                    time.sleep(0.1)
                else:
                    break

            rval, frame = self.capture.read()
            # Break the loop if the video capture object is not working
            if not rval:
                self._log('Camera capture failed!')
                break

            # put frame and time into queue
            self.save_queue.put({'source'   : 'camera',
                                 'data'     : frame,
                                 'timestamp': time.time(),
                                 })

            # display window
            if self.window_enable:
                cv2.imshow(self.name, frame)  # show frames
                cv2.waitKey(1)  # must use cv2 builtin keyboard functions if show frames in real-time

        # close window
        cv2.destroyAllWindows()
        self._log('Camera is off.')

    def _log(self, txt):  # print with device name
        print(f'[{self.__class__.__name__}]\t{txt}')

    def __del__(self):
        self.capture.release()
        self._log(f"Closed. Timestamp: {datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
        self.status['Module_CAM'] = False


def cam_proc_method(_run_flag, _shared_param_dict, _kwargs_CFG):
    c = Camera(_run_flag, _shared_param_dict, **_kwargs_CFG)
    c.run()


def monitor_method(_run_flag, _shared_param_dict):
    while _run_flag.value:
        _shared_param_dict['save_queue'].get()


if __name__ == '__main__':
    CAMERA_CFG = {
        'name'                     : 'Camera',
        'camera_index'             : 1,
        'capture_resolution'       : (1280, 720),  # (1280, 720) or (960, 540)
        'window_enable'            : True,  # if True, the radar data and camera data can not be saved together

        'auto_inactive_skip_enable': False,  # long skip all camera frames when no object is detected
    }
    kwargs_CFG = {'CAMERA_CFG'     : CAMERA_CFG,
                  'AUTOSAVE_ENABLE': False}

    run_f = Manager().Value('b', True)
    # generate save flag dict
    shared_param_d = {'save_queue'         : Manager().Queue(),
                      'mansave_flag'       : Manager().Value('c', None),  # set as None, 'image' or 'video', only triggered at the end of recording
                      'autosave_flag'      : Manager().Value('b', False),  # set as False, True or False, constantly high from the beginning to the end of recording
                      'compress_video_file': Manager().Value('c', None),  # the record video file waiting to be compressed
                      'email_image'        : Manager().Value('f', None),  # for image info from save_center to email_notifier module
                      'proc_status_dict': Manager().dict(),  # for process status
                      }
    proc_list = []

    camera_proc = Process(target=cam_proc_method, args=(run_f, shared_param_d, kwargs_CFG))
    proc_list.append(camera_proc)
    monitor_proc = Process(target=monitor_method, args=(run_f, shared_param_d))
    proc_list.append(monitor_proc)

    # start the processes and wait to finish
    for t in proc_list:
        t.start()
    for t in proc_list:
        t.join()

    # capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # 0为电脑内置摄像头
    # cv2.namedWindow('video', cv2.WINDOW_NORMAL)
    # while 1:
    #     ret, frame = capture.read()  # 摄像头读取,ret为是否成功打开摄像头,true,false。 frame为视频的每一帧图像
    #     # frame = cv2.flip(frame, 1)  # 摄像头是和人对立的，将图像左右调换回来正常显示。
    #     cv2.imshow("video", frame)
    #     c = cv2.waitKey(1)
    #     if c == 27:
    #         break
    #
    #     if cv2.waitKey(1) & 0xFF == ord('q'):  # 如果按下q 就截图保存并退出
    #         cv2.imwrite("test.png", frame)  # 保存路径
    #         break
    #
    # capture.release()
    # cv2.destroyAllWindows()
