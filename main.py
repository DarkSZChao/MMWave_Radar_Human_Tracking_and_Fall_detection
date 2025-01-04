"""
Auther: DarkSZChao
Date: 20/09/2024

In this project:
file_name is os.path.basename()
file_dir is os.path.dirname()
file_path is dir+name

All the limits are set as [a, b)
"""

import socket
import tkinter as tk
import tkinter.font as tkf
import warnings
from multiprocessing import Process, Manager
from time import sleep

import winsound

# import essential modules
from library import RadarReader, Visualizer, SyncMonitor

# import optional modules
try:
    from library import SaveCenter

    SVC_enable = True
except:
    pass
try:
    from library import Camera

    CAM_enable = True
except:
    pass
try:
    from library import EmailNotifier

    EMN_enable = True
except:
    pass
try:
    from library import VideoCompressor

    VDC_enable = True
except:
    pass

# import module configs
hostname = socket.gethostname()
if hostname == 'IT077979RTX2080':
    from cfg.config_queens059 import *
elif hostname == 'IT080027':
    from cfg.config_demo import *
elif hostname == 'SZC-LAPTOP-Pro':
    from cfg.config_demo import *
    # from cfg.config_cp107 import *
else:
    from cfg.config_demo import *
    raise warnings.warn('Hostname is not found! Default config is applied.')


# start the instance for the process
def radar_proc_method(_run_flag, _radar_rd_queue, _shared_param_dict, **_kwargs_CFG):
    radar = RadarReader(run_flag=_run_flag, radar_rd_queue=_radar_rd_queue, shared_param_dict=_shared_param_dict, **_kwargs_CFG)
    radar.run()


def vis_proc_method(_run_flag, _radar_rd_queue_list, _shared_param_dict, **_kwargs_CFG):
    vis = Visualizer(run_flag=_run_flag, radar_rd_queue_list=_radar_rd_queue_list, shared_param_dict=_shared_param_dict, **_kwargs_CFG)
    vis.run()


def monitor_proc_method(_run_flag, _radar_rd_queue_list, _shared_param_dict, **_kwargs_CFG):
    sync = SyncMonitor(run_flag=_run_flag, radar_rd_queue_list=_radar_rd_queue_list, shared_param_dict=_shared_param_dict, **_kwargs_CFG)
    sync.run()


def save_proc_method(_run_flag, _shared_param_dict, **_kwargs_CFG):
    save = SaveCenter(run_flag=_run_flag, shared_param_dict=_shared_param_dict, **_kwargs_CFG)
    save.run()


def camera_proc_method(_run_flag, _shared_param_dict, **_kwargs_CFG):
    cam = Camera(run_flag=_run_flag, shared_param_dict=_shared_param_dict, **_kwargs_CFG)
    cam.run()


def email_proc_method(_run_flag, _shared_param_dict, **_kwargs_CFG):
    email = EmailNotifier(run_flag=_run_flag, shared_param_dict=_shared_param_dict, **_kwargs_CFG)
    email.run()


def vidcompress_proc_method(_run_flag, _shared_param_dict, **_kwargs_CFG):
    vidcompress = VideoCompressor(run_flag=_run_flag, shared_param_dict=_shared_param_dict, **_kwargs_CFG)
    vidcompress.run()


def test_proc_method(_run_flag, _shared_param_dict):
    _run_flag = _run_flag
    _manual_save_flag = _shared_param_dict['mansave_flag']
    _auto_save_flag = _shared_param_dict['autosave_flag']

    def gui_button_style():
        style = tkf.Font(family='Calibri', size=16, weight=tkf.BOLD, underline=False, overstrike=False)
        return style

    def gui_label_style():
        style = tkf.Font(family='Calibri', size=18, weight=tkf.BOLD, underline=False, overstrike=False)
        return style

    def manual_save_image():
        _manual_save_flag.value = 'image'

    def manual_save_video():
        _manual_save_flag.value = 'video'

    def quit():
        _run_flag.value = False
        root.destroy()

    # create main window for display
    root = tk.Tk()
    root.withdraw()

    # sub-window for buttons
    window1 = tk.Toplevel()
    window1.wm_attributes('-topmost', 1)  # keep the window at the top
    window1.overrideredirect(True)  # remove label area
    window1.geometry('+40+510')
    window1.resizable(False, False)
    tk.Button(window1, text='Save_image', bg='lightblue', width=12, font=gui_button_style(), command=manual_save_image).grid(row=1, column=1)
    tk.Button(window1, text='Save_video', bg='lightblue', width=12, font=gui_button_style(), command=manual_save_video).grid(row=1, column=2)
    tk.Button(window1, text='Quit', bg='firebrick', width=12, font=gui_button_style(), command=quit).grid(row=1, column=3)

    # sub-window for process status
    window2 = tk.Toplevel()
    window2.wm_attributes('-topmost', 1)  # keep the window at the top
    window2.overrideredirect(True)  # remove label area
    window2.geometry('+670+60')
    window2.resizable(False, False)

    # create labels
    proc_status_label_dict = {}
    for _proc, _status in _shared_param_dict['proc_status_dict'].items():
        label_status = tk.Label(window2, text='', font=gui_label_style(), anchor="w")
        label_status.pack(fill="x", padx=20, pady=5)
        proc_status_label_dict[_proc] = label_status

    def update_status():
        for __proc, __status in _shared_param_dict['proc_status_dict'].items():
            proc_status_label_dict[__proc].config(text=f"{__proc}:\t{'ON' if __status else 'OFF'}")
            if __status:
                proc_status_label_dict[__proc].config(fg='green')
            else:
                proc_status_label_dict[__proc].config(fg='red')

        root.after(50, update_status)  # update

    # for update
    update_status()
    # for display
    root.mainloop()


# program entrance
if __name__ == '__main__':
    # generate shared variables between processes
    run_flag = Manager().Value('b', True)  # this flag control whole system running
    # generate save flag dict
    shared_param_dict = {'mansave_flag'       : Manager().Value('c', None),  # set as None, 'image' or 'video', only triggered at the end of recording
                         'autosave_flag'      : Manager().Value('b', False),  # set as False, True or False, constantly high from the beginning to the end of recording
                         'compress_video_file': Manager().Value('c', None),  # the record video file waiting to be compressed
                         'email_image'        : Manager().Value('f', None),  # for image info from save_center to email_notifier module
                         'proc_status_dict'   : Manager().dict(),  # for process status
                         }

    # generate shared queues
    radar_rd_queue_list = []  # radar rawdata queue list
    proc_list = []
    for RADAR_CFG in RADAR_CFG_LIST:
        radar_rd_queue = Manager().Queue()
        kwargs_CFG = {'RADAR_CFG': RADAR_CFG, 'FRAME_EARLY_PROCESSOR_CFG': FRAME_EARLY_PROCESSOR_CFG}
        radar_proc = Process(target=radar_proc_method, args=(run_flag, radar_rd_queue, shared_param_dict), kwargs=kwargs_CFG, name=RADAR_CFG['name'])
        radar_rd_queue_list.append(radar_rd_queue)
        proc_list.append(radar_proc)
    kwargs_CFG = {'VISUALIZER_CFG'          : VISUALIZER_CFG,
                  'RADAR_CFG_LIST'          : RADAR_CFG_LIST,
                  'MANSAVE_ENABLE'          : MANSAVE_ENABLE,
                  'AUTOSAVE_ENABLE'         : AUTOSAVE_ENABLE,
                  'FRAME_POST_PROCESSOR_CFG': FRAME_POST_PROCESSOR_CFG,
                  'DBSCAN_GENERATOR_CFG'    : DBSCAN_GENERATOR_CFG,
                  'BGNOISE_FILTER_CFG'      : BGNOISE_FILTER_CFG,
                  'HUMAN_TRACKING_CFG'      : HUMAN_TRACKING_CFG,
                  'HUMAN_OBJECT_CFG'        : HUMAN_OBJECT_CFG,
                  'SYNC_MONITOR_CFG'        : SYNC_MONITOR_CFG}
    # start essential processes
    vis_proc = Process(target=vis_proc_method, args=(run_flag, radar_rd_queue_list, shared_param_dict), kwargs=kwargs_CFG, name='Module_VIS')  # visualization process
    proc_list.append(vis_proc)
    monitor_proc = Process(target=monitor_proc_method, args=(run_flag, radar_rd_queue_list, shared_param_dict), kwargs=kwargs_CFG, name='Module_SCM')  # queue monitor process
    proc_list.append(monitor_proc)

    # optional processes, can be disabled
    try:
        kwargs_CFG.update({'SAVE_CENTER_CFG': SAVE_CENTER_CFG})
        shared_param_dict.update({'save_queue': Manager().Queue(maxsize=2000)})
        if SVC_enable:
            save_proc = Process(target=save_proc_method, args=(run_flag, shared_param_dict), kwargs=kwargs_CFG, name='Module_SVC')  # save center process
            proc_list.append(save_proc)
    except:
        pass
    try:
        kwargs_CFG.update({'CAMERA_CFG': CAMERA_CFG})
        if CAM_enable:
            camera_proc = Process(target=camera_proc_method, args=(run_flag, shared_param_dict), kwargs=kwargs_CFG, name='Module_CAM')  # camera process
            proc_list.append(camera_proc)
    except:
        pass
    try:
        kwargs_CFG.update({'EMAIL_NOTIFIER_CFG': EMAIL_NOTIFIER_CFG})
        if EMN_enable:
            email_proc = Process(target=email_proc_method, args=(run_flag, shared_param_dict), kwargs=kwargs_CFG, name='Module_EMN')  # email notifier process
            proc_list.append(email_proc)
    except:
        pass
    # try:
    #     kwargs_CFG.update({'VIDEO_COMPRESSOR_CFG': VIDEO_COMPRESSOR_CFG})
    #     if VDC_enable:
    #         vidcompress_proc = Process(target=vidcompress_proc_method, args=(run_flag, shared_param_dict), kwargs=kwargs_CFG)  # video compress process
    #         proc_list.append(vidcompress_proc)
    # except:
    #     pass

    test_proc = Process(target=test_proc_method, args=(run_flag, shared_param_dict), name='Module_GUI')  # GUI process
    proc_list.insert(0, test_proc)
    for proc in proc_list:
        if proc.name == 'Module_GUI':
            shared_param_dict['proc_status_dict'][proc.name] = True
        else:
            shared_param_dict['proc_status_dict'][proc.name] = False

    # start the processes and wait to finish
    for t in proc_list:
        t.start()
        sleep(0.5)
    for t in proc_list:
        t.join()
        sleep(0.2)

    winsound.Beep(1000, 500)
