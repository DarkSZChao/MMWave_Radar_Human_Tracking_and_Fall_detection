"""
Designed to compress the video after save, abbr. VDC
"""

import os
import time
from datetime import datetime
from multiprocessing import Manager

from moviepy.editor import VideoFileClip


class VideoCompressor:
    def __init__(self, run_flag, shared_param_dict, **kwargs_CFG):
        """
        get shared values and queues
        """
        self.run_flag = run_flag
        # shared params
        self.compress_video_file = shared_param_dict['compress_video_file']  # sent from save_center
        self.status = shared_param_dict['proc_status_dict']
        self.status['Module_VDC'] = True

        """
        pass config static parameters
        """
        """ module own config """
        VDC_CFG = kwargs_CFG['VIDEO_COMPRESSOR_CFG']
        self.target_bitrate = VDC_CFG['target_bitrate']

        """
        self content
        """
        self._log('Start...')

    # module entrance
    def run(self):
        while self.run_flag.value:
            if self.compress_video_file.value:
                input_file = self.compress_video_file.value
                output_file = os.path.join(os.path.dirname(input_file), '.'.join([os.path.basename(input_file).split('.')[0] + '_compressed', os.path.basename(input_file).split('.')[-1]]))
                self.compress_video_file.value = None
                self.compress_video(input_file, output_file)
            time.sleep(1)

    def compress_video(self, _input_file, _output_file):
        video = VideoFileClip(_input_file)
        video.write_videofile(_output_file, bitrate=self.target_bitrate)

    def _log(self, txt):  # print with device name
        print(f'[{self.__class__.__name__}]\t{txt}')

    def __del__(self):
        self._log(f"Closed.")
        self.status['Module_VDC'] = False


if __name__ == '__main__':
    # generate save flag dict
    shared_param = {'compress_video_file': None,  # the record video file waiting to be compressed
                    }
    VIDEO_COMPRESSOR_CFG = {
        # video quality
        'target_bitrate': '2000k',
    }
    vc = VideoCompressor(Manager().Value('b', True), shared_param, **{'VIDEO_COMPRESSOR_CFG': VIDEO_COMPRESSOR_CFG})

    import glob
    from send2trash import send2trash

    file_dir = '../data/MVB_501/**/'
    file_path_list = glob.glob(os.path.join(file_dir, '*.mp4'), recursive=True)
    for i, f_path in enumerate(file_path_list):
        f_dir = os.path.dirname(f_path)
        f_name = os.path.basename(f_path).split('.')
        f_name.insert(-1, '_comp.')
        f_outputpath = os.path.join(f_dir, ''.join(f_name))
        vc.compress_video(f_path, f_outputpath)
        time.sleep(2)
        send2trash(f_path)
        time.sleep(2)
        os.renames(f_outputpath, f_path)
        print(f'Finish [{i + 1}/{len(file_path_list)}]')
