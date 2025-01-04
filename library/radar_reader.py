"""
Designed for data collection from radars, abbr. RDR
"""

import multiprocessing
import re
import time
from datetime import datetime

import numpy as np
import serial

from library.TI.parser_mmw_demo import parser_one_mmw_demo_output_packet
from library.frame_early_processor import FrameEProcessor

header_length = 8 + 32
magic_word = b'\x02\x01\x04\x03\x06\x05\x08\x07'
stop_word = 'sensorStop'


class RadarReader:
    def __init__(self, run_flag, radar_rd_queue, shared_param_dict, **kwargs_CFG):
        """
        get shared values and queues
        """
        self.run_flag = run_flag
        self.radar_rd_queue = radar_rd_queue

        self.status = shared_param_dict['proc_status_dict']
        self.status[kwargs_CFG['RADAR_CFG']['name']] = True

        """
        pass config static parameters
        """
        """ module own config """
        RDR_CFG = kwargs_CFG['RADAR_CFG']
        self.name = RDR_CFG['name']
        self.cfg_port_name = RDR_CFG['cfg_port_name']
        self.data_port_name = RDR_CFG['data_port_name']
        self.cfg_file_name = RDR_CFG['cfg_file_name']

        """self content"""
        self.fep = FrameEProcessor(**kwargs_CFG)  # call other class
        self.cfg_port = None
        self.data_port = None

        self._log('Start...')

    # radar connection
    def connect(self) -> bool:
        try:
            cfg_port, data_port = self._connect_port(self.cfg_port_name, self.data_port_name)
            self._send_cfg(self._read_cfg(self.cfg_file_name), cfg_port, print_enable=1)
        except:
            return False
        # set property value
        self.cfg_port = cfg_port
        self.data_port = data_port
        return True

    # module entrance
    def run(self):
        if not self.connect():
            self._log(f"Radar {self.name} Connection Failed")
            self.run_flag.value = False

        data = b''
        while self.run_flag.value:
            data += self.data_port.read(self.data_port.inWaiting())  # may have incomplete frame which is not multiple of 32 Bytes

            # guarantee at lease 2 headers which is 1 frame in this data line
            if magic_word in data:
                data_cache = data[data.index(magic_word) + header_length:]
                if magic_word in data_cache:
                    # parse the data
                    parser_result, \
                        headerStartIndex, \
                        totalPacketNumBytes, \
                        numDetObj, \
                        numTlv, \
                        subFrameNumber, \
                        detectedX_array, \
                        detectedY_array, \
                        detectedZ_array, \
                        detectedV_array, \
                        detectedRange_array, \
                        detectedAzimuth_array, \
                        detectedElevation_array, \
                        detectedSNR_array, \
                        detectedNoise_array = parser_one_mmw_demo_output_packet(data, len(data), print_enable=0)
                    # put data into queue, convert list to nparray, and transpose from (channels, points) to (points, channels)
                    try:
                        frame = self.fep.FEP_accumulate_update(np.array((detectedX_array, detectedY_array, detectedZ_array, detectedV_array, detectedSNR_array)).transpose())
                    except:
                        # print('Data parser BROKEN!!!!!!!!!!!!!!!')
                        pass
                    self.radar_rd_queue.put(frame)
                    # self._log(str(len(data)) + '\t' + str(data))
                    # winsound.Beep(500, 200)

                    # remove the first header and get ready for next header detection
                    data = data_cache

    # connect the ports
    def _connect_port(self, cfg_port_name, data_port_name):
        try:
            cfg_port = serial.Serial(cfg_port_name, baudrate=115200)
            data_port = serial.Serial(data_port_name, baudrate=921600)
            assert cfg_port.is_open and data_port.is_open
            self._log('Hardware connected')
            return cfg_port, data_port
        except serial.serialutil.SerialException:
            return

    # read cfg file
    def _read_cfg(self, _cfg_file_name):
        cfg_list = []
        with open(_cfg_file_name) as f:
            lines = f.read().split('\n')
        for line in lines:
            if not (line.startswith('%') or line == ''):
                cfg_list.append(line)
        return cfg_list

    # send cfg list
    def _send_cfg(self, cfg_list, cfg_port, print_enable=1):
        for line in cfg_list:
            # send cfg line by line
            line = (line + '\n').encode()
            cfg_port.write(line)
            # wait for port response
            while cfg_port.inWaiting() <= 20:
                pass
            time.sleep(0.01)
            res_str = cfg_port.read(cfg_port.inWaiting()).decode()
            res_list = [i for i in re.split('\n|\r', res_str) if i != '']
            if print_enable == 1:
                self._log('\t'.join(res_list[-1:] + res_list[0:-1]))
        self._log('cfg SENT\n')

    # print with device name
    def _log(self, txt):
        print(f'[{self.name}]\t{txt}')

    def __del__(self):
        # stop the radar
        try:
            self._send_cfg([stop_word], self.cfg_port)
            self.cfg_port.close()
            self.data_port.close()
        except:
            pass
        self._log(f"Closed. Timestamp: {datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
        self.status[self.name] = True
        self.run_flag.value = False


if __name__ == '__main__':
    config = {'name'          : 'Test',
              'cfg_port_name' : 'COM4',
              'data_port_name': 'COM5',
              'cfg_file_name' : '../cfg/IWR1843_3D.cfg',
              'xlim'          : (-2, 2),
              'ylim'          : (0.2, 4),
              'zlim'          : (-2, 2),
              'pos_offset'    : (0, 0),  # default pos_offset is (0, 0)
              'facing_angle'  : 0,  # facing_angle is 0 degree(forward in field is up in map, degree 0-360 counted clockwise)
              'enable_save'   : True,  # data saved for single radar
              'save_length'   : 10,  # saved frame length for single radar
              }
    v = multiprocessing.Manager().Value('b', True)
    q = multiprocessing.Manager().Queue()
    r = RadarReader(v, q, config)
    success = r.connect()
    if not success:
        raise ValueError(f'Radar {config["name"]} Connection Failed')
    r.run()
