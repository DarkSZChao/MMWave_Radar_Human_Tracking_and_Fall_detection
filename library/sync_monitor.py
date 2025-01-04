"""
Designed to monitor and sync the queues, abbr. SCM
"""

from datetime import datetime
from multiprocessing import Manager
from time import sleep

import numpy as np


class SyncMonitor:
    def __init__(self, run_flag, radar_rd_queue_list, shared_param_dict, **kwargs_CFG):
        """
        get shared values and queues
        """
        self.run_flag = run_flag
        # radar rawdata queue list
        self.radar_rd_queue_list = radar_rd_queue_list
        # shared params
        try:
            self.save_queue = shared_param_dict['save_queue']
        except:
            self.save_queue = Manager().Queue(maxsize=0)
        self.status = shared_param_dict['proc_status_dict']
        self.status['Module_SCM'] = True

        """
        pass config static parameters
        """
        """ module own config """
        SCM_CFG = kwargs_CFG['SYNC_MONITOR_CFG']
        self.rd_qsize_warning = SCM_CFG['rd_qsize_warning']
        self.sc_qsize_warning = SCM_CFG['sc_qsize_warning']

        """
        self content
        """
        self._log('Start...')

    def run(self):
        while self.run_flag.value:
            # monitor radar queues
            rd_qsize_np = np.array([self.radar_rd_queue_list[i].qsize() for i in range(len(self.radar_rd_queue_list))])
            if sum(rd_qsize_np) > self.rd_qsize_warning:
                self._log(f'Radar queue size: {list(rd_qsize_np)}, pls save data until no traffic!')
            # sync the queue list when there are multiple queue
            if len(rd_qsize_np) > 1:
                diff_np = rd_qsize_np - min(rd_qsize_np)
                # need calibration when out of sync
                if max(diff_np) > 1 and min(rd_qsize_np) == 0:
                    diff_np = diff_np - 1  # allow diff no more than 1
                    diff_np[diff_np < 0] = 0
                    # for each queue
                    for q_index in range(len(self.radar_rd_queue_list)):
                        for _ in range(diff_np[q_index]):
                            self.radar_rd_queue_list[q_index].get(block=True, timeout=1)
                            self._log(f'Self Sync Calibration: Queue {q_index}')

            # monitor SaveCenter queue
            sc_qsize = self.save_queue.qsize()
            if sc_qsize > self.sc_qsize_warning:
                self._log(f'SaveCenter queue size: {sc_qsize}, pls save data until no traffic!')

            sleep(2)

    def _log(self, txt):  # print with device name
        print(f'[{self.__class__.__name__}]\t{txt}')

    def __del__(self):
        self._log(f"Closed. Timestamp: {datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
        self.status['Module_SCM'] = False
        self.run_flag.value = False
