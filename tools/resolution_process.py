import glob
import pickle

from library.utils import *
from tools.data_load import Visualizer

file_path = 'C:/SZC/PhD/MMWave_Radar/ID/data/MVB_Hall/Resolution_2/'
files_list = glob.glob(f'{file_path}*')
data_list = []
data_np = np.ndarray([0, 5])

for idx, file in enumerate(files_list):
    with open(file, 'rb') as f:
        data = pickle.load(f)[0]['IWR1843_Ori']
        data, _ = np_filter(data, axis=0, range_lim=(-0.3, 0.3))
        data, _ = np_filter(data, axis=1, range_lim=(0.9 + idx * 0.01, 1.1 + idx * 0.01))
        data, _ = np_filter(data, axis=2, range_lim=(0.6, 1.2))

        # data = np_repeated_points_removal(data, axes=(0, 1, 2))
        # data_list.append({'IWR1843_Ori': data})
        data_np = np.concatenate([data_np, data])
data_np = np_repeated_points_removal(data_np, axes=(0, 1, 2))

# with open(f'resolution', 'wb') as file:
#     pickle.dump(data_np, file)

data_list.append({'IWR1843_Ori': data_np})


vis = Visualizer(data_list[0:], interval=50, image_output_enable=False)
vis.run()
