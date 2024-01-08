import glob
import os

file_path = 'C:/SZC/PhD/MMWave_Radar/ID/data/MVB_Hall/Resolution_2/'
files_list = glob.glob(f'{file_path}*')

for idx, file in enumerate(files_list):
    os.rename(file, f'{file_path + str(idx)}')
pass
