import os, sys
from pathlib import Path

from util.utils_data import *

print("Generate synthetic segmentation dataset.")

root_dir = Path(__file__).resolve().parents[1]
save_path = os.path.join(root_dir, 'Data/MID_1n1e_SUFFIX/') # save in folder

cpi = 1
past = 3
minT = 10
maxT = 10
sim_time_per_scene = 60 # times
index_list = [1,2,3,4,5,6,7,8,9] # each index should have at least 60 trajectories
# Test: [10,11,12]

save_MID_data(index_list, save_path, sim_time_per_scene, channel_per_image=cpi)

gen_csv_trackers(save_path) # generate CSV tracking files first
print('CSV records for each object generated.')

gather_all_data(save_path, past, maxT=maxT, minT=minT, channel_per_image=cpi) # go through all the obj folders and put them together in one CSV
print('Final CSV generated!')