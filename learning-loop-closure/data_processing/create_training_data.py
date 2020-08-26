import rosbag
import argparse
import numpy as np
import os
import sys
import random
import json
from tqdm import tqdm
from scipy import spatial
from data_processing_helpers import scan_to_point_cloud, get_scans_and_localizations_from_bag

sys.path.append(os.path.join(os.getcwd(), '..'))
from config import data_generation_config

parser = argparse.ArgumentParser(
    description='Create some training data from ROS bags')
parser.add_argument('--bag_file', type=str,
                    help='path to the bag file containing training data')
parser.add_argument('--dataset_name', type=str,
                    help='defines the folder in which this generated data will be placed')
parser.add_argument('--info_only', type=bool,
                    help='if set, only write dataset_info.json, assuming the data has already been written')

args = parser.parse_args()
bag = rosbag.Bag(args.bag_file)

start_timestamp = bag.get_start_time()

dataset_info = {
    'name': args.dataset_name,
    'sourceBag': os.path.abspath(args.bag_file),
    'startTime': start_timestamp,
}

scans, localizations, metadata = get_scans_and_localizations_from_bag(bag, data_generation_config['LIDAR_TOPIC'], data_generation_config['LOCALIZATION_TOPIC'], data_generation_config['TIME_SPACING'], data_generation_config['TIME_SPACING'])

dataset_info['scanMetadata'] = metadata
dataset_info['numScans'] = len(scans.keys())

loc_timestamps = sorted(localizations.keys())
localizationTree = spatial.cKDTree([list([l]) for l in loc_timestamps])

base_path = '../data/' + args.dataset_name + '/'
if not os.path.exists(base_path):
    os.makedirs(base_path)

if not args.info_only:
    print("Writing data to disk for {0} scans...".format(
        dataset_info['numScans']))
filenames = []
for timestamp, cloud in tqdm(list(scans.items())):
    d, indices = localizationTree.query([timestamp], k=2)

    before_timestamp = loc_timestamps[indices[0]]
    after_timestamp = loc_timestamps[indices[1]]
    time_interval = (after_timestamp - before_timestamp)
    before_weight = 1 - (timestamp - before_timestamp) / time_interval
    after_weight = 1 - (after_timestamp - timestamp) / time_interval
    before_loc = localizations[before_timestamp]
    after_loc = localizations[after_timestamp]
    
    timestamp = round(timestamp, 5)

    location = before_loc * before_weight + after_loc * after_weight
    cloud_file_name = 'point_' + str(timestamp)
    if not args.info_only:
        np.save(base_path + cloud_file_name, cloud)
        loc_file_name = 'location_' + str(timestamp)
        np.save(base_path + loc_file_name, location)
    filenames.append(cloud_file_name)

print("Writing dataset information...", len(filenames))

count = len(filenames)
train_indices = random.sample(
    list(range(count)), int(round(count * data_generation_config['TRAIN_SPLIT'])))
dev_indices = random.sample(list(range(count)), int(round(count * data_generation_config['DEV_SPLIT'])))
val_indices = set(range(count)) - set(dev_indices)

dataset_info['train_data'] = [filenames[i] for i in train_indices]
dataset_info['dev_data'] = [filenames[i] for i in dev_indices]
dataset_info['val_data'] = [filenames[i] for i in val_indices]

info_path = os.path.join(base_path, 'dataset_info.json')
with open(info_path, 'w') as f:
    f.write(json.dumps(dataset_info, indent=2))
