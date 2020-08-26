import argparse
import numpy as np
import rosbag
from matplotlib import pyplot as plt
import json

import sys
import os
sys.path.append(os.path.join(os.getcwd(), '..'))
from data_processing.data_processing_helpers import LCBagDataReader
from evaluation.evaluation_helpers import visualize_cloud, visualize_location, draw_map
from config import data_generation_config

parser = argparse.ArgumentParser()
parser.add_argument(
    '--bag_file', type=str, required=True, help='bag file to walk through for evaluation scans')
parser.add_argument(
    '--map_name', type=str, required=True, help='map over which bag file is walking')
parser.add_argument(
    '--dataset', type=str, default='manual_evaluation', help='output evaluation dataset')

args = parser.parse_args()
# The Basic idea: Walk through all the scans in a bag file, along with their locations, and present them to the user
# Store those frames the user marks as "keyframes", along with their locations, in a dataset

map_file = '../../cobot/maps/{0}/{0}_vector.txt'.format(args.map_name)

dataset_info = {}

bag = rosbag.Bag(args.bag_file)
start_timestamp = bag.get_start_time()

data_reader = LCBagDataReader(bag, data_generation_config['LIDAR_TOPIC'], data_generation_config['LOCALIZATION_TOPIC'])

pairs = sorted(data_reader.get_scans().items(), lambda x, y: 1 if float(x[0]) > float(y[0]) else -1)

accepted = []

for timestamp, cloud in pairs[0::50]:
  loc = data_reader.get_closest_localization_by_time(timestamp)[0]
  plt.figure(1)
  plt.clf()
  visualize_cloud(plt, cloud, color='blue')
  plt.figure(2)
  plt.clf()
  draw_map(plt, map_file)
  visualize_location(plt, loc, 'green')

  plt.show(block=False)
  accept = str(raw_input('Is this scan a keyframe? (y/n): '))

  if accept == 'y' or accept == 'Y':
    accepted.append((timestamp, loc, cloud))
  elif accept =='break':
    break

dataset_info['numScans'] = len(accepted)
dataset_info['name'] = args.dataset
dataset_info['scanMetadata'] = data_reader.metadata
dataset_info['sourceBag'] = args.bag_file
dataset_info['startTime'] = start_timestamp
dataset_info['mapName'] = args.map_name

base_path = args.dataset + '_' + args.map_name
info_path = os.path.join(base_path, 'dataset_info.json')
with open(info_path, 'w') as f:
    f.write(json.dumps(dataset_info, indent=2))

for timestamp, loc, cloud in accepted:
    stamp = str(round(timestamp, 5))
    np.save(os.path.join(base_path, 'location_{0}.npy'.format(stamp)), loc)
    np.save(os.path.join(base_path, 'point_{0}.npy'.format(stamp)), cloud)