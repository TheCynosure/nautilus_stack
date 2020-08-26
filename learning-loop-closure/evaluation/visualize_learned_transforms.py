import argparse

import rosbag
import numpy as np
import torch

import os
import sys
import evaluation_helpers

sys.path.append(os.path.join(os.getcwd(), '..'))
import helpers
from data_processing.data_processing_helpers import LCBagDataReader, scan_to_point_cloud
from config import data_generation_config, Configuration

config = Configuration(False, True)
config.add_argument('--alt_bag_file', type=str)
config.add_argument('--time_spacing', type=float, default=1.5)
config = config.parse()

scan_conv, scan_match, scan_transform = helpers.create_laser_networks(config.model_dir, config.model_epoch)
scan_conv.eval()
scan_match.eval()
scan_transform.eval()
convert_to_clouds = False

bag = rosbag.Bag(config.bag_file)

if config.alt_bag_file:
  alt_bag = rosbag.Bag(config.alt_bag_file)
else:
  alt_bag = rosbag.Bag(config.bag_file)

bag_reader = LCBagDataReader(bag, config.lidar_topic, config.localization_topic, convert_to_clouds, config.time_spacing, config.time_spacing)

with torch.no_grad():
  # Walk through the bag and get some transforms, try to recover them
  for base_idx in np.linspace(0, 200, 21):
    base_idx = int(base_idx)
    base_timestamp = bag_reader.get_localization_timestamps()[base_idx]
    base_loc = bag_reader.get_localizations()[base_timestamp]

    nearby_localizations, nearby_location_timestamps, nearby_location_indices = bag_reader.get_nearby_locations(base_loc[:2], 1.5)
    idx = np.random.randint(0, len(nearby_localizations))
    alt_loc, alt_timestamp, alt_idx = nearby_localizations[idx], nearby_location_timestamps[idx], nearby_location_indices[idx]
    
    base_scan = torch.tensor(bag_reader.get_closest_scan_by_time(base_timestamp)[0].ranges).cuda()
    alt_scan = torch.tensor(bag_reader.get_closest_scan_by_time(alt_timestamp)[0].ranges).cuda()

    conv = scan_conv(base_scan.unsqueeze(0), alt_scan.unsqueeze(0))
    predicted_transform = scan_transform(conv).squeeze()
    import pdb; pdb.set_trace()
  plt.show()