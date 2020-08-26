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
from config import Configuration

config = Configuration(False, True)
config.add_argument('--map_name', type=str)
config.add_argument('--time_spacing', type=float, default=5.0)
config = config.parse()

scan_conv, scan_uncertainty = helpers.create_lu_networks(config.model_dir, config.model_epoch)
scan_conv.eval()
scan_uncertainty.eval()
convert_to_clouds = False

bag = rosbag.Bag(config.bag_file)
bag_reader = LCBagDataReader(bag, config.lidar_topic, config.localization_topic, convert_to_clouds, config.time_spacing, config.time_spacing)
  
for idx, timestamp in enumerate(bag_reader.get_localization_timestamps()):
  with torch.no_grad():
    scan_msg = bag_reader.get_closest_scan_by_time(timestamp)[0]
    loc = bag_reader.get_closest_localization_by_time(timestamp)[0]
    scan = torch.tensor(scan_msg.ranges).cuda()
    conv = scan_conv(scan.unsqueeze(0).unsqueeze(0))
    stats = scan_uncertainty(conv)[0]

    print ("{0} STATS: ({1}, {2})".format(timestamp, stats[0], stats[1]))
    # TODO do away with matplotlib
    import matplotlib.pyplot as plt
    plt.figure(0)
    evaluation_helpers.visualize_cloud(plt, scan_to_point_cloud(scan_msg), color='green')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('lc_results/scan_{0}.png'.format(timestamp))
    plt.clf()