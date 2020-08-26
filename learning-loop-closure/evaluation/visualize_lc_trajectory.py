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
config.add_argument('--alt_bag_file', type=str)
config.add_argument('--time_spacing', type=float, default=1.5)
config.add_argument('--interactive', action='store_true')
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
alt_reader = LCBagDataReader(alt_bag, config.lidar_topic, config.localization_topic, convert_to_clouds, config.time_spacing, config.time_spacing)

base_trajectory = []
for timestamp in bag_reader.get_localization_timestamps():
  loc = bag_reader.get_localizations()[timestamp]
  base_trajectory.append(loc[:2])
base_trajectory = np.array(base_trajectory)


target_trajectory = []
for timestamp in alt_reader.get_localization_timestamps():
  loc = alt_reader.get_localizations()[timestamp]
  target_trajectory.append(loc[:2])
target_trajectory = np.array(target_trajectory)

#Set up trajectory plot
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

def setup_plot():
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot(base_trajectory[:, 0], base_trajectory[:, 1], 0, color='blue')
  ax.plot(target_trajectory[:, 0], target_trajectory[:, 1], 5, color='green')
  return ax, fig

def add_matches_for_index(base_idx):
  tp = 0
  fp = 0
  base_timestamp = bag_reader.get_localization_timestamps()[base_idx]
  base_scan = torch.tensor(bag_reader.get_closest_scan_by_time(base_timestamp)[0].ranges).cuda()

  for idx, timestamp in enumerate(alt_reader.get_localization_timestamps()):
    with torch.no_grad():
      scan = torch.tensor(alt_reader.get_closest_scan_by_time(timestamp)[0].ranges).cuda()
      conv = scan_conv(base_scan.unsqueeze(0), scan.unsqueeze(0))
      scores = scan_match(conv)
      prediction = torch.argmax(torch.nn.functional.softmax(scores, dim=1), dim=1)

    if (prediction):
      if np.linalg.norm(base_trajectory[base_idx] - target_trajectory[idx]) < 2.5:
        tp += 1
        c = 'gray'
        if (config.bag_file == config.alt_bag_file):
          if (abs(base_idx - target_trajectory) > 5):
            c = 'green'
      else:
        fp +=1 
        c = 'red'
      ax.plot([base_trajectory[base_idx, 0], target_trajectory[idx, 0]], [base_trajectory[base_idx, 1], target_trajectory[idx, 1]], [0, 5], color=c)

  return tp, fp

def finish_and_show_plot(ax, fig, TP, FP):
  info_string = "True Positives: {0}\tFalse Positives: {1}".format(TP, FP)

  legend_elements = [
    plt.Line2D([0], [0], color='gray', lw=4, label='Correct Match ({0})'.format(TP)),
    Line2D([0], [0], color='red', lw=4, label='Incorrect Match ({0})'.format(FP)),
  ]

  ax.legend(handles=legend_elements)
  print(info_string)
  # plt.text(10, 10, info_string)
  plt.show()

if config.interactive:
  idx = 0
  while idx < len(bag_reader.get_localization_timestamps()):
    idx = input("Enter index:")
    ax, fig = setup_plot()
    tp, fp = add_matches_for_index(idx)
    finish_and_show_plot(ax, fig)

# Now find loop closures along these trajectories
else:
  FP = 0
  TP = 0
  ax, fig = setup_plot()
  for base_idx in np.linspace(0, len(bag_reader.get_localization_timestamps()) - 1, 50):
    # if response == 'y':
    base_idx = int(base_idx)
    tp, fp = add_matches_for_index(base_idx)
    FP += fp
    TP += tp

  finish_and_show_plot(ax, fig, TP, FP)
