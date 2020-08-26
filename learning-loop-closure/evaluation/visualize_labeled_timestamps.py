import rosbag
import rospy
import numpy as np
from scipy import spatial
import torch
import argparse
from learning.dataset import LCDataset
from helpers import visualize_cloud
import random

parser = argparse.ArgumentParser(
    description='Visualize the models classification for given inputs')
parser.add_argument('--dataset', type=str, help='the dataset from which to pull the triplet')
parser.add_argument('--timestamps', type=str, help='timestamps file to visualize')
parser.add_argument('--pos_only', type=str, help='only visualize positive timestamps')
args = parser.parse_args()

timestamps = np.load(args.timestamps)
dataset = LCDataset(args.dataset)

import matplotlib.pyplot as plt

indices = range(len(timestamps))
random.shuffle(indices)

for i in indices:
    timestamp = timestamps[i][0]
    if float(timestamp) > 5e5: # must be since epoch...
        timestamp = round(float(timestamp) - dataset.dataset_info['startTime'], 5)
    label = int(timestamps[i][1])

    cloud, loc, ts = dataset.get_by_nearest_timestamp(timestamp)
    plt.suptitle('Timestamp: ' + str(timestamp))
    plt.figure(1)
    if label:
        visualize_cloud(plt, cloud, color='green')
        if len(timestamps[i]) == 4: # must be covariance, has scale and condition
            plt.title('Condition: {0} Scale: {1}'.format(timestamps[i][2],timestamps[i][3]))
        elif len(timestamps[i]) == 3: #must be uniqueness, has just score
            plt.title('Uniqueness Score: {0}'.format(timestamps[i][2]))
        plt.show()
    elif not args.pos_only:
        visualize_cloud(plt, cloud, color='red')
        if len(timestamps[i]) == 4: # must be covariance, has scale and condition
            plt.title('Condition: {0} Scale: {1}'.format(timestamps[i][2],timestamps[i][3]))
        elif len(timestamps[i]) == 3: #must be uniqueness, has just score
            plt.title('Uniqueness Score: {0}'.format(timestamps[i][2]))

        plt.show()