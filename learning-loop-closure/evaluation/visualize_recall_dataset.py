import rospy
import numpy as np
from scipy import spatial
import argparse
from learning.dataset import LCDataset
from helpers import visualize_location, visualize_cloud, visualize_location, draw_map
import os
import json

parser = argparse.ArgumentParser(
    description='Visualize the models classification for given inputs')
parser.add_argument('--recall_dataset', type=str, help='the dataset to visualize')
args = parser.parse_args()

obs_timestamps = np.load(os.path.join(args.recall_dataset, 'observation_timestamps.npy'))

ds_file = os.path.join(args.recall_dataset, 'datasets.json')
with open(ds_file) as f:
    dataset_names = json.loads(f.read())

location_file = os.path.join(args.recall_dataset, 'target_locations.json')
with open(location_file) as f:
    locations = json.loads(f.read())

def load_cloud(ds, timestamp):
    dataset_name = dataset_names[ds]
    cloud_data = os.path.join(dataset_name, 'point_{0}.npy'.format(timestamp))
    return np.load(cloud_data)

import matplotlib.pyplot as plt

for ds in range(obs_timestamps.shape[1]):
    for location_idx in range(obs_timestamps.shape[2]):
        location_timestamps = obs_timestamps[:, ds, location_idx]
        clouds = [load_cloud(ds, lt[0]) for lt in location_timestamps]
        plt.figure(1, figsize=(len(clouds), 1))
        plt.suptitle('Location {0}'.format(location_idx))
        for i in range(len(clouds)):
            plt.subplot(int('1{0}{1}'.format(len(clouds), i+1)))
            visualize_cloud(plt, clouds[i])
        plt.figure(2)
        visualize_location(plt, locations[location_idx])
        draw_map(plt, '../../cobot/maps/GDC3/GDC3_vector.txt')
        plt.title('Location {1}')
        plt.show()


# msg = create_ros_pointcloud()
# publish_ros_pointcloud(anchor_pub, msg, anchor)
# publish_ros_pointcloud(similar_pub, msg, similar)
# publish_ros_pointcloud(distant_pub, msg, distant)
