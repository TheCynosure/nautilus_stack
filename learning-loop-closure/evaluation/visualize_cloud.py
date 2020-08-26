import rospy
import numpy as np
from scipy import spatial
import argparse
from learning.dataset import LCDataset
from helpers import create_ros_pointcloud, publish_ros_pointcloud, visualize_location, visualize_cloud

parser = argparse.ArgumentParser(
    description='Visualize the models classification for given inputs')
parser.add_argument('--dataset', type=str, help='the dataset from which to pull the cloud')
parser.add_argument('--timestamp', type=str, help='timestamp within dataset to visualize. Only works if "dataset" is present.')
parser.add_argument('--cloud_file', type=str, help='exact path to cloud .npy file to visualize.')
parser.add_argument('--location_file', type=str, help='exact path to location .npy file to visualize.')
args = parser.parse_args()


if args.dataset:
    dataset = LCDataset(args.dataset)
    cloud, location, _ = dataset.get_by_timestamp(args.timestamp, include_angle=True)
elif args.cloud_file:
    cloud = np.load(args.cloud_file)
    if args.location_file:
        location = np.load(args.location_file)
    else:
        location = None

import matplotlib.pyplot as plt
plt.figure(1)
visualize_cloud(plt, cloud)

if location:
    plt.figure(2)
    visualize_location(plt, location, 'blue')

plt.show()

# msg = create_ros_pointcloud()
# publish_ros_pointcloud(anchor_pub, msg, anchor)
# publish_ros_pointcloud(similar_pub, msg, similar)
# publish_ros_pointcloud(distant_pub, msg, distant)
