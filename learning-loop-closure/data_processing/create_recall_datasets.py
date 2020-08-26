import rospy
import argparse
import numpy as np
import statistics
import math
from geometry_msgs.msg import Pose
import roslib
roslib.load_manifest('cobot_msgs')
from cobot_msgs.msg import CobotLocalizationMsg

from scipy.spatial import cKDTree
from helpers import compute_overlap, normalize_point_cloud
import os
import json
from tqdm import tqdm
import glob

parser = argparse.ArgumentParser(
    description='Find loop closure locations for some ROS bag')
parser.add_argument('--first_dataset', type=str, help='path to first dataset')
parser.add_argument('--second_dataset', type=str, help='path to second dataset')
parser.add_argument('--out_dataset', default='recall_evall_dataset', type=str, help='path to output dataset')
args = parser.parse_args()

scans = {}
localizations = {}

out_dir = args.out_dataset

try:
    os.makedirs(out_dir)
except OSError:
    pass

target_locations = list()

def save_target_locations():
    location_file = os.path.join(out_dir, 'target_locations.json')
    loc_info = json.dumps(target_locations, indent=2)
    with open(location_file, 'w') as f:
        f.write(loc_info)

def save_dataset_info():
    dataset_file = os.path.join(out_dir, 'datasets.json')
    ds_info = json.dumps([args.first_dataset, args.second_dataset])
    with open(dataset_file, 'w') as f:
        f.write(ds_info)

def handle_location_input(publisher):
    def handle_input(data):
        print("Recieved Input from localization gui", data)
        print("Possible angle", math.atan2(data.orientation.z, data.orientation.w))
        angle = math.atan2(data.orientation.z, data.orientation.w)
        location = (data.position.x, data.position.y, angle)
        print("Target pose: ", location)
        target_locations.append(location)

    return handle_input

def location_listener():
    rospy.init_node('evaluate_recall', anonymous=True)
    pub = rospy.Publisher('Cobot/Localization', Pose, queue_size=5)
    rospy.Subscriber('localization_gui/nav_goal', Pose, handle_location_input(pub))
    rospy.spin()

print("Please run localization_gui to provide target locations. Press ctrl+C when done")
location_listener()

save_target_locations()
print('Saved target location information')

save_dataset_info()

# Now, find the locations in the datasets that are close to this spot.

def load_locations(ds_dir):
    location_files = glob.glob(os.path.join(ds_dir, 'location_*.npy'))
    locations = np.ndarray((len(location_files), 3)).astype(np.float32)
    for idx, f in tqdm(enumerate(location_files)):
        loc = np.load(f).astype(np.float32)
        locations[idx] = loc

    return locations, location_files

ds1_locations, ds1_files = load_locations(args.first_dataset)
ds2_locations, ds2_files = load_locations(args.second_dataset)

ds1_timestamps = np.asarray([name[name.find('location_') + len('location_'):name.find('.npy')] for name in ds1_files])
ds2_timestamps = np.asarray([name[name.find('location_') + len('location_'):name.find('.npy')] for name in ds2_files])

ds1_time_tree = cKDTree(np.asarray([d[:2] for d in ds1_locations]))
ds2_time_tree = cKDTree(np.asarray([d[:2] for d in ds2_locations]))

THRESHOLD = 0.75

def check_overlap(target_loc, locations):
    def checker(loc_idx):
        overlap = compute_overlap(target_loc, locations[loc_idx])
        return overlap > THRESHOLD
    
    return checker

# For each target location, find a fixed number of nearby observations in each dataset
obs_per_target = 5
observation_loc_timestamps = np.ndarray((len(target_locations), 2, obs_per_target, 1))
observation_scan_files = np.ndarray((len(target_locations), 2, obs_per_target, 1))
for idx, target_loc in enumerate(target_locations):
    ds1_neighbors = ds1_time_tree.query_ball_point(target_loc[:2], 1)
    filtered_ds1 = np.asarray(list(filter(check_overlap(target_loc, ds1_locations), ds1_neighbors)))
    assert(len(filtered_ds1) >= obs_per_target)
    ds2_neighbors = ds2_time_tree.query_ball_point(target_loc[:2], 1)
    filtered_ds2 = np.asarray(list(filter(check_overlap(target_loc, ds2_locations), ds2_neighbors)))
    assert(len(filtered_ds2) >= obs_per_target)

    filtered_ds1 = np.random.choice(filtered_ds1, obs_per_target)
    filtered_ds2 = np.random.choice(filtered_ds2, obs_per_target)
    
    observation_loc_timestamps[idx, 0] = [[ts,] for ts in ds1_timestamps[filtered_ds1]]
    observation_loc_timestamps[idx, 1] = [[ts,] for ts in ds2_timestamps[filtered_ds2]]

np.save(os.path.join(out_dir, 'observation_timestamps.npy'), observation_loc_timestamps)