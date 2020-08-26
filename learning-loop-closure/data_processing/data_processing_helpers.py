import numpy as np
import torch
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import Pose, PoseStamped
from rospy import rostime
from scipy import spatial
from tqdm import tqdm

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))
from config import data_config, lidar_config, data_generation_config

test_distances = np.linspace(0.25, data_config['OVERLAP_RADIUS'], num=data_config['OVERLAP_SAMPLE_RESOLUTION'])

def fix_angle(theta):
    if (theta < 0):
        theta += np.pi * 2
    if (theta > np.pi * 2):
        theta -= np.pi * 2
    return theta

def test_point(loc, point):
    center = loc[:2]
    relative_point = point - center
    theta = fix_angle(np.arctan2(relative_point[1], relative_point[0]))
    orientation = fix_angle(loc[2])
    start = fix_angle(orientation - lidar_config['FOV'] / 2)
    end = fix_angle(orientation + lidar_config['FOV'] / 2)

    if start < end and (theta < start or theta > end):
        return False
    elif end < start and (theta < start and theta > end):
        return False
    
    distance = np.linalg.norm(relative_point)
    if distance <= data_config['OVERLAP_RADIUS']:
        return True
    return False

def get_test_points(location):
    orientation = fix_angle(location[2])
    start = orientation - lidar_config['FOV'] / 2
    end = orientation + lidar_config['FOV'] / 2
    test_angles = np.linspace(start, end, num=data_config['OVERLAP_SAMPLE_RESOLUTION'])

    return np.concatenate(
        [[location[:2] + d * np.array([np.cos(angle), np.sin(angle)]) for d in test_distances]  for angle in test_angles]
    )

def compute_overlap(loc_a, loc_b):
    test_points = get_test_points(loc_a)
    matches = 0.0
    for point in test_points:
        if (test_point(loc_b, point)):
            matches += 1.0
    match_ab = matches / len(test_points)

    test_points = get_test_points(loc_b)
    matches = 0.0
    for point in test_points:
        if (test_point(loc_a, point)):
            matches += 1.0    
    match_ba = matches / len(test_points)

    return np.average([match_ab, match_ba])

def scan_to_point_cloud(scan, trim_edges=True, normalize=False):
    angle_offset = scan.angle_min
    num_scans = len(scan.ranges) if not trim_edges else int((scan.angle_max - scan.angle_min - 2 * np.pi / 12.0) / scan.angle_increment) + 1
    cloud = np.zeros((num_scans, 2)).astype(np.float32)
    point_idx = 0
    for idx,r in enumerate(scan.ranges):
        if trim_edges and (angle_offset < scan.angle_min + np.pi / 12.0 or angle_offset > scan.angle_max - np.pi / 12.0):
            pass #print("SKIPPING", idx)
        else:
            if r >= scan.range_min and r <= scan.range_max:
                point = np.array([r, 0])
                cos, sin = np.cos(angle_offset), np.sin(angle_offset)
                rotation = np.array([(cos, -sin), (sin, cos)])
                point = np.matmul(rotation, point)
                cloud[point_idx][0] = point[0]
                cloud[point_idx][1] = point[1]
            point_idx += 1
        angle_offset += scan.angle_increment

    if normalize:
        cloud = normalize_point_cloud(cloud, scan.range_max)
    # else:
    #     cloud = np.delete(cloud, 2, axis=1)

    return cloud

def normalize_point_cloud(point_set, max_range=lidar_config['MAX_RANGE'], delete_axis=True):
    point_set = point_set - \
        np.expand_dims(np.mean(point_set, axis=0), 0)  # center
    # if delete_axis:
    #     point_set = np.delete(point_set, 2, axis=1)
    # normalize
    point_set = point_set / max_range  # scale
    return point_set

def partition_point_cloud(point_set, threshold):
    prev_pt = point_set[0]
    partitions = []
    curr_partition = []
    for point in point_set:
        if (np.linalg.norm(point - prev_pt) < threshold) and len(curr_partition) < data_generation_config['MAX_PARTITION_SIZE']:
            curr_partition.append(point)
        elif len(curr_partition) > data_generation_config['MIN_PARTITION_SIZE']:
            partitions.append(curr_partition)
            curr_partition = []
        else:
            curr_partition = []
        prev_pt = point

    if len(curr_partition) > data_generation_config['MIN_PARTITION_SIZE']:
        partitions.append(curr_partition)

    lengths = [len(p) for p in partitions]
    length = len(partitions)

    sorted_indices = np.argsort(lengths)[::-1][:data_generation_config['MAX_PARTITION_COUNT']] # Make sure we only retain the largest partitions
    partition_array = np.zeros((data_generation_config['MAX_PARTITION_COUNT'], data_generation_config['MAX_PARTITION_SIZE'] + 1, 2)).astype(np.float32)
    for i in range(data_generation_config['MAX_PARTITION_COUNT']):
        if i in sorted_indices:
            center = np.mean(partitions[i], axis=0)
            centered_partition = np.array(partitions[i]) - center
            partition_array[i, 0:len(partitions[i]), :] = centered_partition
            partition_array[i, data_generation_config['MAX_PARTITION_SIZE']:data_generation_config['MAX_PARTITION_SIZE']+1, :] = center

    # from matplotlib import pyplot as plt
    # import random
    # plt.figure()
    # plt.scatter(point_set[:, 0], point_set[:, 1])
    # lower_bound = min(np.min(point_set[:, 0]), np.min(point_set[:, 1]))
    # upper_bound = max(np.max(point_set[:, 0]), np.max(point_set[:, 1]))
    # plt.xlim(lower_bound, upper_bound)
    # plt.ylim(lower_bound, upper_bound)
    # plt.figure()
    # plt.xlim(lower_bound, upper_bound)
    # plt.ylim(lower_bound, upper_bound)
    # for partition in partition_array:
    #     if len(partition.nonzero()[0]):
    #         color = (random.random(), random.random(), random.random())
    #         plt.scatter(partition[:, 0], partition[:, 1], color=color)
    # plt.show()

    return partition_array, length

def get_scans_from_bag(bag, lidar_topic):
    scans = {}

    for topic, msg, t in tqdm(bag.read_messages(topics=[lidar_topic])):
        timestamp = t.secs + t.nsecs * 1e-9
        timestamp = '{:.5f}'.format(timestamp)
        scans[timestamp] = msg
    print ("Finished processing Bag file: {0} scans localizations".format(len(scans.keys())))
    return scans

def get_scans_and_localizations_from_bag(bag, lidar_topic, localization_topic, convert_to_clouds=True, scan_timestep=0, loc_timestep=0):
    localizations = {}
    scans = {}
    metadata = {}
    start_time = bag.get_start_time()
    last_scan_time = -10
    last_localization_time = -10
    print ("Loading scans & Localization from Bag file")
    print("Start time:", bag.get_start_time())
    print("End time:", bag.get_end_time())

    for topic, msg, t in tqdm(bag.read_messages(topics=[lidar_topic, localization_topic])):
        timestamp = t.secs + t.nsecs * 1e-9 - start_time
        if (topic == lidar_topic and timestamp - last_scan_time > scan_timestep):
            if not 'range_max' in metadata:
                metadata['range_max'] = msg.range_max
                metadata['range_min'] = msg.range_min
                metadata['angle_min'] = msg.angle_min
                metadata['angle_max'] = msg.angle_max
                metadata['angle_increment'] = msg.angle_increment

            last_scan_time = timestamp
            scans[timestamp] = scan_to_point_cloud(msg) if convert_to_clouds else msg
        elif (topic == localization_topic and timestamp - last_localization_time > loc_timestep):
            loc = []
            if msg._type == PoseStamped._type:
                msg = msg.pose
                angle = np.arctan2(msg.orientation.y, msg.orientation.x)
                loc = [msg.position.x, msg.position.y, angle]
            elif msg._type == Pose._type:
                angle = np.arctan2(msg.orientation.y, msg.orientation.x)
                loc = [msg.position.x, msg.position.y, angle]
            else:
                loc = [msg.x, msg.y, msg.angle]
                
            last_localization_time = timestamp
            localizations[timestamp] = np.asarray(loc)
    print ("Finished processing Bag file: {0} scans, {1} localizations".format(len(scans.keys()), len(localizations.keys())))
    return scans, localizations, metadata

class LCBagDataReader:
    def __init__(self, bag, lidar_topic, localization_topic, convert_to_clouds=False, scan_timestep=0, loc_timestep=0):
        scans, localizations, metadata = get_scans_and_localizations_from_bag(bag, lidar_topic, localization_topic, convert_to_clouds, scan_timestep, loc_timestep)
        self.scans = scans
        self.localizations = localizations
        self.metadata = metadata
        self.localization_timestamps = sorted(self.localizations.keys())
        self.scan_timestamps = sorted(self.scans.keys())
        if len(self.localization_timestamps) > 0:
            self.scan_timestamps = self.scan_timestamps[:len(self.localization_timestamps)]
            self.localization_timestamps = self.localization_timestamps[:len(self.scan_timestamps)]
            self.localizationTimeTree = spatial.cKDTree(np.asarray([[t] for t in self.localization_timestamps]))
            self.localizationTree = spatial.cKDTree(np.asarray([localizations[t][:2] for t in self.localization_timestamps]))
        self.scanTimeTree = spatial.cKDTree(np.asarray([[t] for t in self.scan_timestamps]))

    def get_scans(self):
        return self.scans

    def get_localizations(self):
        return self.localizations

    def get_scan_timestamps(self):
        return self.scan_timestamps

    def get_localization_timestamps(self):
        return self.localization_timestamps

    def get_localization_tree(self):
        return self.localizationTree
    
    def get_scan_time_tree(self):
        return self.scanTimeTree

    def get_localization_time_tree(self):
        return self.localizationTimeTree

    def get_closest_localization_by_time(self, time):
        _, loc_idx = self.localizationTimeTree.query([time])
        loc_timestamp = self.localization_timestamps[loc_idx]
        localization = self.localizations[loc_timestamp]

        return localization, loc_timestamp, loc_idx

    def get_closest_localization_by_location(self, location):
        _, loc_idx = self.localizationTree.query(location)
        loc_timestamp = self.localization_timestamps[loc_idx]
        localization = self.localizations[loc_timestamp]

        return localization, loc_timestamp, loc_idx

    def get_closest_scan_by_time(self, time):
        _, scan_idx = self.scanTimeTree.query([time])
        scan_timestamp = self.scan_timestamps[scan_idx]
        scan = self.scans[scan_timestamp]

        return scan, scan_timestamp, scan_idx

    def get_closest_scan_by_location(self, location):
        _, loc_timestamp, _ = self.get_closest_localization_by_location(location)
        return self.get_closest_scan_by_time(loc_timestamp)

    def get_nearby_locations(self, location, threshold):
        nearby_location_indices = self.localizationTree.query_ball_point(location, threshold)
        nearby_location_timestamps = [self.localization_timestamps[i] for i in nearby_location_indices]
        nearby_localizations = [self.localizations[t] for t in nearby_location_timestamps]

        return nearby_localizations, nearby_location_timestamps, nearby_location_indices

    def get_closest_localizations_by_time(self, times):
        if not isinstance(times, list):
            raise Error('times must be a list')

        _, loc_indices = self.localizationTimeTree.query(times)
        loc_timestamps = [self.localization_timestamps[loc_idx] for loc_idx in loc_indices]
        localizations = [self.localizations[loc_timestamp] for loc_timestamp in loc_timestamps]

        return localizations, loc_timestamps, loc_indices
    
    def get_closest_scans_by_time(self, times):
        if not isinstance(times, list):
            raise Error('times must be a list')

        _, scan_indices = self.scanTimeTree.query(times)
        scan_timestamps = [self.scan_timestamps[scan_idx] for scan_idx in scan_indices]
        scans = [self.scans[loc_timestamp] for loc_timestamp in scan_timestamps]

        return scans, scan_timestamps, scan_indices

def create_ros_pointcloud():
    msg = PointCloud2()
    msg.header.frame_id="map"
    msg.header.stamp = rostime.Time.now()
    msg.header.seq = 1
    msg.fields = [
        PointField('x', 0, 7, 1),
        PointField('y', 4, 7, 1),
        PointField('z', 8, 7, 1)
    ]
    msg.height = 1
    msg.width = 0
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 0
    msg.is_dense = True
    return msg

def publish_ros_pointcloud(publisher, msg, cloud):
    msg.data += cloud.tobytes()
    msg.width += len(cloud)
    publisher.publish(msg)
