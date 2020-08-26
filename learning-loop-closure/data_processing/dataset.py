import torch.utils.data as data
from scipy.spatial import cKDTree
import multiprocessing as mp
import os
import os.path
import glob
import torch
import numpy as np
import sys
import random
import math
from tqdm import tqdm
import json
import pickle
from data_processing_helpers import compute_overlap, partition_point_cloud, LCBagDataReader, get_scans_from_bag
from ltf_segmentation.ltf_helpers import discretize_point_cloud
import rosbag

sys.path.append(os.path.join(os.getcwd(), '..'))
from config import data_config, data_generation_config
class LCDataset(data.Dataset):
    def __init__(self,
                 root,
                 split=None):
        self.root = root
        self.split = split
        self.timestamp_tree = None

        info_file = os.path.join(self.root, 'dataset_info.json')

        self.dataset_info = json.load(open(info_file, 'r'))
        if self.split:
            self.file_list = self.dataset_info[self.split + '_data']
        else:
            self.file_list = [f[:f.rfind('.npy')] for f in glob.glob(os.path.join(self.root, 'point_*.npy'))]

        self.timestamps = [[float(f[f.find('point_') + len('point_'):])] for f in self.file_list]

    def __getitem__(self, index):
        fname = self.file_list[index]
        timestamp = fname[fname.rfind('_')+1:]

        return self.get_by_timestamp(timestamp)

    def get_by_timestamp(self, timestamp, include_angle=False):
        location_file = os.path.join(
            self.root, 'location_{0}.npy'.format(timestamp))
        location = np.load(location_file).astype(np.float32)
        cloud_file = os.path.join(self.root, 'point_{0}.npy'.format(timestamp))
        cloud = np.load(cloud_file).astype(np.float32)
        if not include_angle:
            location = location[:2]
        return cloud, location, timestamp

    def get_by_nearest_timestamp(self, target_timestamp, include_angle=False):
        if not self.timestamp_tree:
            self.timestamp_tree = cKDTree(self.timestamps)
        
        _, timestamp_idx = self.timestamp_tree.query([target_timestamp])
        timestamp =  self.timestamps[timestamp_idx][0]
        return self.get_by_timestamp(timestamp, include_angle)

    def __len__(self):
        return len(self.file_list)

class MergedDataset(data.Dataset):
    def __init__(self, datasets, name):
        self.datasets = datasets
        self.name = name

    def load_data(self):
        for dataset in self.datasets:
            dataset.load_data()
    
    def load_triplets(self):
        for dataset in self.datasets:
            dataset.load_triplets()
    
    def load_all_triplets(self):
        for dataset in self.datasets:
            dataset.load_all_triplets()

    def set_distance_threshold(self, threshold):
        for dataset in self.datasets:
            dataset.set_distance_threshold(threshold)

    def __getitem__(self, index):
        for dataset in self.datasets:
            if index >= len(dataset):
                index -= len(dataset)
                continue
            return dataset[index]

    def __len__(self):
        return np.sum([len(dataset) for dataset in self.datasets])

class LCTripletDataset(data.Dataset):
    def __init__(self,
                 root,
                 split='train',
                 evaluation=False):
        self.root = root
        if not evaluation:
            self.augmentation_prob = data_config['AUGMENTATION_PROBABILITY']
        else:
            self.augmentation_prob = 0
            
        self.M = data_config['MATCH_REPEAT_FACTOR']
        self.split = split

        info_file = os.path.join(self.root, 'dataset_info.json')
        #from IPython import embed; embed()
        self.dataset_info = json.load(open(info_file, 'r'))
        # self.overlap_radius = self.dataset_info['scanMetadata']['range_max'] * 0.4 if 'scanMetadata' in self.dataset_info else 4
        self.data_loaded = False
        self.triplets_loaded = False
        self.computed_new_distances = False
        self.data = []
        self.overlaps = {}
        self.triplets = []

    def load_data(self):
        # Use dataset_info to load data files
        if len(self.split) > 0:
            filelist = self.dataset_info[self.split + '_data']
        else:
            filelist = [os.path.basename(f[:f.rfind('.npy')]) for f in glob.glob(os.path.join(self.root, 'point_*.npy'))]
            
        for fname in tqdm(filelist):
            timestamp = fname[fname.rfind('_')+1:]
            location_file = os.path.join(
                self.root, 'location_{0}.npy'.format(timestamp))
            location = np.load(location_file).astype(np.float32)
            cloud = np.load(os.path.join(self.root, fname + '.npy')).astype(np.float32)
            self.data.append((cloud, location, timestamp))
        self.location_tree = cKDTree(np.asarray([d[1][:2] for d in self.data]))
        self.data = np.array(self.data)
        self.data_loaded = True

    def _create_triplet(self, cloud, location, timestamp, similar_cloud, similar_loc, similar_timestamp):
        idx = random.randint(0, len(self.data) - 1)
        # We don't want anything that's even remotely nearby to count as "distant"
        dist_neighbors = self.location_tree.query_ball_point(location[:2], data_config['FAR_DISTANCE_THRESHOLD'])
        while idx in dist_neighbors:
            idx = random.randint(0, len(self.data) - 1)
        distant_cloud, distant_loc, distant_timestamp = self.data[idx]
        return (
            (cloud, location, timestamp),
            (similar_cloud, similar_loc, similar_timestamp),
            (distant_cloud, distant_loc, distant_timestamp)
        )

    def _create_all_triplets(self, cloud, location, timestamp, similar_cloud, similar_loc, similar_timestamp):
        idx = random.randint(0, len(self.data) - 1)
        # We don't want anything that's even remotely nearby to count as "distant"
        dist_neighbors = self.location_tree.query_ball_point(location[:2], data_config['FAR_DISTANCE_THRESHOLD'])
        non_neighbors = np.setdiff1d(range(len(self.data)), dist_neighbors)
        return [(
            (cloud, location, timestamp),
            (similar_cloud, similar_loc, similar_timestamp),
            (self.data[idx][0], self.data[idx][1], self.data[idx][2])
        ) for idx in non_neighbors]

    # We want to end up with self.M augmented neighbors for this cloud, if it was chosen
    def _generate_augmented_triplets(self, cloud, location, timestamp):
        augmented_neighbors = self.generate_augmented_neighbors(cloud)
        triplets = []
        for similar in augmented_neighbors:
            triplets.append(self._create_triplet(cloud, location, timestamp, similar, location, timestamp))

        return triplets

    def _generate_triplets(self, cloud, location, timestamp):
        neighbors = self.location_tree.query_ball_point(location[:2], data_config['CLOSE_DISTANCE_THRESHOLD'])
        filtered_neighbors = self.filter_scan_matches(timestamp, location, neighbors[1:])
        if len(filtered_neighbors) > 0:
            triplets = []
            indices = np.random.choice(filtered_neighbors, self.M)
            for idx in indices:
                s = self.data[idx]
                triplets.append(self._create_triplet(cloud, location, timestamp, s[0], s[1], s[2]))
            return triplets
        else:
            return None

    def _generate_all_triplets(self, cloud, location, timestamp):
        neighbors = self.location_tree.query_ball_point(location[:2], data_config['CLOSE_DISTANCE_THRESHOLD'])
        filtered_neighbors = self.filter_scan_matches(timestamp, location, neighbors[1:])

        if len(filtered_neighbors) > 0:
            triplets = []
            similar = [self.data[idx] for idx in filtered_neighbors]
            for s in similar:
                triplets.extend(self._create_all_triplets(cloud, location, timestamp, s[0], s[1], s[2]))
            return triplets
        else:
            return []

    def load_triplets(self):
        if not self.data_loaded:
            raise Exception('Call load_data before attempting to load triplets')
        del self.triplets[:]

        for cloud, location, timestamp in tqdm(self.data):
            triplets = self._generate_triplets(cloud, location, timestamp)
            if triplets:
                self.triplets.extend(triplets)

        augment_indices = np.random.choice(range(len(self.data)), int(self.augmentation_prob * len(self.data)))
        for augment_idx in tqdm(augment_indices):
            cloud, location, timestamp = self.data[augment_idx]
            self.triplets.extend(self._generate_augmented_triplets(cloud, location, timestamp))
        
        self.triplets_loaded = True

    # This loads the exhaustive set of triplets; should only be used on relatively small datasets
    # Also, does not include data augmentation
    def load_all_triplets(self):
        if not self.data_loaded:
            raise Exception('Call load_data before attempting to load all triplets')
        del self.triplets[:]

        for cloud, location, timestamp in tqdm(self.data):
            triplets = self._generate_all_triplets(cloud, location, timestamp)
            if len(triplets) > 0:
                self.triplets.extend(triplets)
        self.triplets_loaded = True

    def generate_augmented_neighbors(self, cloud):
        neighbors = []
        def _rotation_augmented():
            # random perturbations, because why not
            theta = np.random.uniform(-np.pi / 3, np.pi / 3)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            augmented = np.zeros(cloud.shape).astype(np.float32)
            # random rotation
            augmented[:, :] = cloud[:, :].dot(rotation_matrix)
            return augmented
        
        #We will roll instead of randomly permuting, so sequential local features are maintained
        def _roll_augmented():
            return np.roll(cloud, np.random.randint(0, cloud.shape[0] / 10), 0)
        
        def _missing_augmented():
            indices = np.random.choice(range(len(cloud)), int(len(cloud) * 0.95))
            return np.pad(cloud[indices], ((0, len(cloud) - len(indices)), (0, 0)), 'constant')

        def _translation_augmented():
            shift = np.random.rand(2) * 5
            augmented = np.zeros(cloud.shape).astype(np.float32)
            augmented[:, :] = cloud[:, :] + shift
            return augmented
        
        while len(neighbors) < self.M:
            neighbors.append(_rotation_augmented())
            neighbors.append(_roll_augmented())
            neighbors.append(_missing_augmented())
            neighbors.append(_translation_augmented())

        return neighbors

    def filter_scan_matches(self, timestamp, location, neighbors):
        filtered = list(filter(self.time_filter(timestamp), neighbors))
        filtered = list(filter(self.check_overlap(location, timestamp), filtered))
        return np.array(filtered)

    def time_filter(self, timestamp):
        def filter_checker(alt_idx):
            alt_timestamp = self.data[alt_idx][2]
            return abs(float(timestamp) - float(alt_timestamp)) > data_config['TIME_IGNORE_THRESHOLD']
        
        return filter_checker

    def check_overlap(self, location, timestamp):
        def overlap_checker(alt_idx):
            alt_loc = self.data[alt_idx][1]
            alt_timestamp = self.data[alt_idx][2]
            key = (timestamp, alt_timestamp) if timestamp < alt_timestamp else (alt_timestamp, timestamp)
            if key in self.overlaps:
                return self.overlaps[key] > data_config['OVERLAP_SIMILARITY_THRESHOLD']
            else:
                overlap = compute_overlap(location, alt_loc)
                self.computed_new_distances = True
                self.overlaps[key] = overlap
                return self.overlaps[key] > data_config['OVERLAP_SIMILARITY_THRESHOLD']

        return overlap_checker

    def _get_distance_cache(self):
        return self.dataset_info['name'] + '_' + self.split + '_distances.pkl'

    def load_distances(self, distance_cache):
        if not distance_cache:
            distance_cache = self._get_distance_cache()
        if os.path.exists(distance_cache):
            print("Loading overlap information from cache...")
            with open(distance_cache, 'rb') as f:
                self.overlaps = pickle.load(f)

    def cache_distances(self):
        print("Saving overlap information to cache...")
        with open(self._get_distance_cache(), 'wb') as f:
            pickle.dump(self.overlaps, f)

    def __getitem__(self, index):
        if not self.triplets_loaded:
            raise Exception('Call load_triplets before attempting to access elements')
        
        return self.triplets[index]

    def __len__(self):
        return len(self.triplets)

class LCTripletDiscretizedDataset(LCTripletDataset):
    def __init__(self,
                 root,
                 split='train'):
        super(LCTripletDiscretizedDataset, self).__init__(root, split)
    
    def load_data(self):
        # Use dataset_info to load data files
        filelist = self.dataset_info[self.split + '_data']
            
        for fname in tqdm(filelist):
            timestamp = fname[fname.rfind('_')+1:]
            location_file = os.path.join(
                self.root, 'location_{0}.npy'.format(timestamp))
            location = np.load(location_file).astype(np.float32)
            cloud = np.load(os.path.join(self.root, fname + '.npy')).astype(np.float32)
            cloud = discretize_point_cloud(cloud, self.dataset_info['scanMetadata']['range_max'], 200)
            self.data.append((cloud, location, timestamp))
        self.location_tree = cKDTree(np.asarray([d[1][:2] for d in self.data]))
        self.data = np.array(self.data)
        self.data_loaded = True

class LCTripletStructuredDataset(LCTripletDataset):
    def __init__(self,
                 root,
                 split='train',
                 threshold=0.5):
        super(LCTripletStructuredDataset, self).__init__(root, split)
        self.threshold = threshold        

    def __getitem__(self, index):
        if not self.triplets_loaded:
            raise Exception('Call load_triplets before attempting to access elements')
        
        triplet = self.triplets[index]
        return (
            (partition_point_cloud(triplet[0][0], self.threshold), triplet[0][1], triplet[0][2]),
            (partition_point_cloud(triplet[1][0], self.threshold), triplet[1][1], triplet[1][2]),
            (partition_point_cloud(triplet[2][0], self.threshold), triplet[2][1], triplet[2][2])
        )

class LCStructuredDataset(LCDataset):
    def __init__(self,
                 root,
                 split=None,
                 threshold=0.5):
        super(LCStructuredDataset, self).__init__(root, split)
        self.threshold = threshold

    def get_by_timestamp(self, timestamp, include_angle=False):
        cloud, location, timestamp = super(LCStructuredDataset, self).get_by_timestamp(timestamp, include_angle)

        return (partition_point_cloud(cloud, self.threshold), location, timestamp)

class LCCDataset(LCDataset):
    def __init__(self,
                 root,
                 timestamps,
                 split='dev'):
        self.labeled_timestamps = np.load(timestamps)
        super(LCCDataset, self).__init__(root, split)

    def __getitem__(self, index):
        timestamp = self.labeled_timestamps[index][0]
        label = int(self.labeled_timestamps[index][1])
        condition = float(self.labeled_timestamps[index][2])
        scale = float(self.labeled_timestamps[index][3])
        cloud, _, timestamp = self.get_by_nearest_timestamp(timestamp)

        return label, condition, scale, cloud, timestamp

    def __len__(self):
        return len(self.labeled_timestamps)
    
# Local Uncertainty
class LUDataset(data.Dataset):
    def __init__(self, bag_file, stats_file):
        self.bag_file = bag_file
        self.stats_file = stats_file
        self.bag = rosbag.Bag(self.bag_file)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def load_data(self):
        self.data_reader = LCBagDataReader(self.bag, data_generation_config['LIDAR_TOPIC'] ,None, False, data_generation_config['TIME_SPACING'], data_generation_config['TIME_SPACING'])

        self.data = []
        with open(self.stats_file, 'r') as f:
            for line in f.readlines():
                timestamp, stats = line.strip().split(': ')
                first, second = stats.strip().split(', ')
                timestamp = float(timestamp) - self.bag.get_start_time() # correct for the fact that our stats dont offset by bag time
                scan = self.data_reader.get_closest_scan_by_time(timestamp)[0].ranges
                self.data.append((np.array(scan).astype(np.float32), np.array([first, second]).astype(np.float32)))

# Loop closure using laser scan
class LCLaserDataset(data.Dataset):
    def __init__(self, config, use_overlap=False):
        super(LCLaserDataset, self).__init__()
        self.bag_file = config.bag_file
        self.bag = rosbag.Bag(config.bag_file)
        self.name = config.name
        self.data_loaded = False
        self.augmentation_prob = config.augmentation_probability
        self.edge_trimming = config.edge_trimming
        self.use_overlap = use_overlap
        self.lidar_max_range = config.lidar_max_range
        self.distance_threshold = data_config['CLOSE_DISTANCE_THRESHOLD']
        self.data_reader = LCBagDataReader(self.bag,  data_generation_config['LIDAR_TOPIC'], data_generation_config['LOCALIZATION_TOPIC'], False, data_generation_config['TIME_SPACING'], data_generation_config['TIME_SPACING'])
        if self.use_overlap:
            self.overlaps = {}

    def set_distance_threshold(self, close_dist):
        self.distance_threshold = close_dist

    def get_scan_by_idx(self, index):
        full_scan = np.asarray(self.data_reader.scans[self.data_reader.scan_timestamps[index]].ranges).astype(np.float32)
        full_scan[:self.edge_trimming] = self.lidar_max_range
        full_scan[-self.edge_trimming:] = self.lidar_max_range
        return full_scan

    def get_location_by_idx(self, index):
        return np.asarray(self.data_reader.localizations[self.data_reader.localization_timestamps[index]]).astype(np.float32)

    def __getitem__(self, index):
        if not self.data_loaded:
            raise Exception("Must load data before accessing")

        should_augment = False
        if (index >= len(self.data)):
            index -= len(self.data)
            should_augment = True

        idx, alt_idx, label = self.data[index]

        scan = self.get_scan_by_idx(idx)
        location = self.get_location_by_idx(idx)
        timestamp = self.data_reader.localization_timestamps[idx]

        alt_location = self.get_location_by_idx(alt_idx)
        alt_scan = self.get_scan_by_idx(alt_idx)
        alt_timestamp = self.data_reader.scan_timestamps[alt_idx]

        if should_augment:
            alt_scan = np.random.normal(0, scale=0.01, size=alt_scan.shape).astype(np.float32) + alt_scan

        return ((scan, location, timestamp), (alt_scan, alt_location, alt_timestamp), label)

    def filter_scan_matches(self, timestamp, location, neighbors):
        if self.use_overlap:
            filtered = list(filter(self.time_filter(timestamp), neighbors))
            filtered = list(filter(self.check_overlap(location, timestamp), filtered))
        else:
            filtered = list(filter(self.transform_filter(location), neighbors))
        return np.array(filtered)

    def time_filter(self, timestamp):
        def filter_checker(alt_idx):
            alt_timestamp = self.data_reader.localization_timestamps[alt_idx]
            return abs(float(timestamp) - float(alt_timestamp)) > data_config['TIME_IGNORE_THRESHOLD']
        
        return filter_checker

    def check_overlap(self, location, timestamp):
        def overlap_checker(alt_idx):
            alt_loc = self.get_location_by_idx(alt_idx)
            alt_timestamp = self.data_reader.localization_timestamps[alt_idx]
            key = (timestamp, alt_timestamp) if timestamp < alt_timestamp else (alt_timestamp, timestamp)
            if key in self.overlaps:
                return self.overlaps[key] > data_config['OVERLAP_SIMILARITY_THRESHOLD']
            else:
                overlap = compute_overlap(location, alt_loc)
                self.computed_new_distances = True
                self.overlaps[key] = overlap
                return self.overlaps[key] > data_config['OVERLAP_SIMILARITY_THRESHOLD']

        return overlap_checker

    def transform_filter(self, location):
        def overlap_checker(alt_idx):
            alt_loc = self.get_location_by_idx(alt_idx)
            return abs(alt_loc[0] - location[0]) < self.distance_threshold and abs(alt_loc[1] - location[1]) < self.distance_threshold and abs(alt_loc[2] - location[2]) < 1.0 

        return overlap_checker
    
    def _get_distance_cache(self):
        return os.path.basename(self.bag_file + '.distances.pkl')

    def load_distances(self, distance_cache=None):
        if not distance_cache:
            distance_cache = self._get_distance_cache()
        if os.path.exists(distance_cache):
            print("Loading overlap information from cache...")
            with open(distance_cache, 'rb') as f:
                self.overlaps = pickle.load(f)

    def cache_distances(self):
        print("Saving overlap information to cache...")
        with open(self._get_distance_cache(), 'wb') as f:
            pickle.dump(self.overlaps, f)

    def load_data(self):
        loc_count = len(self.data_reader.localization_timestamps)
        self.data = []
        for index in tqdm(range(loc_count)):
            scan = self.get_scan_by_idx(index)
            location = self.get_location_by_idx(index)
            timestamp = self.data_reader.localization_timestamps[index]
            
            neighbors = self.data_reader.get_localization_tree().query_ball_point(location[:2], data_config['FAR_DISTANCE_THRESHOLD'])
            non_neighbors = np.setdiff1d(range(loc_count), neighbors)
                        
            filtered_neighbors = self.filter_scan_matches(timestamp, location, neighbors[1:])
            
            for sim_idx in filtered_neighbors:
                self.data.append(np.array([index, sim_idx, 1]).astype(np.int))

                dist_index = np.random.choice(non_neighbors, 1)[0]
                self.data.append((index, dist_index, 0))

        self.data_loaded= True

    
    def __len__(self):
        return int(len(self.data) * (1 + self.augmentation_prob))