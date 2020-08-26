import rosbag
from tqdm import tqdm
from ltf_helpers import discretize_point_cloud
import numpy as np
import sys
import os

import sys
import os
sys.path.append(os.path.join(os.getcwd(), '..'))
from data_processing.data_processing_helpers import scan_to_point_cloud

class LTFDataset:
  def __init__(self, bag_file, base_topic, filtered_topic, dimensions=200, lidar_range=30):
    self.bag_file = bag_file
    self.base_topic = base_topic
    self.filtered_topic = filtered_topic
    self.dimensions = dimensions
    self.lidar_range = lidar_range
    self.load_data()

  def load_data(self):
    bag = rosbag.Bag(self.bag_file)
    timestamps = set()
    self.data = {}
    start_time = bag.get_start_time()
    nonzero_ct = 0
    for topic, msg, t in tqdm(bag.read_messages(topics=[self.base_topic, self.filtered_topic])):
        if len(timestamps) > 500:
          break
        timestamp = round(t.secs + t.nsecs * 1e-9 - start_time, 5)
        timestamps.add(timestamp)
        if timestamp not in self.data:
          self.data[timestamp] = [np.zeros((3, self.dimensions, self.dimensions)).astype(np.float32), np.zeros((self.dimensions, self.dimensions)).astype(np.int)]
        if (topic == self.base_topic):
          self.data[timestamp][0] = np.rollaxis(discretize_point_cloud(scan_to_point_cloud(msg), self.lidar_range, self.dimensions, True), 2, 0)
          nonzero_ct += len(self.data[timestamp][0].nonzero())
        elif (topic == self.filtered_topic):
          self.data[timestamp][1] = discretize_point_cloud(scan_to_point_cloud(msg), self.lidar_range, self.dimensions)
          nonzero_ct += len(self.data[timestamp][0].nonzero())
    
    total_ct = self.dimensions**2 * len(timestamps)
    nonzero_wt = 1.0 / (float(nonzero_ct) / total_ct)
    zero_wt = 1.0 / (float(total_ct - nonzero_ct) /total_ct)
    self.class_weights = (zero_wt, nonzero_wt)
    print("class weights", self.class_weights)
    self.class_weights = (1.0, 10.0)
    self.timestamps = sorted(list(timestamps))

  def __getitem__(self, index):
    return self.data[self.timestamps[index]][0], self.data[self.timestamps[index]][1]

  def __len__(self,):
    return len(self.timestamps)