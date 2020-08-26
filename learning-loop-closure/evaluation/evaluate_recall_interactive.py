import rosbag
import rospy
import argparse
import numpy as np
import random
import torch
import statistics
import math
from geometry_msgs.msg import Pose
import roslib
roslib.load_manifest('cobot_msgs')
from cobot_msgs.msg import CobotLocalizationMsg

from learning.train_helpers import create_embedder
from helpers import scan_to_point_cloud, LCBagDataReader, embedding_for_scan
from scipy import spatial

parser = argparse.ArgumentParser(
    description='Find loop closure locations for some ROS bag')
parser.add_argument('--bag_file', type=str,
                    help='path to the bag file containing training data')
parser.add_argument('--lidar_topic', type=str,
                    help='name of topic containing lidar information')
parser.add_argument('--localization_topic', type=str,
                    help='name of topic containing localization information')
parser.add_argument('--model', type=str,
                    help='state dict of an already trained LC model to use')
parser.add_argument('--sample_size', type=int, default=1000,
                    help='number of embeddings to sample to find closest match')
args = parser.parse_args()
bag = rosbag.Bag(args.bag_file)

scans = {}
localizations = {}

start_time = bag.get_start_time()
data_reader = LCBagDataReader(bag, args.lidar_topic, args.localization_topic, SCAN_TIMESTEP, LOC_TIMESTEP)

model = create_embedder(args.model)
model.eval()

def handle_location_input(publisher):
    def handle_input(data):
        print("Recieved Input from localization gui", data)
        print("Possible angle", math.atan2(data.orientation.z, data.orientation.w))
        angle = math.atan2(data.orientation.z, data.orientation.w)
        location = [data.position.x, data.position.y, angle]
        print("Target pose: ", location)

        closest_location, closest_loc_timestamp, _ = data_reader.get_closest_localization_by_location(location[:2])
        print("Found closest recorded pose:", closest_location, "at", closest_loc_timestamp)
        closest_scan, closest_scan_timestamp, _ = data_reader.get_closest_scan_by_time(closest_loc_timestamp)
        print("Evaluating embeddings to find closest match...")
        with torch.no_grad():
            base_embedding = embedding_for_scan(model, closest_scan)
            random_timestamps = random.sample(data_reader.get_scan_timestamps(), args.sample_size)
            if closest_scan_timestamp in random_timestamps:
                random_timestamps.remove(closest_scan_timestamp)
            embeddings = [embedding_for_scan(model, data_reader.get_scans()[t]) for t in random_timestamps]
            print("Evaluating recall for random scans near these poses...")
            
            embedding_distances = []
            for emb in embeddings:
                distance = torch.norm(base_embedding[0] - emb[0], p=2).item()
                embedding_distances.append(distance)
            
            (vals, indices) = torch.topk(torch.tensor(embedding_distances), 5, largest=False)
            timestamps = [data_reader.get_scan_timestamps()[i] for i in indices]
            print("Timestamps of closest sampled scans in embedding space:")
            print(sorted(timestamps))
            print("Timestamps of actual closest locations:")
            _, location_indices = data_reader.localizationTree.query(closest_location[:2], k=5)
            print(sorted([data_reader.localization_timestamps[i] for i in location_indices]))

            locations, _, _ = data_reader.get_closest_localizations_by_time([[t] for t in timestamps])

            distances = [np.linalg.norm(np.asarray(closest_location[:2]) - np.asarray(l[:2])) for l in locations]
            print("Location Distance: Mean {0}, Median {1}, StDev: {2}".format(statistics.mean(distances), statistics.median(distances), statistics.stdev(distances)))

            msg = Pose()
            msg.position.x = locations[0][0]
            msg.position.y =locations[0][1]
            msg.orientation.z = math.cos(locations[0][2])
            msg.orientation.w = math.sin(locations[0][2])
            publisher.publish(msg)

    return handle_input

def location_listener():
    rospy.init_node('evaluate_recall', anonymous=True)
    pub = rospy.Publisher('Cobot/Localization', Pose, queue_size=5)
    rospy.Subscriber('localization_gui/nav_goal', Pose, handle_location_input(pub))
    rospy.spin()

print("Loaded bag file. Please run localization_gui to provide evaluation locations...")

location_listener()
