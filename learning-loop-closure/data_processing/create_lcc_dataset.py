import rospy
import argparse
import numpy as np
import os
import json
from learning.dataset import LCDataset
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='Find loop closure locations for some ROS bag')
parser.add_argument('--source_dataset', type=str, help='path to first dataset')
parser.add_argument('--split', type=str, help='split within first dataset')
parser.add_argument('--lcc_evaluator', type=str, help='path to an executable for evaluating if a location is good for LC')
parser.add_argument('--dataset_name', type=str, help='dataset name')
args = parser.parse_args()

out_dir = os.path.join('data', args.dataset_name)

try:
    os.makedirs(out_dir)
except OSError:
    pass

dataset = LCDataset(args.source_dataset, args.split)

labels = np.zeros((len(dataset), 1))

for i in tqdm(range(len(dataset))):
    cloud, location, timestamp = dataset[i]

    label = 1 # TODO Actually evaluate

    labels[i] = label

# Write dataset info
info_file = os.path.join(out_dir, 'lcc_info.json')
ds_info = json.dumps({'source_dataset_name': dataset.dataset_info['name'], 'split':args.split, 'name': args.dataset_name})
with open(info_file, 'w') as f:
    f.write(ds_info)

np.save(os.path.join(out_dir, 'labels.npy'), labels)