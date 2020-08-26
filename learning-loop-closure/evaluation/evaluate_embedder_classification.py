import argparse
import os
import select
import sys
import torch
import torch.utils.data
import numpy as np
import pickle
import time
import random
from tqdm import tqdm
sys.path.append(os.path.join(os.getcwd(), '..'))
import helpers
from helpers import initialize_logging, print_output
from config import execution_config, evaluation_config

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--model', type=str, default='', help='model to evaluate');
parser.add_argument('--distance_cache', type=str, default=None, help='cached overlap info to start with')
parser.add_argument('--publish_triplets', type=bool, default=False, help="if included, publish evaluated triplets, as well as classification result.")
parser.add_argument('--exhaustive', type=bool, default=False, help='Whether or not to check the exhaustive list of all triplets')
parser.add_argument('--no_vis', action='store_true', help='when provided, dont visualize the PR curve')
opt = parser.parse_args()
start_time = str(int(time.time()))
initialize_logging(start_time, 'evaluate_')
print_output(opt)

num_workers = int(execution_config['NUM_WORKERS'])

opt.manualSeed = random.randint(1, 10000)  # fix seed
print_output("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

embedder = helpers.create_embedder(opt.model)
embedder.eval()
dataset = helpers.load_dataset(opt.dataset, evaluation_config['EVALUATION_SET'], opt.distance_cache, opt.exhaustive, True)
batch_count = len(dataset) // execution_config['BATCH_SIZE']
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=execution_config['BATCH_SIZE'],
    shuffle=True,
    num_workers=num_workers,
    drop_last=True)

pos_labels = torch.tensor(np.ones((execution_config['BATCH_SIZE'], 1)).astype(np.long)).squeeze(1)
neg_labels = torch.tensor(np.zeros((execution_config['BATCH_SIZE'], 1)).astype(np.long)).squeeze(1)
labels = torch.cat([pos_labels, neg_labels], dim=0).cuda()

thresholds = np.linspace(evaluation_config['THRESHOLD_MIN'], evaluation_config['THRESHOLD_MAX'], (evaluation_config['THRESHOLD_MAX'] - evaluation_config['THRESHOLD_MIN']) * 2 + 1)

metrics = np.zeros((len(thresholds), 4)) # True Positive, True Negative, False Positive, False Negative

triplets = np.zeros((batch_count, execution_config['BATCH_SIZE'], 3, 2))

print("Evaluation over {0} batches of size {1}".format(batch_count, execution_config['BATCH_SIZE']))

for i, data in tqdm(enumerate(dataloader, 0)):
    ((clouds, locations, timestamp), (similar_clouds, similar_locs, similar_timestamp), (distant_clouds, distant_locs, distant_timestamp)) = data

    clouds = clouds.transpose(2, 1).cuda()
    similar_clouds = similar_clouds.transpose(2, 1).cuda()
    distant_clouds = distant_clouds.transpose(2, 1).cuda()
    
    for t in range(len(thresholds)):
        distances = helpers.get_distances_for_model(embedder, clouds, similar_clouds, distant_clouds)
        predictions = (distances < thresholds[t]).int()
        helpers.update_metrics(metrics[t], predictions, labels)

    if opt.publish_triplets:
        triplets[i, :, 0, 0] = timestamp
        triplets[i, :, 1, 0] = similar_timestamp
        triplets[i, :, 2, 0] = distant_timestamp

roc = np.zeros((len(thresholds), 4))
confusions = np.zeros((len(thresholds), 2, 2))
for i in range(len(thresholds)):
    threshold_metrics = metrics[i]
    roc[i][0] = (threshold_metrics[0] + threshold_metrics[1]) / sum(threshold_metrics)
    roc[i][1] = (threshold_metrics[0]) / (threshold_metrics[0] + threshold_metrics[2])
    roc[i][2] = (threshold_metrics[0]) / (threshold_metrics[0] + threshold_metrics[3])
    roc[i][3] = 2 * roc[i][1] * roc[i][2] / (roc[i][1] + roc[i][2])
    confusions[i] = [[threshold_metrics[0], threshold_metrics[2]], [threshold_metrics[3], threshold_metrics[1]]]
    print_output('(Acc: %f, Precision: %f, Recall: %f, F1: %f) for threshold %f' % (roc[i][0], roc[i][1], roc[i][2], roc[i][3], thresholds[i]))

import matplotlib as mpl
if opt.no_vis:
    mpl.use('Agg')
from matplotlib import pyplot as plt

plt.plot(roc[:, 2], roc[:, 1], color='r', label="Threshold")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()

if not opt.no_vis:
    plt.show()
else:
    plt.savefig('precision_recall_curve.png')

if opt.publish_triplets:
    name = os.path.basename(opt.dataset)
    print("Writing triplets_{0}.npy".format(name))
    np.save('triplets_{0}'.format(name), triplets)
    np.save('confusions_{0}'.format(name), confusions)
