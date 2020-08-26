import argparse
import os
import select
import sys
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from model import EmbeddingNet
from data_processing.dataset import LCDataset, LCTripletDataset, MergedDataset
import time
from tqdm import tqdm
import helpers
from helpers import print_output, initialize_logging

from config import training_config, execution_config

class TripletLoss(torch.nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = torch.norm(anchor - positive, p=2, dim=1)
        distance_negative = torch.norm(anchor - negative, p=2, dim=1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.sum()

parser = argparse.ArgumentParser()
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--embedding_model', type=str, default='', help='pretrained embedding model to start with')
parser.add_argument('--datasets', nargs='+', required=True, help="dataset paths")
parser.add_argument('--distance_cache', type=str, default=None, help='cached overlap info to start with')
parser.add_argument('--exhaustive', type=bool, default=False, help='Whether or not to check the exhaustive list of all triplets')

opt = parser.parse_args()
start_time = str(int(time.time()))
initialize_logging(start_time)

print_output(opt)

num_workers = execution_config['NUM_WORKERS']

opt.manualSeed = random.randint(1, 10000)  # fix seed
print_output("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

datasets = []
name = ''
for dataset in opt.datasets:
    ds = helpers.load_dataset(dataset, training_config['TRAIN_SET'], opt.distance_cache, opt.exhaustive)
    name += ds.dataset_info['name'] + '_' + ds.split + '_'
    datasets.append(ds)

merged_dataset = MergedDataset(datasets, name)

out_dir = opt.outf + '_' + merged_dataset.name

try:
    os.makedirs(out_dir)
except OSError:
    pass

embedder = helpers.create_embedder(opt.embedding_model)
embedder.train()

optimizer = optim.Adam(embedder.parameters(), lr=1e-3, weight_decay=1e-5)
lossFunc = TripletLoss(5)

pos_labels = torch.tensor(np.ones((execution_config['BATCH_SIZE'], 1)).astype(np.long)).squeeze(1).cuda()
neg_labels = torch.tensor(np.zeros((execution_config['BATCH_SIZE'], 1)).astype(np.long)).squeeze(1).cuda()
labels = torch.cat([pos_labels, neg_labels], dim=0)

print_output("Press 'return' at any time to finish training after the current epoch.")
for epoch in range(training_config['NUM_EPOCH']):
    total_loss = 0

    # We want to reload the triplets every 5 epochs to get new matches
    if opt.exhaustive:
        merged_dataset.load_all_triplets()
    else:
        merged_dataset.load_triplets()
    batch_count = len(merged_dataset) // execution_config['BATCH_SIZE']
    print_output("Loaded new training triplets: {0} batches of size {1}".format(batch_count, execution_config['BATCH_SIZE']))
    dataloader = torch.utils.data.DataLoader(
        merged_dataset,
        batch_size=execution_config['BATCH_SIZE'],
        shuffle=True,
        num_workers=num_workers,
        drop_last=True)

    total_similar_dist = 0.0
    total_distant_dist = 0.0

    for i, data in tqdm(enumerate(dataloader, 0)):
        ((clouds, locations, _), (similar_clouds, similar_locs, _), (distant_clouds, distant_locs, _)) = data
        clouds = clouds.transpose(2, 1).cuda()
        similar_clouds = similar_clouds.transpose(2, 1).cuda()
        distant_clouds = distant_clouds.transpose(2, 1).cuda()

        optimizer.zero_grad()
        embedder.zero_grad()

        anchor_embeddings, trans, theta = embedder(clouds)
        similar_embeddings, sim_trans, sim_theta = embedder(similar_clouds)
        distant_embeddings, dist_trans, dist_theta = embedder(distant_clouds)

        total_similar_dist += torch.sum(torch.norm(anchor_embeddings - similar_embeddings, dim=1)) 
        total_distant_dist += torch.sum(torch.norm(anchor_embeddings - distant_embeddings, dim=1)) 

        # Compute loss here
        loss = lossFunc.forward(anchor_embeddings, similar_embeddings, distant_embeddings)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_similar_dist = total_similar_dist / (batch_count * execution_config['BATCH_SIZE'])
    avg_distant_dist = total_distant_dist / (batch_count * execution_config['BATCH_SIZE'])

    print_output('[Epoch %d] Total loss: %f' % (epoch, total_loss))
    print_output('Average distance, Similar: {0} Distant: {1}'.format(avg_similar_dist, avg_distant_dist))
    helpers.save_model(embedder, out_dir, epoch)
    if (len(select.select([sys.stdin], [], [], 0)[0])):
        break

print_output("Completed training for {0} epochs".format(epoch + 1))