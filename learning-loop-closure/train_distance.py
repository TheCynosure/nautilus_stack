import argparse
import os
import select
import sys
import random
import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import numpy as np
from model import EmbeddingNet
from data_processing.dataset import LCDataset, LCTripletDataset
import time
from tqdm import tqdm
import helpers
from helpers import print_output

class TripletLoss(torch.nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, distance_positive, distance_negative):
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.sum()

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch_size', type=int, default=2, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=40, help='number of epochs to train for')
parser.add_argument(
    '--train_set', type=str, default='train', help='subset of the data to train on. One of [val, dev, train].')
parser.add_argument(
    '--generate_embeddings', type=bool, default=False, help='if true, generate embeddings for test set in embeddings/*timestamp*')
parser.add_argument('--outf', type=str, default='cls_dist', help='output folder')
parser.add_argument('--embedding_model', type=str, default='', help='pretrained embedding model to start with')
parser.add_argument('--model', type=str, default='', help='pretrained distance model to start with')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--distance_cache', type=str, default=None, help='cached overlap info to start with')

opt = parser.parse_args()
start_time = str(int(time.time()))
helpers.initialize_logging(start_time)

print_output(opt)

num_workers = int(opt.workers)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print_output("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = helpers.load_dataset(opt.dataset, opt.train_set, opt.distance_cache)

out_dir = opt.outf + '_' + dataset.dataset_info['name'] + '_' + dataset.split

try:
    os.makedirs(out_dir)
except OSError:
    pass

distanceModel = helpers.create_distance_model(opt.embedding_model, opt.model)
distanceModel.train()

optimizer = optim.Adam(distanceModel.parameters(), lr=1e-3, weight_decay=1e-5)
lossFunc = TripletLoss(5)

pos_labels = torch.tensor(np.ones((opt.batch_size, 1)).astype(np.long)).squeeze(1).cuda()
neg_labels = torch.tensor(np.zeros((opt.batch_size, 1)).astype(np.long)).squeeze(1).cuda()
labels = torch.cat([pos_labels, neg_labels], dim=0)

print_output("Press 'return' at any time to finish training after the current epoch.")
for epoch in range(opt.nepoch):
    total_loss = 0

    # We want to reload the triplets every 5 epochs to get new matches
    dataset.load_triplets()
    batch_count = len(dataset) // opt.batch_size
    print_output("Loaded new training triplets: {0} batches of size {1}".format(batch_count, opt.batch_size))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True)

    total_similar_dist = 0.0
    total_distant_dist = 0.0

    for i, data in tqdm(enumerate(dataloader, 0)):
        ((clouds, locations, _), (similar_clouds, similar_locs, _), (distant_clouds, distant_locs, _)) = data
        clouds = clouds.transpose(2, 1)
        similar_clouds = similar_clouds.transpose(2, 1)
        distant_clouds = distant_clouds.transpose(2, 1)
        
        clouds = clouds.cuda()
        similar_clouds = similar_clouds.cuda()
        distant_clouds = distant_clouds.cuda()

        optimizer.zero_grad()
        distanceModel.zero_grad()

        similar_distances, _, _ = distanceModel(clouds, similar_clouds)
        distant_distances, _, _ = distanceModel(clouds, distant_clouds)
        
        total_similar_dist += torch.sum(similar_distances, dim=0)
        total_distant_dist += torch.sum(distant_distances, dim=0)

        # Compute loss here
        loss = lossFunc.forward(similar_distances, distant_distances)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_similar_dist = total_similar_dist / (batch_count * opt.batch_size)
    avg_distant_dist = total_distant_dist / (batch_count * opt.batch_size)

    print_output('[Epoch %d] Total loss: %f' % (epoch, total_loss))
    print_output('Average distance, Similar: {0} Distant: {1}'.format(avg_similar_dist, avg_distant_dist))
    helpers.save_model(distanceModel, out_dir, epoch)
    if (len(select.select([sys.stdin], [], [], 0)[0])):
        break

print_output("Completed training for {0} epochs".format(epoch + 1))