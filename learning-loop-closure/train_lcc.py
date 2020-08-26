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
import train_helpers
import time
from tqdm import tqdm
from train_helpers import print_output

start_time = str(int(time.time()))

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch_size', type=int, default=30, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=40, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='lcc', help='output folder')
parser.add_argument('--dataset', type=str, required=True, help="dataset path containing scans")
parser.add_argument('--labeled_timestamps', type=str, required=True, help="binary label for scans within the dataset")
parser.add_argument('--model', type=str, default='', help='pretrained model to start with')
parser.add_argument('--embedding_model', type=str, default='', help='pretrained embedding_model to start with')

opt = parser.parse_args()
train_helpers.initialize_logging(start_time)

print_output(opt)

num_workers = int(opt.workers)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print_output("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = train_helpers.load_lcc_dataset(opt.dataset, opt.labeled_timestamps)

out_dir = opt.outf + '_' + dataset.dataset_info['name']

try:
    os.makedirs(out_dir)
except OSError:
    pass

lcc_model = train_helpers.create_lcc(opt.embedding_model, opt.model)
lcc_model.train()

optimizer = optim.Adam(lcc_model.parameters(), lr=1e-3, weight_decay=1e-5)
lossFunc = torch.nn.MSELoss().cuda()

print_output("Press 'return' at any time to finish training after the current epoch.")
for epoch in range(opt.nepoch):
    total_loss = 0

    # We want to reload the triplets every 5 epochs to get new matches
    batch_count = len(dataset) // opt.batch_size
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True)

    # metrics = [0.0, 0.0, 0.0, 0.0] # True Positive, True Negative, False Positive, False Negative

    for i, data in tqdm(enumerate(dataloader, 0)):
        labels, conditions, scales, clouds, timestamps = data
        conditions = conditions.cuda()
        scales = scales.cuda()

        clouds = clouds.transpose(2, 1).cuda()
        lcc_model.zero_grad()
        optimizer.zero_grad()
        scores = lcc_model(clouds)

        true_scores = torch.stack((conditions, scales)).transpose(0, 1).float()
        loss = lossFunc(scores, true_scores)
        # print("sample scores", scores[0], true_scores[0])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    # acc = (metrics[0] + metrics[1]) / sum(metrics)
    # prec = (metrics[0]) / (metrics[0] + metrics[2])
    # rec = (metrics[0]) / (metrics[0] + metrics[3])
    print_output('[Epoch %d] Total loss: %f' % (epoch, total_loss))
    train_helpers.save_model(lcc_model, out_dir, epoch)
    if (len(select.select([sys.stdin], [], [], 0)[0])):
        break

print_output("Completed training for {0} epochs".format(epoch + 1))

train_helpers.close_logging()
