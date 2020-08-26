import argparse
import select
import random
import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import numpy as np
from ltf_model import SegNet
from ltf_dataset import LTFDataset
import time
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.join(os.getcwd(), '..'))
from helpers import print_output, initialize_logging, save_model

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch_size', type=int, default=64, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=40, help='number of epochs to train for')
parser.add_argument(
    '--bag_file', type=str, default='train', help='subset of the data to train on. One of [val, dev, train].')
parser.add_argument(
    '--filtered_topic', type=str, default='/filtered', help='topic to look for filtered scans')
parser.add_argument(
    '--base_topic', type=str, default='/Cobot/Laser', help='topic to look for base scans')
parser.add_argument('--outf', type=str, default='ltf_', help='output folder')

opt = parser.parse_args()
start_time = str(int(time.time()))
initialize_logging(start_time)

print_output(opt)

num_workers = int(opt.workers)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print_output("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


dataset = LTFDataset(opt.bag_file, opt.base_topic, opt.filtered_topic, 200)
out_dir = opt.outf + '_' + os.path.basename(opt.bag_file)

try:
    os.makedirs(out_dir)
except OSError:
    pass

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True)

ltf_model = SegNet(3, 2).cuda()

optimizer = optim.Adam(ltf_model.parameters(), lr=1e-3, weight_decay=1e-5)
lossFunc = torch.nn.CrossEntropyLoss(weight=torch.tensor(dataset.class_weights).cuda())

print_output("Press 'return' at any time to finish training after the current epoch.")
for epoch in range(opt.nepoch):
    total_loss = 0

    for i, data in tqdm(enumerate(dataloader, 0)):
        original, filtered = data

        generated = ltf_model(original.cuda())[0]
        
        # Compute loss here
        loss = lossFunc.forward(generated, filtered.cuda())

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print_output('[Epoch %d] Total loss: %f' % (epoch, total_loss))
    save_model(ltf_model, out_dir, epoch)
    if (len(select.select([sys.stdin], [], [], 0)[0])):
        break

print_output("Completed training for {0} epochs".format(epoch + 1))