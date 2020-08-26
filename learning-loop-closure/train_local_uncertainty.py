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
import time
from tqdm import tqdm
import helpers
from helpers import print_output, initialize_logging

from config import Configuration, data_config

config = Configuration(True, True)

config.add_argument('--stats_file', type=str, help='path to file containing ground-truth uncertainty stats')

config = config.parse()

start_time = str(int(time.time()))
initialize_logging(start_time)

print_output(config)

num_workers = config.num_workers

config.manualSeed = random.randint(1, 10000)  # fix seed
print_output("Random Seed: ", config.manualSeed)
random.seed(config.manualSeed)
torch.manual_seed(config.manualSeed)

if config.bag_file:
    dataset = helpers.load_uncertainty_dataset(config.bag_file, config.stats_file)
elif config.bag_files:
    raise Exception("not implemented yet")
    # dataset = helpers.load_merged_laser_dataset(config.bag_files, config.name, config.augmentation_probability)
else:
    raise Exception("Must provide bag input")
out_dir = 'local_uncertainty'

try:
    os.makedirs(out_dir)
except OSError:
    pass

scan_conv, scan_uncertainty = helpers.create_lu_networks(config.model_dir, config.model_epoch)
scan_conv.train()
scan_uncertainty.train()

conv_optimizer = optim.Adam(scan_conv.parameters(), lr=1e-3, weight_decay=1e-6)
uncertainty_optimizer = optim.Adam(scan_uncertainty.parameters(), lr=1e-3, weight_decay=1e-6)
uncertaintyLossFunc = torch.nn.MSELoss()

print_output("Press 'return' at any time to finish training after the current epoch.")
for epoch in range(config.num_epoch):
    total_loss = 0

    if config.curriculum:
        # Train on smaller perturbations first, the last 10 epochs including all of them
        dataset.set_distance_threshold(min(float(epoch) / (config.num_epoch - 10) + 0.1, 1.0) * data_config['CLOSE_DISTANCE_THRESHOLD'])

    dataset.load_data()
    batch_count = len(dataset) // config.batch_size
    print_output("Loaded new training data: {0} batches of size {1}".format(batch_count, config.batch_size))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True)

    for i, data in tqdm(enumerate(dataloader, 0)):
        (scans, values) = data
        scans = scans.unsqueeze(1).cuda()
        values = values.cuda()
        
        conv_optimizer.zero_grad()
        uncertainty_optimizer.zero_grad()
        scan_conv.zero_grad()
        scan_uncertainty.zero_grad()

        conv = scan_conv(scans)

        pred_values = scan_uncertainty(conv)

        loss = uncertaintyLossFunc.forward(pred_values, values)

        loss.backward()

        if not config.lock_conv:
            conv_optimizer.step()
        
        uncertainty_optimizer.step()
        
        total_loss += loss.item()

    print_output('[Epoch %d] Total loss: %f' % (epoch, total_loss))

    if not config.lock_conv:
        helpers.save_model(scan_conv, out_dir, epoch, 'conv')

    helpers.save_model(scan_uncertainty, out_dir, epoch, 'uncertainty')
    if (len(select.select([sys.stdin], [], [], 0)[0])):
        break

print_output("Completed training for {0} epochs".format(epoch + 1))