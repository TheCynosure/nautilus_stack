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
from data_processing.dataset import LCLaserDataset
import time
from tqdm import tqdm
import helpers
from helpers import print_output, initialize_logging

from config import Configuration, data_config

config = Configuration(True, True).parse()

if not (config.train_match or config.train_transform):
    raise Exception('You must train for either matching or transformation recovery')

start_time = str(int(time.time()))
initialize_logging(start_time)

print_output(config)

num_workers = config.num_workers

config.manualSeed = random.randint(1, 10000)  # fix seed
print_output("Random Seed: ", config.manualSeed)
random.seed(config.manualSeed)
torch.manual_seed(config.manualSeed)


if config.bag_file:
    dataset = helpers.load_laser_dataset(config)
elif config.bag_files:
    dataset = helpers.load_merged_laser_dataset(config)
else:
    raise Exception("Must provide bag input")
out_dir = config.outf + '_' + dataset.name

try:
    os.makedirs(out_dir)
except OSError:
    pass

scan_conv, scan_match, scan_transform = helpers.create_laser_networks(config.model_dir, config.model_epoch)
scan_conv.train()
scan_match.train()
scan_transform.train()

conv_optimizer = optim.Adam(scan_conv.parameters(), lr=1e-3, weight_decay=1e-6)
match_optimizer = optim.Adam(scan_match.parameters(), lr=1e-3, weight_decay=1e-6)
transform_optimizer = optim.Adam(scan_transform.parameters(), lr=1e-3, weight_decay=1e-6)
matchLossFunc = torch.nn.CrossEntropyLoss(weight=torch.tensor([config.dist_weight_ratio, 1.0]).cuda())
transLossFunc = torch.nn.MSELoss()

print_output("Press 'return' at any time to finish training after the current epoch.")
for epoch in range(config.num_epoch):
    total_loss = 0

    if config.curriculum:
        # Train on smaller perturbations first, the last 10 epochs including all of them
        dataset.set_distance_threshold(min(float(epoch) / (config.num_epoch - 10) + 0.1, 1.0) * data_config['CLOSE_DISTANCE_THRESHOLD'])

    dataset.load_data()

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    batch_count = len(train_set) // config.batch_size
    val_batch_count = len(val_set) // config.batch_size
    print_output("Loaded new training data: {0} batches of size {1}".format(batch_count, config.batch_size))
    print_output("Loaded new validation data: {0} batches of size {1}".format(val_batch_count, config.batch_size))

    dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True)

    val_dataloader = torch.utils.data.DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True)

    for i, data in tqdm(enumerate(dataloader, 0)):
        ((clouds, locations, _), (alt_clouds, alt_locs, _), labels) = data
        clouds = clouds.cuda()
        alt_clouds = alt_clouds.cuda()
        labels = labels.cuda()

        conv_optimizer.zero_grad()
        match_optimizer.zero_grad()
        transform_optimizer.zero_grad()
        scan_conv.zero_grad()
        scan_match.zero_grad()
        scan_transform.zero_grad()

        conv = scan_conv(clouds, alt_clouds)

        # import pdb; pdb.set_trace()
        #Compute match prediction
        scores = scan_match(conv)
        predictions = torch.argmax(F.softmax(scores, dim=1), dim=1)

        if config.train_match:
            loss = matchLossFunc.forward(scores, labels)

        if config.train_transform:
            # Compute transforms, but only for things that *should* match
            match_indices = labels.nonzero()
            filtered = conv[match_indices]
            transforms = scan_transform(filtered)
            true_transforms = (locations - alt_locs)[match_indices]
            # Clamp angle between -pi and pi
            true_transforms[:, :, 2] = torch.fmod(2 * np.pi + true_transforms[:, :, 2], 2 * np.pi) - np.pi
            loss = transLossFunc.forward(transforms, true_transforms.squeeze(1).cuda())

        loss.backward()
        if not config.lock_conv:
            conv_optimizer.step()
        
        if config.train_match:
            match_optimizer.step()

        if config.train_transform:
            transform_optimizer.step()
        
        total_loss += loss.item()

    print_output('[Epoch %d] Total loss: %f' % (epoch, total_loss))

    if config.train_match:
        with torch.no_grad():
            metrics = np.zeros(4)
            for i, data in tqdm(enumerate(val_dataloader, 0)):
                ((clouds, locations, _), (alt_clouds, alt_locs, _), labels) = data
                clouds = clouds.cuda()
                alt_clouds = alt_clouds.cuda()
                labels = labels.cuda()
                conv = scan_conv(clouds, alt_clouds)

                #Compute match prediction
                scores = scan_match(conv)
                predictions = torch.argmax(F.softmax(scores, dim=1), dim=1)
                metrics[0] += torch.sum((predictions + labels == 2)) # both prediction and lable are 1
                metrics[1] += torch.sum((predictions - labels == 1)) # prediction is 1 but label is 0
                metrics[2] += torch.sum((predictions + labels == 0)) # both prediction and label are 0
                metrics[3] += torch.sum((predictions - labels == -1)) # prediction is 0, label is 1
        
            acc = (metrics[0] + metrics[2]) / sum(metrics)
            prec = (metrics[0]) / (metrics[0] + metrics[1])
            rec = (metrics[0]) / (metrics[0] + metrics[3])
            f1 = 2 * prec * rec / (prec + rec)
            print_output('Metrics: (TP %d, FP %d, TN %d, FN %d)' % (metrics[0], metrics[1], metrics[2], metrics[3]))
            print_output('(Acc: %f, Precision: %f, Recall: %f, F1: %f)' % (acc, prec, rec, f1))
        
    if not config.lock_conv:
        helpers.save_model(scan_conv, out_dir, epoch, 'conv')
    if config.train_match:
        helpers.save_model(scan_match, out_dir, epoch, 'match')
    if config.train_transform:
        helpers.save_model(scan_transform, out_dir, epoch, 'transform')
    if (len(select.select([sys.stdin], [], [], 0)[0])):
        break

print_output("Completed training for {0} epochs".format(epoch + 1))