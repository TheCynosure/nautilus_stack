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
from config import Configuration, execution_config, evaluation_config

config = Configuration(False, True).parse()

start_time = str(int(time.time()))
initialize_logging(start_time, 'evaluate_')
print_output(config)

num_workers = int(execution_config['NUM_WORKERS'])

config.manualSeed = random.randint(1, 10000)  # fix seed
print_output("Random Seed: ", config.manualSeed)
random.seed(config.manualSeed)
torch.manual_seed(config.manualSeed)

scan_conv, scan_match, scan_transform = helpers.create_laser_networks(config.model_dir, config.model_epoch)
scan_conv.eval()
scan_match.eval()
dataset = helpers.load_laser_dataset(config.bag_file, '', 0, config.distance_cache, config.edge_trimming)
batch_count = len(dataset) // config.batch_size
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True)

metrics = np.zeros((4)) # True Positive, True Negative, False Positive, False Negative

print("Evaluation over {0} batches of size {1}".format(batch_count, config.batch_size))

with torch.no_grad():
    for i, data in tqdm(enumerate(dataloader, 0)):
        ((clouds, locations, _), (alt_clouds, alt_locs, _), labels) = data
        clouds = clouds.cuda()
        alt_clouds = alt_clouds.cuda()
        labels = labels.cuda()
        
        conv = scan_conv(clouds, alt_clouds)

        # import pdb; pdb.set_trace()
        #Compute match prediction
        scores = scan_match(conv)
        predictions = torch.argmax(torch.nn.functional.softmax(scores, dim=1), dim=1)
        metrics[0] += torch.sum((predictions + labels == 2)) # both prediction and lable are 1
        metrics[1] += torch.sum((predictions - labels == 1)) # prediction is 1 but label is 0
        metrics[2] += torch.sum((predictions + labels == 0)) # both prediction and label are 0
        metrics[3] += torch.sum((predictions - labels == -1)) # prediction is 0, label is 1

    print_output("Metrics:")
    print_output("TP: ", metrics[0])
    print_output("FP: ", metrics[1])
    print_output("TN: ", metrics[2])
    print_output("FN: ", metrics[3])

    acc = (metrics[0] + metrics[2]) / sum(metrics)
    prec = (metrics[0]) / (metrics[0] + metrics[1])
    rec = (metrics[0]) / (metrics[0] + metrics[3])
    f1 = 2 * prec * rec / (prec + rec)

    print_output('(Acc: %f, Precision: %f, Recall: %f, F1: %f)' % (acc, prec, rec, f1))
