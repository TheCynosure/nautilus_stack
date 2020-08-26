
import argparse
import os
import sys
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help="source dataset ")
parser.add_argument('--combine_rule', type=str, default='and', help="one of (and, or)")
parser.add_argument('--out_file', type=str, default='lcc_timestamps', required=True, help="name of output file to store this timestamp info in (default: lcc_timestamps)")
parser.add_argument('--labeled_timestamps', type=str, nargs='+' required=True, help="paths to timestamp labels to join together")
opt = parser.parse_args()
start_time = str(int(time.time()))
print(opt)

dataset = LCDataset(args.dataset)

all_labeled = {}

for dataset in opt.labeled_timestamps:
    timestamps = np.load(args.timestamps)
    for timestamp in timestamps:
        timestamp = timestamps[i][0]
        if float(timestamp) > 5e5: # must be since epoch...
            timestamp = round(float(timestamp) - dataset.dataset_info['startTime'], 5)
        label = int(timestamps[i][1])
        _, _, true_ts = dataset.get_by_nearest_timestamp(timestamp)

        if true_ts in all_labeled:
            if (opt.combine_rule == 'and'):
                all_labeled[true_ts] = label and all_labeled[true_ts]
            elif (opt.combine_rule == 'or'):
                all_labeled[true_ts] = label or all_labeled[true_ts]
            else:
                raise Error('Invalid combine rule')
        else:
            all_labeled[true_ts] = label

labeled_dataset = np.array(list(all_labeled.items()), dtype=np.float32)

np.save(opt.out_file + '.npy', labeled_dataset)