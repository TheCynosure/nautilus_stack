import argparse
import numpy as np
import torch
import random
from ltf_model import SegNet
from ltf_dataset import LTFDataset
import time
from matplotlib import pyplot as plt

from PIL import Image

import sys
import os
sys.path.append(os.path.join(os.getcwd(), '..'))
from helpers import print_output, initialize_logging, save_model


parser = argparse.ArgumentParser()
parser.add_argument(
    '--model', type=str, help='path to a pretrained model')
parser.add_argument(
    '--bag_file', type=str, default='train', help='path to a bag containing base and filtered scans.')
parser.add_argument(
    '--filtered_topic', type=str, default='/filtered', help='topic to look for filtered scans')
parser.add_argument(
    '--base_topic', type=str, default='/Cobot/Laser', help='topic to look for base scans')

opt = parser.parse_args()
start_time = str(int(time.time()))
initialize_logging(start_time)

print_output(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print_output("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = LTFDataset(opt.bag_file, opt.base_topic, opt.filtered_topic, 200)

ltf_model = SegNet(3, 2)
ltf_model.load_state_dict(torch.load(opt.model))
ltf_model.eval()
ltf_model = ltf_model.cuda()
import matplotlib.pyplot as plt

for original, filtered in dataset:
  plt.figure(0)
  plt.title('original')
  plt.imshow(original[0, :, :])
  plt.figure(1)
  plt.title('filtered')
  plt.imshow(filtered)
  import pdb; pdb.set_trace()
  generated = ltf_model(torch.tensor(original).cuda().unsqueeze(0))[1][0]
  # now, its 2 x img_size, where [0] is the prob for class 0, [1] is the prob for class 1
  plt.figure(2)
  plt.title('generated score')
  plt.imshow(generated[1].detach().cpu() * 255)
  plt.figure(3)
  plt.title('generated pred')
  plt.imshow(torch.argmax(generated, axis=0).detach().cpu() * 255)

  plt.show()

print('Done...')