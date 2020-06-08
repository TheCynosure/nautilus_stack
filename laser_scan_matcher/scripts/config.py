import numpy as np

import argparse

class Configuration:
  def __init__(self, train=False, laser=True, evaluation=False, data_processing=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)

    if train:
      parser.add_argument('--train_set', default='train', help='Which subset of the dataset to use for training')
      parser.add_argument('--validation_set', default='val', help='Which subset of the dataset to use for validation')
      parser.add_argument('--num_epoch', default=90, type=int)
      parser.add_argument('--outf', type=str, default='matcher', help='output folder')
      parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for optimizer')
      parser.add_argument('--distance_cache', type=str, default=None, help='cached overlap info to start with')
      if laser:
        parser.add_argument('--laser_match_weight', default=0.8, type=float)
        parser.add_argument('--laser_trans_weight', default=0.2, type=float)
        parser.add_argument('--name', type=str, default='laser_dataset', help="name for the created dataset/models")
        parser.add_argument('--train_transform',action='store_true')
        parser.add_argument('--train_match',action='store_true')
        parser.add_argument('--lock_conv', action='store_true')
        parser.add_argument('--dist_close_ratio', default=20, type=int, help='The number of distant examples to choose per "close" example')

    if laser:
      parser.add_argument('--bag_file', type=str, required=True, help="bag file")
      parser.add_argument('--model_dir', type=str, default='', help='directory containing pretrained model to start with')
      parser.add_argument('--model_epoch', type=str, default='', help='epoch number for pretrained model to start with')
      parser.add_argument('--lidar_topic', type=str, default='/Cobot/Laser')
      parser.add_argument('--localization_topic', type=str, default='/Cobot/Localization')

    if evaluation and not laser:
      parser.add_argument('--threshold_min', type=float, default=0, help='minimum threshold to test for evaluation')
      parser.add_argument('--threshold_max', type=float, default=16, help='maximum threshold to test for evaluation')
      parser.add_argument('--evaluation_set', default='val', help='Which subset of the dataset to use for evaluation')

    if data_processing:
      parser.add_argument('--lidar_fov', type=float, default=np.pi)
      parser.add_argument('--lidar_max_range', type=float, default=10)
      parser.add_argument('--overlap_radius', type=float, default=4)
      parser.add_argument('--overlap_sample_resolution', type=int, default=10)
      parser.add_argument('--close_distance_threshold', type=float, default=2.0)
      parser.add_argument('--far_distance_threshold', type=float, default=3.5)
      parser.add_argument('--overlap_similarity_threshold', type=float, default=0.7)
      parser.add_argument('--time_ignore_threshold', type=float, default=1.0)
      if not laser:
        parser.add_argument('--augmentation_probability', type=float, default=0.8)


    self.parser = parser

  def add_argument(self, *args, **kwargs):
    self.parser.add_argument(*args, **kwargs)

  def parse(self):
    return self.parser.parse_args()


execution_config = {
  'BATCH_SIZE': 768,
  'NUM_WORKERS': 8,
}

training_config = {
  'TRAIN_SET': 'train',
  'VALIDATION_SET': 'val',
  'NUM_EPOCH': 90,
  'LASER_MATCH_WEIGHT': 0.8,
  'LASER_TRANS_WEIGHT': 0.2
}

evaluation_config = {
  'THRESHOLD_MIN': 0,
  'THRESHOLD_MAX': 16,
  'EVALUATION_SET': 'val'
}

lidar_config = {
  'FOV': np.pi,
  'MAX_RANGE': 10,
}

data_config = {
  'OVERLAP_RADIUS': 4,
  'OVERLAP_SAMPLE_RESOLUTION': 10,
  'CLOSE_DISTANCE_THRESHOLD': 1.5,
  'FAR_DISTANCE_THRESHOLD': 3,
  'OVERLAP_SIMILARITY_THRESHOLD': 0.7,
  'TIME_IGNORE_THRESHOLD': 1,
  'MATCH_REPEAT_FACTOR': 1,
  'AUGMENTATION_PROBABILITY': 0.8
}

DEV_SPLIT = 0.8
data_generation_config = {
  'TIME_SPACING': 0.001,
  'TRAIN_SPLIT': 0.15,
  'DEV_SPLIT': DEV_SPLIT,
  'VAL_SPLIT': 1 - DEV_SPLIT,
  'LIDAR_TOPIC': '/Cobot/Laser',
  'LOCALIZATION_TOPIC': '/Cobot/Localization',
  'MAX_PARTITION_COUNT': 10,
  'MAX_PARTITION_SIZE': 1200,
  'MIN_PARTITION_SIZE': 50
}

visualization_config = {
  'TIMESTEP': 1.5
}
