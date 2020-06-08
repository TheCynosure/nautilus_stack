#!/usr/bin/env python
import numpy as np
from sensor_msgs.msg import LaserScan
from laser_scan_matcher.srv import MatchLaserScans, MatchLaserScansResponse
import sys
from os import path
import torch
import rospy

def create_laser_networks(model_dir, model_epoch):
    scan_conv = ScanConvNet()
    if model_dir:
        scan_conv.load_state_dict(torch.load(os.path.join(model_dir, 'model_conv_' + model_epoch + '.pth')))

    scan_transform = ScanTransformNet()
    if model_dir:
        scan_transform.load_state_dict(torch.load(os.path.join(model_dir, 'model_transform_' + model_epoch + '.pth')))

    scan_match = ScanMatchNet()
    if model_dir:
        scan_match.load_state_dict(torch.load(os.path.join(model_dir, 'model_match_' + model_epoch + '.pth')))
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        scan_conv = torch.nn.DataParallel(scan_conv)
        scan_match = torch.nn.DataParallel(scan_match)
        scan_transform = torch.nn.DataParallel(scan_transform)

    scan_conv.cuda()
    scan_match.cuda()
    scan_transform.cuda()
    return scan_conv, scan_match, scan_transform

def create_scan_match_helper(scan_conv, scan_match):
  def match_scans(req):
    
    scan_np = np.array(req.scan.ranges).astype(np.float32)
    alt_scan_np = np.array(req.alt_scan.ranges).astype(np.float32)
    
    scan = torch.tensor(scan_np).cuda()
    alt_scan = torch.tensor(alt_scan_np).cuda()

    conv = scan_conv(scan.unsqueeze(0), alt_scan.unsqueeze(0))
    scores = scan_match(conv)
    probs = torch.nn.functional.softmax(scores.squeeze())

    return MatchLaserScansResponse(probs[1])
    
  return match_scans


def service():
  rospy.init_node('laser_scan_matcher', anonymous=True)
  conv_model = sys.argv[1]
  match_model = sys.argv[2]
  # transform_model = sys.argv[3]
  scan_conv, scan_match, scan_transform = create_laser_networks(conv_model, match_model)
  scan_conv.eval()
  scan_match.eval()
  service = rospy.Service('match_laser_scans', MatchLaserScans, create_scan_match_helper(scan_conv, scan_match), buff_size=2)
  rospy.spin()
  
if __name__ == '__main__':
  try:
    service()
  except rospy.ROSInterruptException:
    pass
