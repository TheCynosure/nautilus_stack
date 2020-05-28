#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from srv import MatchLaserScans, MatchLaserScansResponse
import sys
from os import path
import torch

# TODO fix this hack
sys.path.append(path.dirname(path.dirname(path.dirname( path.dirname( path.abspath(__file__)  ) ) )))
from helpers import create_laser_networks

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