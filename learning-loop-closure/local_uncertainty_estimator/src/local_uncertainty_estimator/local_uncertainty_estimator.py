#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from srv import EstimateLocalUncertainty, EstimateLocalUncertaintyResponse
import sys
from os import path
import torch

# TODO fix this hack
sys.path.append(path.dirname(path.dirname(path.dirname( path.dirname( path.abspath(__file__)  ) ) )))
from helpers import create_lu_networks

def create_lu_helper(scan_conv, scan_uncertainty):
  def match_scans(req):
    scan_np = np.array(req.scan.ranges).astype(np.float32)
    
    scan = torch.tensor(scan_np).cuda()
    with torch.no_grad():
      conv = scan_conv(scan.unsqueeze(0).unsqueeze(0))
      condition, scale = scan_uncertainty(conv)[0]

      return EstimateLocalUncertaintyResponse(condition, scale)
    
  return match_scans


def service():
  rospy.init_node('local_uncertainty_estimator', anonymous=True)
  model_dir = sys.argv[1]
  model_epoch = sys.argv[2]
  # transform_model = sys.argv[3]
  scan_conv, scan_uncertainty = create_lu_networks(model_dir, model_epoch)
  scan_conv.eval()
  scan_uncertainty.eval()
  service = rospy.Service('estimate_local_uncertainty', EstimateLocalUncertainty, create_lu_helper(scan_conv, scan_uncertainty), buff_size=2)
  rospy.spin()
  
if __name__ == '__main__':
  try:
    service()
  except rospy.ROSInterruptException:
    pass