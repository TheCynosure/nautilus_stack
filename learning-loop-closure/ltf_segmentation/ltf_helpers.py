import numpy as np
import math

# Takes an input point cloud and discretizes it into an occupancy grid of size (dimensions x dimensions)
# Assumes the input point cloud is normalized to [-1, 1]
def discretize_point_cloud(cloud, max_range, dimensions, rgb=False):
  discretized = np.zeros((dimensions, dimensions, 3)).astype(np.float32) if rgb else np.zeros((dimensions, dimensions)).astype(np.int)

  for p in cloud:
    norm_p = p / max_range
    cell = (int(math.floor(dimensions/2 + norm_p[0] * dimensions/2)), int(math.floor(dimensions/2 + norm_p[1] * dimensions/2)))
    if rgb:
      discretized[cell][0] = 1
    else:
      discretized[cell] = 1
  
  return discretized