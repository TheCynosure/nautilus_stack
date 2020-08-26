#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from srv import GetPointCloudEmbedding, GetPointCloudEmbeddingResponse
import sys
from os import path

# TODO fix this hack
sys.path.append(path.dirname(path.dirname(path.dirname( path.dirname( path.abspath(__file__)  ) ) )))
from evaluation.evaluation_helpers import embedding_for_scan
from helpers import create_embedder

def create_embed_helper(embedder):
  def embed_cloud(req):
    cloud_np = []
    for point in point_cloud2.read_points(req.cloud, skip_nans=True):
      cloud_np.append([point[0], point[1]])
    cloud_np = np.array(cloud_np).astype(np.float32)
    embedding = embedding_for_scan(embedder, cloud_np)[0].cpu().detach().numpy()
    return GetPointCloudEmbeddingResponse(embedding.squeeze())
  return embed_cloud


def service():
  rospy.init_node('point_cloud_embedder', anonymous=True)
  embedding_model = sys.argv[1]
  embedder = create_embedder(embedding_model)
  embedder.eval()
  service = rospy.Service('embed_point_cloud', GetPointCloudEmbedding, create_embed_helper(embedder), buff_size=2)
  rospy.spin()
  
if __name__ == '__main__':
  try:
    service()
  except rospy.ROSInterruptException:
    pass