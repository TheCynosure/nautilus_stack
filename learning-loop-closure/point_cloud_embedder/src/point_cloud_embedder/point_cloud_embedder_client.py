#!/usr/bin/env python

import sys
import rospy
from sensor_msgs.msg import PointField, PointCloud2
from std_msgs.msg import Header
import numpy as np
from srv import GetPointCloudEmbedding

def create_point_cloud_message(points):
    """ Creates a point cloud message.
    Args:
        points: Nx2 array of xy position
    Returns:
        sensor_msgs/PointCloud2 message
    """
    ros_dtype = PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize

    points = np.append(points, np.zeros((points.shape[0], 1)), axis=1)

    data = points.astype(dtype).tobytes()

    fields = [PointField(
        name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate('xyz')]

    header = Header(frame_id='map', stamp=rospy.Time.now())

    pc = PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        is_dense=True,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 3),
        row_step=(itemsize * 3 * points.shape[0]),
        data=data
    )

    return pc

def embed_cloud_client(cloud_file):
    rospy.init_node('embed_cloud_client')
    rospy.wait_for_service('embed_point_cloud')
    cloud = np.load(cloud_file)
    msg = create_point_cloud_message(cloud)
    try:
        get_embedding = rospy.ServiceProxy('embed_point_cloud', GetPointCloudEmbedding)
        resp1 = get_embedding(msg)
        return resp1.embedding
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

if __name__ == "__main__":
    point_cloud_file = sys.argv[1]
    print "The Result of embedding the point cloud is: {0}".format(embed_cloud_client(point_cloud_file))
    