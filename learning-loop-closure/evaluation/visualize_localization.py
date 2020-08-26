import rosbag
import rospy
import argparse
from sensor_msgs.msg import PointCloud2
from evaluation_helpers import draw_map, visualize_localizations_from_bag

parser = argparse.ArgumentParser(
    description='Visualize the models classification for given inputs')
parser.add_argument('--bag_file', type=str, help='the bag from which to pull the localizations')
parser.add_argument('--localization_topic', type=str, default='/Cobot/Localization', help='localization topic')
parser.add_argument('--map_name', type=str, default='GDC3', help='map')
args = parser.parse_args()

bag = rosbag.Bag(args.bag_file)

# TODO do away with matplotlib
import matplotlib.pyplot as plt

visualize_localizations_from_bag(plt, bag, args.localization_topic)
draw_map(plt, '../../cobot/maps/{0}/{0}_vector.txt'.format(args.map_name))

plt.show()
