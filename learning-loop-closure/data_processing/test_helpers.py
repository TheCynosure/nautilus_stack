from helpers import compute_overlap, test_point
import numpy as np
import matplotlib.pyplot as plt

loc_a = np.array([0, 0, 0])
loc_b = np.array([0, 0, np.pi / 2])

loc_c = np.array([-2, 0, 0])
loc_d = np.array([2, 0, np.pi])
loc_e = np.array([.75 * 4, 0, np.pi])

print("AA", compute_overlap(loc_a, loc_a))
print("AB", compute_overlap(loc_a, loc_b))
print("AC", compute_overlap(loc_a, loc_c))
print("AD", compute_overlap(loc_a, loc_d))
print("AE", compute_overlap(loc_a, loc_e))

test_a = [66.32367  , 51.624554 , -1.6080709]
test_b = [66.05289  , 50.675644 , -2.0107317]

print("test", compute_overlap(test_b, test_a))
print("test", compute_overlap(test_a, test_b))

test_c = np.array([119.85415  ,  68.4586   ,   1.9963497])
test_d = np.array([119.95093  ,  67.765594 ,   1.6020247])

print("test", compute_overlap(test_c, test_d))
print("test", compute_overlap(test_d, test_c))

test_locs = [
  [-2.9331049e+01, -1.9391386e+01,  1.2786581e-02],
  [-29.3533  , -19.266445,  -3.127946]
]

print("test real", compute_overlap(test_locs[0], test_locs[1]))

# Explicitly testing test_point fn.

center = np.array([5, 5])
thetas = np.linspace(0, 2 * np.pi, 12)

location_ne = [center[0], center[1], np.pi / 4]
location_nw = [center[0], center[1], 3 * np.pi / 4]
location_sw = [center[0], center[1], 5 * np.pi / 4]
location_se = [center[0], center[1], 7 * np.pi / 4]
locations = [location_ne, location_nw, location_sw, location_se]

for location in locations:
  points = []
  colors = []
  for theta in thetas:
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    distance = 2.5
    point = center + np.dot(R, [distance, 0])
    result = test_point(location, point)
    points.append(point)
    colors.append(result)
  points = np.array(points)
  plt.title("LOCATION")
  plt.arrow(0, 0, 2 * np.cos(location[2]), 2 * np.sin(location[2]))
  plt.scatter(points[:, 0], points[:, 1], c=colors)
  plt.show()