from matplotlib import pyplot as plt
import numpy as np

thresholds = np.array([2, 4, 10, 15, 20, 30, 50, 80])
accuracies = np.array([
    0.645588,
    0.683333,
    0.761765,
    0.805882,
    0.822549,
    0.773039,
    0.638728,
    0.504456,
])

precisions = np.array([
    0.983713,
    0.979487,
    0.936275,
    0.891960,
    0.845588,
    0.714396,
    0.583843,
    0.502238
])

recalls = np.array([
    0.296078,
    0.374510,
    0.561765,
    0.696078,
    0.789216,
    0.909804,
    0.966040,
    1
])

plt.plot(recalls, precisions, color='r', label="Threshold")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('Recall')
plt.ylabel('Precision')

plt.legend()

plt.show()