import matplotlib
matplotlib.use("QtAgg")

import matplotlib.pyplot as plt

print("before plot")

plt.plot([1,2,3],[1,4,9])

print("before show")

plt.show()

print("after show")