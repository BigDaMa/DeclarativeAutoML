import numpy as np

dynamic = np.array([1,2,3,4,5])
static = np.array([8, 3, 1, 0, 0])

static.sort()
dynamic.sort()

print(np.sum(dynamic>static)/5.0)

print(static)