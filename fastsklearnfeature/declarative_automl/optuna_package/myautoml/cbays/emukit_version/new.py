import numpy as np


all = list(range(10))
val_list = [0,2]

print(np.random.choice(list(set(all) - set(val_list))))