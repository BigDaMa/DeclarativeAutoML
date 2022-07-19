import matplotlib.pyplot as plt
import numpy as np
import pickle
import openml

new_dict1 = pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/july19_fairness_distribution/constraints_fairness.p', 'rb'))

print(len(new_dict1))
new_dict = []
for item in new_dict1:
    if type(item) != type(None):
        new_dict.append(item)

plt.hist(new_dict, bins=1000)
print(len(new_dict))
print('min: ' + str(np.min(new_dict)) + ' max: ' + str(np.max(new_dict)) + ' median: ' + str(np.median(new_dict)))
print("1 perc: " + str(np.percentile(new_dict, 1)))
print("2 perc: " + str(np.percentile(new_dict, 2)))
print("4 perc: " + str(np.percentile(new_dict, 4)))
print("8 perc: " + str(np.percentile(new_dict, 8)))
print("16 perc: " + str(np.percentile(new_dict, 16)))
print("32 perc: " + str(np.percentile(new_dict, 32)))

print("99 perc: " + str(np.percentile(new_dict, 99)))
print("98 perc: " + str(np.percentile(new_dict, 98)))
print("96 perc: " + str(np.percentile(new_dict, 96)))
print("92 perc: " + str(np.percentile(new_dict, 92)))
print("84 perc: " + str(np.percentile(new_dict, 84)))
print("68 perc: " + str(np.percentile(new_dict, 68)))

plt.show()
