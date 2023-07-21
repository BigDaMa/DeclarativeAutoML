import pickle
import matplotlib.pyplot as plt

my_dict_new = pickle.load(open('/home/felix/phd2/machine_aware/avg_series_fast_laptop.p', "rb"))
my_dict = pickle.load(open('/home/felix/phd2/machine_aware/avg_series.p', "rb"))

my_dict_new = pickle.load(open('/home/felix/phd2/machine_aware/big_data/avg_series_4job.p', "rb"))
my_dict = pickle.load(open('/home/felix/phd2/machine_aware/big_data/avg_series_1job.p', "rb"))

print(my_dict.keys())

plt.plot(my_dict_new['time'], my_dict_new['avg'], label='new')
plt.plot(my_dict['time'], my_dict['avg'], label='old')
plt.legend()
plt.show()

specified_time = 60

#find
found_val = 0.0
for i in range(len(my_dict_new['time'])):
    if my_dict_new['time'][i] <= specified_time:
        found_val = my_dict_new['avg'][i]
    else:
        break

print('found val: ' + str(found_val))

found_time = 0.0
for i in range(len(my_dict['time'])):
    if my_dict['avg'][i] <= found_val:
        found_time = my_dict['time'][i]
    else:
        break

print('found time: ' + str(found_time))