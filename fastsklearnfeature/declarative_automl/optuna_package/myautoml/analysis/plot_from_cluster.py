import matplotlib.pyplot as plt
import numpy as np
import pickle
import openml

#path = '/home/neutatz/phd2/experiments_autoML/1hour/autosklearn/'
path = '/home/neutatz/phd2/experiments_autoML/1hour/mine/'

import glob

my_list = list(glob.glob(path + "*.p"))
my_list.sort()

for file_data in my_list:
    openml_id = int(file_data.split('_')[-1][:-2])
    new_dct = pickle.load(open(file_data, 'rb'))
    avg = np.average(new_dct[openml_id]['dynamic'][0])
    std = np.nanstd(new_dct[openml_id]['dynamic'][0])

    avg = np.average(new_dct[openml_id]['static'][0])
    std = np.nanstd(new_dct[openml_id]['static'][0])

    #print(str(openml_id) + ': ' + str(avg) + ' +- ' + str(std))
    print(str(openml_id) + ':' + str(avg))

