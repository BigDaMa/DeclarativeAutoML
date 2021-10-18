import matplotlib.pyplot as plt
import numpy as np
import pickle
import openml
import glob
from scipy import stats as st
from anytree import RenderTree


openml.config.apikey = '4384bd56dad8c3d2c0f6630c52ef5567'
openml.config.cache_directory = '/home/neutatz/phd2/cache_openml'

mmpath = '/home/neutatz/data/automl_runs/'
nme = 'sample_instances_new2_all_'

file_list = glob.glob(mmpath + nme +"*.p")
#file_list = glob.glob("/home/neutatz/data/automl_runs/smac_*.p")
#file_list = glob.glob("/home/neutatz/data/validation_10runs/1min_min_v12_*.p")
#file_list = glob.glob("/home/neutatz/data/sampling_10runs/sample_v1_*.p")


print(file_list)

file_list_log = glob.glob(mmpath + "log_" + nme + "*.p")#
print(file_list_log)


new_dict_sum_dynamic = {}
new_dict_sum_default = {}

better_than_default = {}
same_as_default = {}
worse_than_default = {}
significantly_better = {}
significantly_worse = {}


taskids = [168794, 168797, 168796, 189871, 189861, 167185, 189872, 189908, 75105, 167152, 168793, 189860, 189862, 126026, 189866, 189873, 168792, 75193, 168798, 167201, 167149, 167200, 189874, 167181, 167083, 167161, 189865, 189906, 167168, 126029, 167104, 126025, 75097, 168795, 75127, 189905, 189909, 167190, 167184]

number_params = {}

for ttaskid in taskids:
    log_data = pickle.load(open(mmpath + 'log_' + nme + str(ttaskid) + '.p', 'rb'))

    for ii in range(len(log_data['dynamic'])):
        print('\n\n')
        constraints_i = ii

        if not constraints_i in number_params:
            number_params[constraints_i] = []

        dynamic_vals = []
        static_vals = []
        best_run_id = -1
        best_run_val = 0
        for r in range(len(log_data['dynamic'][ii].runs)):
            dynamic_vals.append(log_data['dynamic'][ii].runs[r].test_score)

            if log_data['dynamic'][ii].runs[r].more.value > best_run_val:
                best_run_val = log_data['dynamic'][ii].runs[r].more.value
                best_run_id = r


        print(ttaskid)
        task = openml.tasks.get_task(ttaskid)
        name = task.get_dataset().name
        print('data: ' + name)

        dynamic_avg = np.average(dynamic_vals)
        print(dynamic_avg)

        #log_data['dynamic'][ii].runs[best_run_id].print_space()


        count_params = 0
        for pre, _, node in RenderTree(log_data['dynamic'][ii].runs[best_run_id].space.parameter_tree):
            if node.status == True:
                count_params += 1
        print(count_params)

        number_params[constraints_i].append(count_params)

        print('\n\n')


for k,v in number_params.items():
    print(str(k) + ': ' + str(np.average(v)) + ' +- ' + str(np.std(v)))