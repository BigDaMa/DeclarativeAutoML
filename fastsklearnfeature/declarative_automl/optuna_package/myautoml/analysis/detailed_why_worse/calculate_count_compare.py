import matplotlib.pyplot as plt
import numpy as np
import pickle
import openml

#new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results_step1.p', 'rb'))
#new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results.p', 'rb'))
#new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results_auto_sklearn.p', 'rb'))
#new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results_training_time_constraint.p', 'rb'))
#new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results_inference_time_constraint.p', 'rb'))

#new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results_only_success.p', 'rb'))

#new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results_pipeline_size_constraint.p', 'rb'))
#new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results_memory_size_constraint.p', 'rb'))

new_dct = pickle.load(open('/home/neutatz/phd2/picture_progress/all_test_datasets/all_results_cost_sens.p', 'rb'))#test
#new_dct = pickle.load(open('/home/neutatz/phd2/picture_progress/all_test_datasets/all_results_squared4.p', 'rb'))#test

new_dct_p2 = pickle.load(open('/home/neutatz/phd2/picture_progress/all_test_datasets/all_results_cost_sens_p2.p', 'rb'))#test

dest = dict(new_dct)  # or orig.copy()
dest.update(new_dct_p2)
new_dct = dest

#new_dct = pickle.load(open('/home/neutatz/phd2/picture_progress/all_test_datasets/all_results_3weeks_5min_limit.p', 'rb'))#test


new_dict_sum_dynamic = {}
new_dict_sum_default = {}

for k, v in new_dct.items():
    print(k)
    name = openml.datasets.get_dataset(dataset_id=k, download_data=False).name
    print('data: ' + name)

    static = v['static']
    dynamic = v['dynamic']

    for constraints_i in range(len(dynamic)):
        if not constraints_i in new_dict_sum_dynamic:
            new_dict_sum_dynamic[constraints_i] = 0.0
            new_dict_sum_default[constraints_i] = 0.0

        dynamic_avg = np.average(dynamic[constraints_i])
        static_avg = np.average(static[constraints_i])



        max_avg = max(np.average(dynamic[constraints_i]), np.average(static[constraints_i]))
        #max_avg = max(np.max(dynamic[constraints_i]), np.max(static[constraints_i]))

        if dynamic_avg >= static_avg:
            new_dict_sum_dynamic[constraints_i] += 1

        if static_avg >= dynamic_avg:
            new_dict_sum_default[constraints_i] += 1

print(float(len(new_dct)))

for k, v in new_dict_sum_default.items():
    print('dynamic: ' + str(new_dict_sum_dynamic[k] / float(len(new_dct))))
    print('default: ' + str(new_dict_sum_default[k] / float(len(new_dct))))
    print('\n\n')
print(new_dict_sum_default)
print(new_dict_sum_dynamic)


