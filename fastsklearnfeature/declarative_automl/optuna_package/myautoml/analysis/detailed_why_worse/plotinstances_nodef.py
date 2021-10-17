import matplotlib.pyplot as plt
import numpy as np
import pickle
import openml
import glob
from scipy import stats as st


openml.config.apikey = '4384bd56dad8c3d2c0f6630c52ef5567'
openml.config.cache_directory = '/home/neutatz/phd2/cache_openml'

#new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results_step1.p', 'rb'))
#new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results.p', 'rb'))
#new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results_auto_sklearn.p', 'rb'))
#new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results_training_time_constraint.p', 'rb'))
#new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results_inference_time_constraint.p', 'rb'))

#new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results_only_success.p', 'rb'))

#new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results_pipeline_size_constraint.p', 'rb'))
#new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results_memory_size_constraint.p', 'rb'))


#new_dct = pickle.load(open('/home/neutatz/phd2/picture_progress/all_test_datasets/all_results_3weeks_5min_limit.p', 'rb'))#test


#file_list = glob.glob("/home/neutatz/data/automl_runs/mine_*.p") #sync long minute - no bias


#file_list = glob.glob("/home/neutatz/data/automl_runs_m2/new_p_run_*.p")

#nme = '1min_min_v12_'


#mmpath = '/home/neutatz/phd2/decAutoML2weeks_compare2default/sampling_sep20/rep1/'
#mmpath = '/home/neutatz/phd2/decAutoML2weeks_compare2default/sampling_sep20/rep10_tune/'
#nme = 'sample_instances_new_'

#mmpath = '/home/neutatz/phd2/decAutoML2weeks_compare2default/sampling_sep20/rep10_random/'
#nme = 'sample_instances_new_'

#mmpath = '/home/neutatz/phd2/decAutoML2weeks_compare2default/dynamic_space/space_random/'
#nme = 'sample_instances_new2_'

#bestttt
#mmpath = '/home/neutatz/phd2/decAutoML2weeks_compare2default/dynamic_space/space_al/'
#nme = 'sample_instances_new2_'

#mmpath = '/home/neutatz/phd2/decAutoML2weeks_compare2default/sep30/cv/'
#nme = 'sample_instances_new2_'

#mmpath = '/home/neutatz/phd2/decAutoML2weeks_compare2default/sep30/eval/'
#nme = 'sample_instances_new2_'

#mmpath = '/home/neutatz/phd2/decAutoML2weeks_compare2default/sep30/val/'
#nme = 'sample_instances_new2_'

#mmpath = '/home/neutatz/phd2/decAutoML2weeks_compare2default/oct4_features/'
#nme = 'sample_instances_new2_'

#mmpath = '/home/neutatz/phd2/decAutoML2weeks_compare2default/sep30/dynamic_data/'
#nme = 'sample_instances_new2_'

#mmpath = '/home/neutatz/phd2/decAutoML2weeks_compare2default/okt5_autosklearn/'
#nme = 'autosklearn_'

#mmpath = '/home/neutatz/phd2/decAutoML2weeks_compare2default/oct7_autogluon_all_cores/'
#nme = 'gluon_'

#mmpath = '/home/neutatz/phd2/decAutoML2weeks_compare2default/over_time/10min_model/'
#nme = 'sample_instances_new2_all_'

#mmpath = '/home/neutatz/phd2/decAutoML2weeks_compare2default/oct10/ddata/'
#nme = 'sample_v1_'

#mmpath = '/home/neutatz/phd2/decAutoML2weeks_compare2default/oct10/space/'
#nme = 'sample_v1_'

#mmpath = '/home/neutatz/phd2/decAutoML2weeks_compare2default/tpot/'
#nme = 'tpot_'

#mmpath = '/home/neutatz/phd2/decAutoML2weeks_compare2default/oct11_same_sampling/'
#nme = 'sample_v1_'

#mmpath = '/home/neutatz/phd2/decAutoML2weeks_compare2default/oct12_random_init/169/'
#nme = 'sample_v1_'

#mmpath = '/home/neutatz/phd2/decAutoML2weeks_compare2default/oct14_number_al_iterations/100/'
#nme = 'sample_v1_'

#mmpath = '/home/neutatz/phd2/decAutoML2weeks_compare2default/autosklearn_oct15/'
#nme = 'autosklearn_'

#mmpath = '/home/neutatz/phd2/decAutoML2weeks_compare2default/oct_17/m2_generalize/'
mmpath = '/home/neutatz/data/automl_runs/'
nme = 'sample_instances_new2_all_'



#mmpath = '/home/neutatz/phd2/decAutoML2weeks_compare2default/dynamic_data/al_dynamic/'
#nme = 'sample_instances_new2_'

#mmpath =  '/home/neutatz/phd2/decAutoML2weeks_compare2default/dynamic_data/random_dynamic/'
#nme = 'sample_instances_new2_'

#mmpath =  '/home/neutatz/phd2/decAutoML2weeks_compare2default/dynamic_data/al_dynamic/'
#nme = 'sample_instances_new2_'

#mmpath = '/home/neutatz/phd2/decAutoML2weeks_compare2default/sampling_sep20/rep10/'
#nme = 'sample_instances_'

#mmpath = '/home/neutatz/phd2/decAutoML2weeks_compare2default/sampling_sep20/autosklearn/'
#nme = 'autosklearn_'

#mmpath = '/home/neutatz/data/validation_10runs_v2/'
#nme = '1min_min_v12_'

file_list = glob.glob(mmpath + nme + "*.p")
#file_list = glob.glob("/home/neutatz/data/automl_runs/smac_*.p")
#file_list = glob.glob("/home/neutatz/data/validation_10runs/1min_min_v12_*.p")
#file_list = glob.glob("/home/neutatz/data/sampling_10runs/sample_v1_*.p")


print(file_list)

file_list_log = glob.glob(mmpath + "log_"+ nme +"*.p")#
print(file_list_log)


new_dict_sum_dynamic = {}
new_dict_sum_default = {}

better_than_default = {}
same_as_default = {}
worse_than_default = {}
significantly_better = {}
significantly_worse = {}


taskids = [168794, 168797, 168796, 189871, 189861, 167185, 189872, 189908, 75105, 167152, 168793, 189860, 189862, 126026, 189866, 189873, 168792, 75193, 168798, 167201, 167149, 167200, 189874, 167181, 167083, 167161, 189865, 189906, 167168, 126029, 167104, 126025, 75097, 168795, 75127, 189905, 189909, 167190, 167184]

for ttaskid in taskids:
    failed = False
    try:
        log_data = pickle.load(open(mmpath + 'log_' + nme + str(ttaskid) + '.p', 'rb'))
    except:
        failed = True
        print('failed: ' + str(ttaskid))

    #log_data_stat = pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/oct4_features/' + 'log_' + 'sample_instances_new2_' + str(ttaskid) + '.p', 'rb'))

    for ii in range(len(log_data['dynamic'])):

        constraints_i = ii

        dynamic_vals = []
        static_vals = []
        for r in range(len(log_data['dynamic'][ii].runs)):
            #if log_data['dynamic'][ii].runs[r].more.value*5.0/10.0 > 0.5:
            #if log_data['dynamic'][ii].runs[r].more.value > 0.5:
            if True:
                if failed==False:
                    dynamic_vals.append(log_data['dynamic'][ii].runs[r].test_score)
                    #dynamic_vals.append(log_data['static'][ii].runs[r].test_score)
                else:
                    dynamic_vals.append(0.0)
            else:
                #dynamic_vals.append(log_data_stat['static'][0].runs[r].test_score)
                pass
            #static_vals.append(log_data_stat['static'][0].runs[r].test_score)


        print(ttaskid)
        task = openml.tasks.get_task(ttaskid)
        name = task.get_dataset().name
        print('data: ' + name)

        dynamic_avg = np.average(dynamic_vals)
        static_avg = np.average(static_vals)

        print(st.ttest_ind(a=dynamic_vals, b=static_vals, equal_var=False))

        pval = (st.ttest_ind(a=dynamic_vals, b=static_vals, equal_var=False)).pvalue

        max_avg = max(np.average(dynamic_vals), np.average(static_vals))
        #max_avg = max(np.max(dynamic[constraints_i]), np.max(static[constraints_i]))

        #dynamic_avg /= max_avg
        #static_avg /= max_avg

        #dynamic_avg = dynamic_avg / static_avg
        #static_avg = 1.0

        print('dyn avg: constr:' + str(constraints_i) + ' : ' + str(dynamic_avg))
        print('static avg: constr:' + str(constraints_i) + ' : ' + str(static_avg))


        if not constraints_i in new_dict_sum_dynamic:
            new_dict_sum_dynamic[constraints_i] = []
            new_dict_sum_default[constraints_i] = []
            better_than_default[constraints_i] = []
            significantly_better[constraints_i] = []
            same_as_default[constraints_i] = []
            worse_than_default[constraints_i] = []
            significantly_worse[constraints_i] = []

        new_dict_sum_dynamic[constraints_i].append(dynamic_avg)
        new_dict_sum_default[constraints_i].append(static_avg)

        better_than_default[constraints_i].append(dynamic_avg > static_avg)
        significantly_better[constraints_i].append(dynamic_avg > static_avg and pval < 0.05)
        same_as_default[constraints_i].append(dynamic_avg == static_avg)
        worse_than_default[constraints_i].append(dynamic_avg < static_avg)
        significantly_worse[constraints_i].append(dynamic_avg < static_avg and pval < 0.05)

for k, v in new_dict_sum_dynamic.items():
    print(str(k) + ' dynamic: ' + str(np.nanmean(new_dict_sum_dynamic[k])) + ' +- ' + str(np.std(new_dict_sum_dynamic[k])))
    print(str(k) + ' default: ' + str(np.nanmean(new_dict_sum_default[k])) + ' +- ' + str(np.std(new_dict_sum_default[k])))

    print(str(k) + ' better: ' + str(np.sum(better_than_default[k])) + ' same: ' + str(np.sum(same_as_default[k])) + ' worse: ' + str(np.sum(worse_than_default[k])))
    print(str(k) + ' sig better: ' + str(np.sum(significantly_better[k])))
    print(str(k) + ' sig worse: ' + str(np.sum(significantly_worse[k])))
    print('\n\n')

print('default:')
print(new_dict_sum_default)

print('\ndynamic:')
print(new_dict_sum_dynamic)



