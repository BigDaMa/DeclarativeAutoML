import pickle
import glob
import  numpy as np


#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/autosklearn_oct15'
#nametag = '/autosklearn_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/tpot'
#nametag = '/tpot_'


#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/dez31_1week_res_no_fallback'
#nametag = '/sample_instances_new2_all_'

folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/dez31_1week_res_no_fallback'
nametag = '/sample_instances_new2_all_'


#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/mai26_3weeks'
#nametag = '/sample_instances_new2_all_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/june27_autosklearn_pipelinesize'
#nametag = '/autosklearn_pipelinesize_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/dez3_pipeline_size_m1'
#nametag = '/sample_instances_new2_all_constraints_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/jul3_training_time'
#nametag = '/autosklearn_training_time_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/dez3_training_time_m2'
#nametag = '/sample_instances_new2_all_constraints_training_time__'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/jun30_autosklearn_inference_time'
#nametag = '/autosklearn_inference_time_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/dez3_inference_time'
#nametag = '/

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/jul_3_autosklearn_inference_time2'
#nametag = '/autosklearn_inference_time_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/jul20_static_fair'
#nametag = '/sample_instances_new2_all_constraints_fairness__'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/jul_26_emukit_inference_time'
#nametag = '/sample_instances_new2_all_constraints_emukit_inference_time__'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/jul27_emukit_training_time'
#nametag = '/sample_instances_new2_all_constraints_emukit_training_time__'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/jul27_autosklearn_fairness'
#nametag = '/autosklearn_fairness_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/jul28_static_fair'
#nametag = '/sample_instances_new2_all_constraints_fairness__'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/jul28_emukit_pipeline_size'
#nametag = '/sample_instances_new2_all_constraints_emukit_pipeline_size__'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/jul29_emukit_inference_time_times_1'
#nametag = '/sample_instances_new2_all_constraints_emukit_inference_time__'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/jul29_emukit_fair'
#nametag = '/emukit_fairness__'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/jul31_emukit_inference_time'
#nametag = '/sample_instances_new2_all_constraints_emukit_inference_time__'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/aug1_emukit_training_time'
#nametag = '/sample_instances_new2_all_constraints_emukit_training_time__'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/aug2_fairness_dynamic'
#nametag = '/sample_instances_new2_all_constraints_fairness__'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/aug4_results_2week_dynamic'
#nametag = '/sample_instances_new2_all_constraints_fairness__'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/aug16_normal_5min'
#nametag = '/run5__'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/aug16_ensemble_5min'
#nametag = '/run5_ensemble__'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/aug17_ensemble_5min_full_val'
#nametag = '/run5_ensemble__'

#ensemble but training without ensemble
#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/aug21_prob_ensemble_with_val_single_tuning'
#nametag = '/ensemble_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/aug25_ensemble_median_pruning'
#nametag = '/static_ensemble_'

#autosklearn
folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/aug21_autosklearn2_describe'
nametag = '/autosklearn_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/aug30_probe_1min_ensemble_successive'
#nametag = '/ensemble_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/sep1_new_system'
#nametag = '/static_ensemble_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/sep3_1min_1day_new_ensemble_probe'
#nametag = '/ensemble_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/sep9_1week_results'
#nametag = '/ensemble_'

folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/sep16_indus_40cpus'
nametag = '/ensemble_'

print(glob.glob( folder + nametag + '*'))

my_list = ['168797', '167200', '189862', '168794', '75097', '189905', '189866', '168796', '75193', '167185', '75105', '167168', '168793', '167152', '126025', '168798', '167083', '167149', '189908', '126026', '189865', '167181', '189906', '189874', '189872', '167190', '167104', '189909', '167161', '75127', '167201', '189861', '126029', '189860', '189873', '167184', '168795', '168792', '189871']

#my_list = ['0', '1', '2', '3', '4']


data_means = {}

#tag = 'static'
tag = 'dynamic'

for constraint_i in range(10):
    data_all = []
    data_means[constraint_i] = {}

    for datasetid in my_list:
        data = pickle.load(open(folder + nametag + datasetid + '.p', "rb"))
        #print(data)
        data_all.append(data[tag][constraint_i])
        data_means[constraint_i][datasetid] = np.mean(data[tag][constraint_i])
        print(str(datasetid) + ',' + str(data_means[constraint_i][datasetid]))
    print('##################\n\n')

