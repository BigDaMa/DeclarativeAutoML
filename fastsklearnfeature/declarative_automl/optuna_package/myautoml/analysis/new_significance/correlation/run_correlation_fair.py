import pickle
import glob
import numpy as np
from scipy.stats import mannwhitneyu


#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/autosklearn_oct15'
#nametag = '/autosklearn_'

folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/tpot'
nametag = '/tpot_'


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
folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/aug21_prob_ensemble_with_val_single_tuning'
nametag = '/ensemble_'

#autosklearn
folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/aug21_autosklearn2_describe'
nametag = '/autosklearn_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/aug23_ensemble_with_ensemble_specific_1day_tuning'
#nametag = '/ensemble_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/aug24_new-ensemble_static'
#nametag = '/static_ensemble_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/aug24_single_best_random_val'
#nametag = '/static_ensemble_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/aug_25_ensemble_with_dummy'
#nametag = '/static_ensemble_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/aug25_ensemble_fix_dummy'
#nametag = '/static_ensemble_'

#aug25_ensemble_incremental_data
#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/aug25_ensemble_incremental_data'
#nametag = '/static_ensemble_'

#aug25_ensemble_median_pruning
#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/aug25_ensemble_median_pruning'
#nametag = '/static_ensemble_'

#dynamic ensemble with pruning
#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/aug30_probe_1min_ensemble_successive'
#nametag = '/ensemble_'

#sep1_new_system
#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/sep1_new_system'
#nametag = '/static_ensemble_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/sep3_1min_1day_new_ensemble_probe'
#nametag = '/ensemble_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/sep7_indus_40_cpus_1day_1min'
#nametag = '/ensemble_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/sep7_machine4_parallel_20'
#nametag = '/ensemble_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/sep9_1week_results'
#nametag = '/ensemble_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/sep12_training_time_1min'
#nametag = '/sample_instances_new2_all_constraints_training_time__'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/sep16_indus_40cpus'
#nametag = '/ensemble_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/sep15_machine4_1week_2hps_probe'
#nametag = '/ensemble_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/sep16_running_40cpu_model_on_20_machine'
#nametag = '/ensemble_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/sep16_10seconds'
#nametag = '/ensemble_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/sep19_1day_autosklearn_comparison_res'
#nametag = '/ensemble_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/sep19_9-days_6min_2days'
#nametag = '/ensemble_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/sep21_fair_results'
#nametag = '/sample_instances_new2_all_constraints_fairness__'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/sep23_3days_compare_both_indus'
#nametag = '/ensemble_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/september23_training_time'
#nametag = '/sample_instances_new2_all_constraints_training_time__'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/sep24_pipeline_size'
#nametag = '/sample_instances_new2_all_constraints_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/sep25_autosklearn_comp_res'
#nametag = '/ensemble_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/sep26_inference_time'
#nametag = '/sample_instances_new2_all_constraints_inference_time__'

folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/sep26_searchtime_constraint_model'
nametag = '/ensemble_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/sep26_threshold_pipelinesize'
#nametag = '/sample_instances_new2_all_constraints_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/tpot'
#nametag = '/tpot_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/oct2_constraint_model_2week_res_time'
#nametag = '/ensemble_'

folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/okt3_2week-optimzed_model_res'
nametag = '/ensemble_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/okt4_2week_opt_fairness_res'
#nametag = '/sample_instances_new2_all_constraints_fairness__'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/okt6_1day_compare_autosklearn2_pegasus'
#nametag = '/ensemble_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/okt6_try_optimized_model_on_indus_searchtime'
#nametag = '/ensemble_'

folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/okt6_pipeline_size_halfoptimized'
nametag = '/sample_instances_new2_all_constraints_'

folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/autosklearn2_full_space_probe'
nametag = '/autosklearn_'

folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/aug21_autosklearn2_describe'
nametag = '/autosklearn_'

folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/okt6_autosklearn2_10_30sec'
nametag = '/autosklearn_'

folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/okt6_10_30sec_opt_autosklearn'
nametag = '/autosklearn_'

folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/okt6_autogluon_10s_30s'
nametag = '/gluon_'

folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/okt7_m1_al_stepwise'
nametag = '/stepwise_'

folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/okt7_m2_random_stepwise'
nametag = '/stepwise_'

folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/okt8_probingtime'
nametag = '/probing_time_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/okt7_optimized2w_pipeline_size'
#nametag = '/sample_instances_new2_all_constraints_'


folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/okt8_static_system'
nametag = '/static_ensemble_'

folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/okt9_probing_time_fallback'
nametag = '/probing_time_'

folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/okt9_inference_time_optimized'
nametag = '/sample_instances_new2_all_constraints_inference_time__'


folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/okt10_emu'
nametag = '/emukit__'

folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/okt6_autogluon_10s_30s'
nametag = '/gluon_'

folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/okt10_dynamic'
nametag = '/ensemble_'

folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/okt11_optimized-seachtime_range'
nametag = '/ensemble_'


folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/okt10_optimized_stepwise_al'
nametag = '/stepwise_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/okt12_training_time_opt_model'
#nametag = '/sample_instances_new2_all_constraints_training_time__'


#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/okt13_random_stepwise-13_opt'
#nametag = '/stepwise_'


folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/okt13_random_opt_4-7'
nametag = '/stepwise_'

folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/okt14_autogluon'
nametag = '/gluon_'


#folder = '/home/felix/phd2/dec_automl/okt26_res_opt'
#nametag = '/ensemble_'

#folder = '/home/felix/phd2/dec_automl/oct29_100k_random_configs'
#nametag = '/ensemble_'

#folder = '/home/felix/phd2/dec_automl/okt28_res_with_shuffled_al'
#nametag = '/ensemble_'



#folder = '/home/felix/phd2/dec_automl/okt29_random_opt_1week'
#nametag = '/ensemble_'

#folder = '/home/felix/phd2/dec_automl/okt31_stepwise_al'
#nametag = '/stepwise_'

#folder = '/home/felix/phd2/dec_automl/ok31_stepwise_random'
#nametag = '/stepwise_'

#folder = '/home/felix/phd2/dec_automl/nov1_2week_al_model_opt_res'
#nametag = '/ensemble_'

#folder = '/home/felix/phd2/dec_automl/nov02_1k_good_random_configs_res'
#nametag = '/ensemble_'

#nov2_augemented_bo_configs_res
#folder = '/home/felix/phd2/dec_automl/nov2_augemented_bo_configs_res'
#nametag = '/ensemble_'

#dez04_random_opt_model_2weeks_res
folder = '/home/felix/phd2/dec_automl/dez04_random_opt_model_2weeks_res'
nametag = '/ensemble_'

#nov10_res_1_week_bo_configs
folder = '/home/felix/phd2/dec_automl/nov10_res_1_week_bo_configs'
nametag = '/ensemble_'

#nov11_1week_bo_configs_1week_opt_random_model
folder = '/home/felix/phd2/dec_automl/nov11_1week_bo_configs_1week_opt_random_model'
nametag = '/ensemble_'


#nov13_1week_random_uniform_data_sampling
folder = '/home/felix/phd2/dec_automl/nov13_1week_random_uniform_data_sampling'
nametag = '/ensemble_'

#nov15_random_sampling_with_eval_res
folder = '/home/felix/phd2/dec_automl/nov15_random_sampling_with_eval_res'
nametag = '/ensemble_'


#folder = '/home/felix/phd2/dec_automl/nov18_training_time_random_model_bo'
#nametag = '/sample_instances_new2_all_constraints_training_time__'


#nov19_1week_good_configs-with_random_model_opt
#folder = '/home/felix/phd2/dec_automl/nov19_1week_good_configs-with_random_model_opt'
#nametag = '/ensemble_'


#folder = '/home/felix/phd2/dec_automl/nov20_random_model_good_random_trainingtime'
#nametag = '/good_random_trainingtime_'

#folder = '/home/felix/phd2/dec_automl/nov21_good_random_random_model_al_good_config'
#nametag = '/good_random_trainingtime_'

#nov22_random_model_pipeline_size
folder = '/home/felix/phd2/dec_automl/nov22_random_model_pipeline_size'
nametag = '/good_random_pipelinesize_'

#nov22-inference_time
folder = '/home/felix/phd2/dec_automl/nov22-inference_time'
nametag = '/good_random_inferencetime_'

#nov24_al_with_uniform_data_time_res
folder = '/home/felix/phd2/dec_automl/nov24_al_with_uniform_data_time_res'
nametag = '/ensemble_'


folder = '/home/felix/phd2/dec_automl/nov25_run_until1h_good_random'
nametag = '/ensemble_'


#folder = '/home/felix/phd2/dec_automl/nov29_random_good_configs_fairness'
#nametag = '/good_random_fairness_'

#folder = '/home/felix/phd2/dec_automl/nov30_2week_res'
#nametag = '/ensemble_'

#folder = '/home/felix/phd2/dec_automl/nov30_1week-1day_fairness'
#nametag = '/good_random_fairness_'

folder = '/home/felix/phd2/dec_automl/dez05_joined_model_searchtime'
nametag = '/ensemble_'

#folder = '/home/felix/phd2/dec_automl/dez05_joined_model_fairness'
#nametag = '/good_random_fairness_'

#dez06_joined_model_fairness_1week_1day_configs
#folder = '/home/felix/phd2/dec_automl/dez06_joined_model_fairness_1week_1day_configs'
#nametag = '/good_random_fairness_'

#dec06_1week_1week__constraints_joined_model_fairness
#folder = '/home/felix/phd2/dec_automl/dec06_1week_1week__constraints_joined_model_fairness'
#nametag = '/good_random_fairness_'

#dec10_rand5,10,al
#folder = '/home/felix/phd2/dec_automl/dec10_rand5,10,al'
#nametag = '/ensemble_'

#dec10_alternating_opt_search
folder = '/home/felix/phd2/dec_automl/dec10_alternating_opt_search'
nametag = '/ensemble_'

#folder = '/home/felix/phd2/dec_automl/dec10_rand5,al_search' #not weighted
#nametag = '/ensemble_'

#dec11_rand5_al_weighted_opt_searchtime
#folder = '/home/felix/phd2/dec_automl/dec11_rand5_al_weighted_opt_searchtime'
#nametag = '/ensemble_'

#dec11_alternating_weighted_opt_searchtime
#folder = '/home/felix/phd2/dec_automl/dec11_alternating_weighted_opt_searchtime'
#nametag = '/ensemble_'


#dec12_rand5_10_al_weighted_opt_searchtime
#folder = '/home/felix/phd2/dec_automl/dec12_rand5_10_al_weighted_opt_searchtime'
#nametag = '/ensemble_'

#dec12_al_rand5_weighted_fair
#folder = '/home/felix/phd2/dec_automl/dec12_al_rand5_weighted_fair'
#nametag = '/good_random_fairness_'

#dec12_alternating_opt_fair
#folder = '/home/felix/phd2/dec_automl/dec12_alternating_opt_fair'
#nametag = '/good_random_fairness_'

#dec12_alternating_opt_fair
#folder = '/home/felix/phd2/dec_automl/dec13_aternating_weihgted_opt_fairness'
#nametag = '/good_random_fairness_'

#folder = '/home/felix/phd2/dec_automl/dec14_al_rand5_smart_opt_searchtime'
#nametag = '/ensemble_'

#dec14_al_rand5_smart_opt_fair
#folder = '/home/felix/phd2/dec_automl/dec14_al_rand5_smart_opt_fair'
#nametag = '/good_random_fairness_'

#folder = '/home/felix/phd2/dec_automl/dec15_al_rand5_smart_opt_2weeks_configs'
#nametag = '/ensemble_'

#folder = '/home/felix/phd2/dec_automl/dec16_al_rand5_smart_opt_all_features_searchtime'
#nametag = '/ensemble_'

#folder = '/home/felix/phd2/dec_automl/dec17_al_rand5_smart_opt_all_est_searchtime'
#nametag = '/ensemble_'

#folder = '/home/felix/phd2/dec_automl/dec16_al_rand5_smart_opt_all_features_fairness'
#nametag = '/good_random_fairness_'

#folder = '/home/felix/phd2/dec_automl/dec19_al_rand5_smart_opt_inference_time'
#nametag = '/good_random_inferencetime_'

# dec19_al_rand5_smart_opt_trainingtime
#folder = '/home/felix/phd2/dec_automl/dec19_al_rand5_smart_opt_trainingtime'
#nametag = '/good_random_trainingtime_'

#dec20_al_rand5_smart_opt_pipelinesize
folder = '/home/felix/phd2/dec_automl/dec20_al_rand5_smart_opt_pipelinesize'
nametag = '/good_random_pipelinesize_'

folder = '/home/felix/phd2/dec_automl/dec21_alternating_smart_opt_pipelinesize'
nametag = '/good_random_pipelinesize_'

#dec21_alternating_smart_opt_pipelinesize_small_good_configs
folder = '/home/felix/phd2/dec_automl/dec21_alternating_smart_opt_pipelinesize_small_good_configs'
nametag = '/good_random_pipelinesize_'

#dec22_best_model_sofar_2weeks_configs_pipelinesize
#folder = '/home/felix/phd2/dec_automl/dec22_best_model_sofar_2weeks_configs_pipelinesize'
#nametag = '/good_random_pipelinesize_'

#folder = '/home/felix/phd2/dec_automl/dec23_rand5_pipelinesize'
#nametag = '/good_random_pipelinesize_'

#folder = '/home/felix/phd2/dec_automl/dec28_alternating_2weeks_pipelinesize_second'
#nametag = '/good_random_pipelinesize_'


#folder = '/home/felix/phd2/dec_automl/dec27_alternating_2weeks_smart_opt_pipelinesize'
#nametag = '/good_random_pipelinesize_'

folder = '/home/felix/phd2/dec_automl/dec28_alternating_2weeks_searchtime'
nametag = '/ensemble_'

#dec29_non_opt_alternating_pipelinesize
#folder = '/home/felix/phd2/dec_automl/dec29_non_opt_alternating_pipelinesize'
#nametag = '/good_random_pipelinesize_'

#dec30_alter_pipelinesize_r2_score
#folder = '/home/felix/phd2/dec_automl/dec30_alter_pipelinesize_r2_score'
#nametag = '/good_random_pipelinesize_'

#dec30_all_joined_smart_opt_pipelinesize
#folder = '/home/felix/phd2/dec_automl/dec30_all_joined_smart_opt_pipelinesize'
#nametag = '/good_random_pipelinesize_'

#dec30_alternating_absolute_pipelinesize
#folder = '/home/felix/phd2/dec_automl/dec30_alternating_absolute_pipelinesize'
#nametag = '/good_random_pipelinesize_'

#jan02_classification_model_pipelinesize
#folder = '/home/felix/phd2/dec_automl/jan02_classification_model_pipelinesize'
#nametag = '/good_random_pipelinesize_'

#best ever
#jan03_classification01_pipelinesize
folder = '/home/felix/phd2/dec_automl/jan03_classification01_pipelinesize'
nametag = '/good_random_pipelinesize_'

#folder = '/home/felix/phd2/dec_automl/jan03_best_sofar_1h'
#nametag = '/ensemble_'


#jan03_alternating_2w_training_time
folder = '/home/felix/phd2/dec_automl/jan03_alternating_2w_training_time'
nametag = '/good_random_trainingtime_'

#jan03_alternating_2w_inference_time
#folder = '/home/felix/phd2/dec_automl/jan03_alternating_2w_inference_time'
#nametag = '/good_random_inferencetime_'

#jan04_alternating_classification01_searchtime
#folder = '/home/felix/phd2/dec_automl/jan04_alternating_classification01_searchtime'
#nametag = '/ensemble_'

#jan04_alter_classification_training_time
folder = '/home/felix/phd2/dec_automl/jan04_alter_classification_training_time'
nametag = '/good_random_trainingtime_'

folder = '/home/felix/phd2/dec_automl/jan07_alternating_classification01_fairness_big_constraints'
nametag = '/good_random_fairness_'



print(glob.glob( folder + nametag + '*'))

#my_list = ['168797', '167200', '189862', '168794', '75097', '189905', '189866', '168796', '75193', '167185', '75105', '167168', '168793', '167152', '126025', '168798', '167083', '167149', '189908', '126026', '189865', '167181', '189906', '189874', '189872', '167190', '167104', '189909', '167161', '75127', '167201', '189861', '126029', '189860', '189873', '167184', '168795', '168792', '189871']

my_list = ['0', '1', '2', '3', '4']



#tag = 'static'
tag = 'dynamic'

data_all_constraint = {}
for constraint_i in [0,1,2,3,4]:
    data_all = []
    for datasetid in my_list:
        try:
            data = pickle.load(open(folder + nametag + datasetid + '.p', "rb"))
            data_all.append(data[tag][constraint_i])
        except:
            data_all.append(np.zeros(10))
    data_all_constraint[constraint_i] = data_all


print(data_all_constraint)

folder = '/home/felix/phd2/dec_automl/sep21_fair_results'
nametag = '/sample_instances_new2_all_constraints_fairness__'
tag = 'static'


data_all_constraint_compare = {}
for constraint_i in [0,1,2,3,4]:
    data_all = []
    for datasetid in my_list:
        try:
            data = pickle.load(open(folder + nametag + datasetid + '.p', "rb"))
            data_all.append(data[tag][constraint_i])
        except:
            data_all.append(np.zeros(10))
    data_all_constraint_compare[constraint_i] = data_all

print(data_all_constraint_compare)

constraint_ids = [0,1,2,3,4]
constraint_ids_compare = [0,1,2,3,4]

for constraint_indices in range(10):
    pop_dyn, pop_static = [], []
    for i in range(10000):
        list_dyn, list_static = [], []
        for d in range(len(my_list)):
            list_dyn.append(np.random.choice(data_all_constraint[constraint_ids[constraint_indices]][d]))
            list_static.append(np.random.choice(data_all_constraint_compare[constraint_ids_compare[constraint_indices]][d]))
        pop_dyn.append(np.mean(list_dyn))
        pop_static.append(np.mean(list_static))
        #print(str(np.mean(list_dyn)) + ' : ' + str(np.mean(list_static)))
    w, p = mannwhitneyu(pop_static, pop_dyn, alternative='less')
    #w, p = mannwhitneyu(pop_dyn, pop_static, alternative='less')
    print('w: ' + str(w) + ' p: ' + str(p))

