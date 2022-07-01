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

folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/dez3_pipeline_size_m1'
nametag = '/sample_instances_new2_all_constraints_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/jun30_autosklearn_inference_time'
#nametag = '/autosklearn_inference_time_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/dez3_inference_time'
#nametag = '/sample_instances_new2_all_constraints_inference_time__'

print(glob.glob( folder + nametag + '*'))

my_list = ['168797', '167200', '189862', '168794', '75097', '189905', '189866', '168796', '75193', '167185', '75105', '167168', '168793', '167152', '126025', '168798', '167083', '167149', '189908', '126026', '189865', '167181', '189906', '189874', '189872', '167190', '167104', '189909', '167161', '75127', '167201', '189861', '126029', '189860', '189873', '167184', '168795', '168792', '189871']

data_means = {}

tag = 'static'
#tag = 'dynamic'

for constraint_i in range(10):
    data_all = []
    data_means[constraint_i] = {}

    for datasetid in my_list:
        data = pickle.load(open(folder + nametag + datasetid + '.p', "rb"))
        data_all.append(data[tag][constraint_i])
        data_means[constraint_i][datasetid] = np.mean(data[tag][constraint_i])

    pop_dyn, pop_static = [], []
    for i in range(10000):
        list_dyn, list_static = [], []
        for d in range(len(my_list)):
            list_dyn.append(np.random.choice(data_all[d]))
            #list_static.append()
        pop_dyn.append(np.mean(list_dyn))
        #pop_static.append(np.mean(list_static))
        #w, p = scipy.stats.mannwhitneyu(pop_dyn, pop_static, alternative='less')
    print(str(np.mean(pop_dyn)) + ' +- ' + str(np.std(pop_dyn)))

print(data_means)