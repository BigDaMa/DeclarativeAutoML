import pickle
import glob
import  numpy as np
import openml


#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/autosklearn_oct15'
#nametag = '/autosklearn_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/tpot'
#nametag = '/tpot_'


#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/dez31_1week_res_no_fallback'
#nametag = '/sample_instances_new2_all_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/dez31_1week_res_no_fallback'
#nametag = 'sample_instances_new2_all_'


#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/mai26_3weeks'
#nametag = '/sample_instances_new2_all_'

folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/okt10_dynamic'
nametag = 'ensemble_'

folder = '/home/felix/phd2/dec_automl/jan06_alternating_classification01_searchtime_big_constraints'
nametag = 'ensemble_'

print(glob.glob( folder + nametag + '*'))

my_list = ['168797', '167200', '189862', '168794', '75097', '189905', '189866', '168796', '75193', '167185', '75105', '167168', '168793', '167152', '126025', '168798', '167083', '167149', '189908', '126026', '189865', '167181', '189906', '189874', '189872', '167190', '167104', '189909', '167161', '75127', '167201', '189861', '126029', '189860', '189873', '167184', '168795', '168792', '189871']

data_means = {}

space_len = {}

dict_data2space_size = {}

for constraint_i in range(6):
    data_all = []
    data_means[constraint_i] = {}

    for datasetid in my_list:
        data = pickle.load(open(folder + '/' + nametag + datasetid + '.p', "rb"))
        data_all.append(data['dynamic'][constraint_i])
        data_means[constraint_i][datasetid] = np.mean(data['dynamic'][constraint_i])

        data_space = pickle.load(open(folder + '/' + 'log_' + nametag + datasetid + '.p', "rb"))
        best_run_space = None
        best_run_score = 0.0
        best_str = None
        for my_run in data_space['dynamic'][constraint_i].runs:
            #print(data_space['dynamic'][constraint_i].runs)
            if my_run.test_score > best_run_score:
                best_run_score = my_run.test_score
                best_run_space = my_run.params
                best_str = my_run.space_str
        #print(datasetid + ': ' + str(best_run_space))

        #dataset = openml.datasets.get_dataset(dataset_id=datasetid)
        #print(datasetid +  ': ' + str(len(best_run_space)))
        #print(datasetid + ': ' + str(best_run_space['sample_fraction']))
        #if datasetid == '168796':
        #if datasetid == '168796':
        #if datasetid == '189908':
        if True:
            print('----------------------------')
            print('constraint: ' + str(constraint_i))
            print(best_run_space)
            hold_out_fraction = -1
            try:
                hold_out_fraction = best_run_space['hold_out_fraction']
                print('validation_fraction: ' + str(hold_out_fraction))
            except:
                pass
            print(best_str)
            print('----------------------------\n\n\n')
            if not datasetid in dict_data2space_size:
                dict_data2space_size[datasetid] = []
            else:
                #dict_data2space_size[datasetid].append(len(str(best_str)))
                dict_data2space_size[datasetid].append(hold_out_fraction)
            print(dict_data2space_size)

        ##print best space

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

print(dict_data2space_size)