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

folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/aug21_autosklearn2_describe'
nametag = 'autosklearn_'

print(glob.glob( folder + nametag + '*'))

my_list = ['168797', '167200', '189862', '168794', '75097', '189905', '189866', '168796', '75193', '167185', '75105', '167168', '168793', '167152', '126025', '168798', '167083', '167149', '189908', '126026', '189865', '167181', '189906', '189874', '189872', '167190', '167104', '189909', '167161', '75127', '167201', '189861', '126029', '189860', '189873', '167184', '168795', '168792', '189871']

data_means = {}

space_len = {}

for constraint_i in range(4):
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
            if datasetid == '75193':
                print(my_run.space)
        #print(datasetid + ': ' + str(best_str))
    print('###########################')

