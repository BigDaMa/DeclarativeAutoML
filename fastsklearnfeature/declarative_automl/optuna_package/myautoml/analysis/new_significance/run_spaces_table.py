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

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/sep26_searchtime_constraint_model'
#nametag = 'ensemble_'

#folder = '/home/neutatz/phd2/decAutoML2weeks_compare2default/okt10_dynamic'
#nametag = 'ensemble_'

folder = '/home/felix/phd2/dec_automl/nov25_run_until1h_good_random'
nametag = 'ensemble_'

folder = '/home/felix/phd2/dec_automl/jan06_alternating_classification01_pipelinesize_big_constraints'
nametag = 'good_random_pipelinesize_'

folder = '/home/felix/phd2/dec_automl/jan06_alternating_classification01_searchtime_big_constraints'
nametag = 'ensemble_'


print(glob.glob( folder + nametag + '*'))

my_list = ['168797', '167200', '189862', '168794', '75097', '189905', '189866', '168796', '75193', '167185', '75105', '167168', '168793', '167152', '126025', '168798', '167083', '167149', '189908', '126026', '189865', '167181', '189906', '189874', '189872', '167190', '167104', '189909', '167161', '75127', '167201', '189861', '126029', '189860', '189873', '167184', '168795', '168792', '189871']

data_means = {}

space_len = {}

classifier_list = [
    'classifier##AdaBoostClassifierOptuna()',
    'classifier##BernoulliNBOptuna()',
    'classifier##DecisionTreeClassifierOptuna()',
    'classifier##ExtraTreesClassifierOptuna()',
    'classifier##GaussianNBOptuna()',
    'classifier##HistGradientBoostingClassifierOptuna()',
    'classifier##KNeighborsClassifierOptuna()',
    'classifier##LinearDiscriminantAnalysisOptuna()',
    'classifier##LinearSVCOptuna()',
    'classifier##MLPClassifierOptuna()',
    'classifier##MultinomialNBOptuna()',
    'classifier##PassiveAggressiveOptuna()',
    'classifier##QuadraticDiscriminantAnalysisOptuna()',
    'classifier##RandomForestClassifierOptuna()',
    'classifier##SGDClassifierOptuna()',
    'classifier##SVCOptuna()'
]

classifier_names = [
    'AdaBoost',
    'BernoulliNB',
    'DT',
    'ExtraTrees',
    'GaussianNB',
    'HistGradientBoost',
    'KNN',
    'LDA',
    'LinearSVC',
    'MLP',
    'MultinomialNB',
    'PA',
    'QDA',
    'RF',
    'SGD',
    'SVC'
]

latex_str = '\\begin{tabular}{@{}l'
for i in range(len(classifier_names)):
    latex_str += 'c'
latex_str += 'c@{}}\\toprule\nConstraint & '
for i in range(len(classifier_names)):
    latex_str += classifier_names[i] + ' & '
latex_str += '|classifiers| \\\\ \\midrule\n'

try:

    for constraint_i in range(6):
        latex_str += str(constraint_i) + ' & '
        data_all = []
        data_means[constraint_i] = {}

        classifier_count = np.zeros(len(classifier_list))
        selected_classifiers =[]

        for datasetid in my_list:
            data = pickle.load(open(folder + '/' + nametag + datasetid + '.p', "rb"))

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
            #print(best_str)
            current_count = 0
            for classifier_str_i in range(len(classifier_list)):
                classifier_str = classifier_list[classifier_str_i]
                if classifier_str in str(best_str):
                    classifier_count[classifier_str_i] += 1
                    current_count += 1
            selected_classifiers.append(current_count)
        print(classifier_list)
        fractions = (classifier_count / float(len(my_list))).tolist()
        for i in range(len(classifier_names)):
            latex_str += "{:.2f}".format(fractions[i]) + ' & '
        latex_str +=  "{:.2f}".format(np.mean(selected_classifiers)) + '\\\\ \n'

        print((classifier_count / float(len(my_list))).tolist())
        print('Number classifiers: ' + str(np.mean(selected_classifiers)))
        print('\n')

except:
    pass


print(data_means)

print(latex_str)