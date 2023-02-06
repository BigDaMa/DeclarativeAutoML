import pickle
import glob
import  numpy as np
import openml
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import get_feature_names

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

#folder = '/home/felix/phd2/dec_automl/nov25_run_until1h_good_random'
#nametag = 'ensemble_'

#folder = '/home/felix/phd2/dec_automl/jan06_alternating_classification01_pipelinesize_big_constraints'
#nametag = 'good_random_pipelinesize_'

folder = '/home/felix/phd2/dec_automl/jan06_alternating_classification01_searchtime_big_constraints'
nametag = 'ensemble_'

#jan07_alternating_classification01_trainingtime_big_constraints
#folder = '/home/felix/phd2/dec_automl/jan07_alternating_classification01_trainingtime_big_constraints'
#nametag = 'good_random_trainingtime_'

#jan07_alternating_classification01_inferencetime_big_constraints
#folder = '/home/felix/phd2/dec_automl/jan07_alternating_classification01_inferencetime_big_constraints'
#nametag = 'good_random_inferencetime_'

#jan07_alternating_classification01_fairness_big_constraints
#folder = '/home/felix/phd2/dec_automl/jan07_alternating_classification01_fairness_big_constraints'
#nametag = 'good_random_fairness_'

#folder = '/home/felix/phd2/dec_automl/jan17_al_smallest_searchtime'
#nametag = 'ensemble_'

#folder = '/home/felix/phd2/dec_automl/jan20_random_fractions_2,10,20'
#nametag = 'random_configs_fraction_'



my_list_constraints = ['global_search_time_constraint',
                           'global_evaluation_time_constraint',
                           'global_memory_constraint',
                           'global_cv',
                           'global_number_cv',
                           'privacy',
                           'hold_out_fraction',
                           'sample_fraction',
                           'training_time_constraint',
                           'inference_time_constraint',
                           'pipeline_size_constraint',
                           'fairness_constraint',
                           'use_ensemble',
                           'use_incremental_data',
                           'shuffle_validation'
                       ]

feature_names, feature_names_new = get_feature_names(my_list_constraints)

shuffle_index = feature_names_new.index('shuffle_validation')
print('shuffle index: ' + str(shuffle_index))

hold_out_fraction_index = feature_names_new.index('hold_out_fraction')
incremental_index = feature_names_new.index('use_incremental_data')
ensemble_index = feature_names_new.index('use_ensemble')

print(glob.glob( folder + nametag + '*'))

my_list = ['168797', '167200', '189862', '168794', '75097', '189905', '189866', '168796', '75193', '167185', '75105', '167168', '168793', '167152', '126025', '168798', '167083', '167149', '189908', '126026', '189865', '167181', '189906', '189874', '189872', '167190', '167104', '189909', '167161', '75127', '167201', '189861', '126029', '189860', '189873', '167184', '168795', '168792', '189871']

#my_list = ['0', '1', '2', '3', '4']

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


hold_out_per_constraint = {}
use_incremental_data_constraint = {}
use_shuffle = {}
use_ensemble = {}
class_augmentation = {}
number_parameters = {}
training_sampling_factor = {}

min_length = np.inf
min_dataset = None

try:

    for constraint_i in range(6):
        latex_str += str(constraint_i) + ' & '
        data_all = []
        data_means[constraint_i] = {}

        classifier_count = np.zeros(len(classifier_list))
        selected_classifiers =[]

        hold_out_fraction_all = []
        use_incremental_data_constraint_all = []
        use_shuffle_all = []
        use_ensemble_all = []
        class_augmentation_all = []
        number_parameters_all = []
        training_sampling_factor_all = []

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
                    best_str = str(my_run.space_str)

            if True:
            #if constraint_i == 2:
                print('################################################################################')
                print(str(datasetid) + '( ' + str(str(best_str).count('\n')) + ') : ' + str(best_str))
                try:
                    #print(best_run_space)
                    print('constraint: ' + str(constraint_i))
                    print('shuffle: ' + str(best_run_space[shuffle_index] == 1.0))

                    hold_out_fraction_all.append(best_run_space[hold_out_fraction_index])
                    print('holdout: ' + str(hold_out_fraction_all[-1]))
                    use_incremental_data_constraint_all.append(best_run_space[incremental_index])
                    print('incremental: ' + str(use_incremental_data_constraint_all[-1] == 1.0))

                    use_shuffle_all.append(best_run_space[shuffle_index])
                    use_ensemble_all.append(best_run_space[ensemble_index])
                    print('ensemble: ' + str(use_ensemble_all[-1] == 1.0))

                    if best_str.count('\n') > 4:
                        number_parameters_all.append(best_str.count('\n'))

                    if number_parameters_all[-1] < min_length:
                        min_length = number_parameters_all[-1]
                        min_dataset = datasetid

                    class_augmentation_all.append( 'augmentation##RandomOverSamplerOptuna()' in best_str or 'augmentation##ADASYNOptuna()' in best_str or 'augmentation##SMOTEOptuna()' in best_str)
                    training_sampling_factor_all.append('training_sampling_factor' in best_str)

                except:
                    pass
                print('################################################################################\n\n')
                current_count = 0
                for classifier_str_i in range(len(classifier_list)):
                    classifier_str = classifier_list[classifier_str_i]
                    if classifier_str in str(best_str):
                        classifier_count[classifier_str_i] += 1
                        current_count += 1
                selected_classifiers.append(current_count)

                hold_out_per_constraint[constraint_i] = np.average(hold_out_fraction_all)
                use_incremental_data_constraint[constraint_i] = np.average(use_incremental_data_constraint_all)
                use_shuffle[constraint_i] = np.average(use_shuffle_all)
                use_ensemble[constraint_i] = np.average(use_ensemble_all)
                class_augmentation[constraint_i] = np.average(class_augmentation_all)
                number_parameters[constraint_i] = np.average(number_parameters_all)
                training_sampling_factor[constraint_i] = np.average(training_sampling_factor_all)
        print(classifier_list)
        fractions = (classifier_count / float(len(my_list))).tolist()
        for i in range(len(classifier_names)):
            latex_str += "{:.2f}".format(fractions[i]) + ' & '
        latex_str +=  "{:.2f}".format(np.mean(selected_classifiers)) + '\\\\ \n'

        print((classifier_count / float(len(my_list))).tolist())
        print('Number classifiers: ' + str(np.mean(selected_classifiers)))
        print('\n')

        print('holdout all: ' + str(hold_out_per_constraint))
        print('incremental all: ' + str(use_incremental_data_constraint))
        print('shuffle all: ' + str(use_shuffle))
        print('ensemble all: ' + str(use_ensemble))
        print('class augmentation: ' + str(class_augmentation))
        print('training_sampling_factor: ' + str(training_sampling_factor))
        print('number params: ' + str(number_parameters))

        print('min: ' + str(min_dataset) + ' -> ' + str(min_length))

except:
    pass


print(data_means)

print(latex_str)