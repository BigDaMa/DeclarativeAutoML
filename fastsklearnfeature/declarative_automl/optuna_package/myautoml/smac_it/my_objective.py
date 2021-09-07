from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.MyAutoMLProcessClassBalanceDict import MyAutoML
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import get_data
import numpy as np
from sklearn.metrics import make_scorer
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.Space_GenerationTreeBalance import SpaceGenerator
from anytree import RenderTree
from sklearn.metrics import f1_score


def run_AutoML(x):
    my_scorer = make_scorer(f1_score)
    repetitions = 1

    # which hyperparameters to use
    gen = SpaceGenerator()
    space = gen.generate_params()

    #space.sample_parameters_SMAC(x)

    # make this a hyperparameter
    search_time = x['global_search_time_constraint']

    evaluation_time = search_time
    if 'use_evaluation_time_constraint' in x and x['use_evaluation_time_constraint']:
        evaluation_time = x['global_evaluation_time_constraint']

    memory_limit = 10
    if 'use_search_memory_constraint' in x and x['use_search_memory_constraint']:
        memory_limit = x['global_memory_constraint']

    privacy_limit = None
    if 'use_privacy_constraint' in x and x['use_privacy_constraint']:
        privacy_limit = x['privacy_constraint']

    training_time_limit = search_time
    if 'use_training_time_constraint' in x and x['use_training_time_constraint']:
        training_time_limit = x['training_time_constraint']

    inference_time_limit = 60
    if 'use_inference_time_constraint' in x and x['use_inference_time_constraint']:
        inference_time_limit = x['inference_time_constraint']

    pipeline_size_limit = 350000000
    if 'use_pipeline_size_constraint' in x and x['use_pipeline_size_constraint']:
        pipeline_size_limit = x['pipeline_size_constraint']

    cv = 1
    number_of_cvs = 1
    hold_out_fraction = 0.0
    if x['use_hold_out'] == False:
        cv = x['global_cv']
        if x['use_multiple_cvs']:
            number_of_cvs = x['global_number_cv']
    else:
        hold_out_fraction = x['hold_out_fraction']

    sample_fraction = 1.0
    if 'use_sampling' in x and x['use_sampling']:
        sample_fraction = x['sample_fraction']

    dataset_id = x['dataset_id']  # get same random seed

    for pre, _, node in RenderTree(space.parameter_tree):
        if node.status == True:
            print("%s%s" % (pre, node.name))

    my_random_seed = 41

    try:
        X_train, X_test, y_train, y_test, categorical_indicator, attribute_names = get_data(dataset_id,
                                                                                            randomstate=my_random_seed)

        '''
        random_feature_selection = x['random_feature_selection']
        rand_feature_ids = np.arange(X_train.shape[1])
        np.random.seed(my_random_seed)
        np.random.shuffle(rand_feature_ids)
        number_of_sampled_features = int(X_train.shape[1] * random_feature_selection)

        X_train = X_train[:, rand_feature_ids[0:number_of_sampled_features]]
        X_test = X_test[:, rand_feature_ids[0:number_of_sampled_features]]
        categorical_indicator = np.array(categorical_indicator)[rand_feature_ids[0:number_of_sampled_features]]
        attribute_names = np.array(attribute_names)[rand_feature_ids[0:number_of_sampled_features]]
        '''

        # sampling with class imbalancing
        if 'unbalance_data' in x:
            class_labels = np.unique(y_train)

            ids_class0 = np.array((y_train == class_labels[0]).nonzero()[0])
            ids_class1 = np.array((y_train == class_labels[1]).nonzero()[0])

            np.random.seed(my_random_seed)
            np.random.shuffle(ids_class0)
            np.random.seed(my_random_seed)
            np.random.shuffle(ids_class1)

            if x['unbalance_data']:
                fraction_ids_class0 = x['fraction_ids_class0']
                fraction_ids_class1 = x['fraction_ids_class1']
            else:
                sampling_factor_train_only = x['sampling_factor_train_only']
                fraction_ids_class0 = sampling_factor_train_only
                fraction_ids_class1 = sampling_factor_train_only

            number_class0 = int(fraction_ids_class0 * len(ids_class0))
            number_class1 = int(fraction_ids_class1 * len(ids_class1))

            all_sampled_training_ids = []
            all_sampled_training_ids.extend(ids_class0[0:number_class0])
            all_sampled_training_ids.extend(ids_class1[0:number_class1])

            X_train = X_train[all_sampled_training_ids, :]
            y_train = y_train[all_sampled_training_ids]


    except:
        return 0.0

    dynamic_params = []
    for random_i in range(repetitions):
        search = MyAutoML(cv=cv,
                          number_of_cvs=number_of_cvs,
                          n_jobs=1,
                          evaluation_budget=evaluation_time,
                          time_search_budget=search_time,
                          space=space,
                          main_memory_budget_gb=memory_limit,
                          differential_privacy_epsilon=privacy_limit,
                          hold_out_fraction=hold_out_fraction,
                          sample_fraction=sample_fraction,
                          training_time_limit=training_time_limit,
                          inference_time_limit=inference_time_limit,
                          pipeline_size_limit=pipeline_size_limit)

        test_score = 0.0
        try:
            search.fit(X_train, y_train, categorical_indicator=categorical_indicator, scorer=my_scorer)

            best_pipeline = search.get_best_pipeline()
            if type(best_pipeline) != type(None):
                test_score = my_scorer(search.get_best_pipeline(), X_test, y_test)
        except:
            pass
        dynamic_params.append(test_score)

    dynamic_values = np.array(dynamic_params)

    if np.sum(dynamic_values) == 0:
        return 0.0

    static_params = []
    for random_i in range(repetitions):
        # default params
        gen_new = SpaceGenerator()
        space_new = gen_new.generate_params()
        for pre, _, node in RenderTree(space_new.parameter_tree):
            if node.status == True:
                print("%s%s" % (pre, node.name))

        search_static = MyAutoML(n_jobs=1,
                                 time_search_budget=search_time,
                                 space=space_new,
                                 evaluation_budget=int(0.1 * search_time),
                                 main_memory_budget_gb=memory_limit,
                                 differential_privacy_epsilon=privacy_limit,
                                 hold_out_fraction=0.33,
                                 training_time_limit=training_time_limit,
                                 inference_time_limit=inference_time_limit,
                                 pipeline_size_limit=pipeline_size_limit
                                 )

        try:
            best_result = search_static.fit(X_train, y_train, categorical_indicator=categorical_indicator,
                                            scorer=my_scorer)
            test_score_default = my_scorer(search_static.get_best_pipeline(), X_test, y_test)
        except:
            test_score_default = 0.0
        static_params.append(test_score_default)

    static_values = np.array(static_params)

    dynamic_values.sort()
    static_values.sort()

    frequency = np.sum(dynamic_values > static_values) / 5.0

    return -1.0 * frequency