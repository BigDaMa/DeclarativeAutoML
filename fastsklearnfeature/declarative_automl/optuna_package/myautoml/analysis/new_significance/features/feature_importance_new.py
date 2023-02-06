from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import plot_most_important_features
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import pickle
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import get_feature_names
import openml

openml.config.apikey = '4384bd56dad8c3d2c0f6630c52ef5567'
openml.config.cache_directory = '/home/neutatz/phd2/cache_openml'


#model_success = pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/sep20_constraint_model/my_great_model_compare_scaled.p', "rb"))

#classification
model_success = pickle.load(open('/home/felix/phd2/dec_automl/jan01_alter_smart_opt_classification/my_great_model_compare_scaled14_0.1.p', "rb"))

#regression
#model_success = pickle.load(open('/home/felix/phd2/dec_automl/dec27_alternating_2weeks_smart_opt/my_great_model_compare_scaled.p', "rb"))


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

_, feature_names = get_feature_names(my_list_constraints)

plot_most_important_features(model_success, feature_names, k=len(feature_names))