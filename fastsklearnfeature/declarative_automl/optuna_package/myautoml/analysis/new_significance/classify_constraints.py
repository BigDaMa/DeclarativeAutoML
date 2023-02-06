from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import plot_most_important_features
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import pickle
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import get_feature_names
import openml
from sklearn.ensemble import RandomForestClassifier
import numpy as np

openml.config.apikey = '4384bd56dad8c3d2c0f6630c52ef5567'
openml.config.cache_directory = '/home/neutatz/phd2/cache_openml'


#model_success = pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/sep20_constraint_model/my_great_model_compare_scaled.p', "rb"))

#classification
#training_time = pickle.load(open('/home/felix/phd2/dec_automl/jan29_features/trainingtime.p', "rb"))
#training_time = pickle.load(open('/home/felix/phd2/dec_automl/jan29_features/pipelinesize.p', "rb"))
#training_time = pickle.load(open('/home/felix/phd2/dec_automl/jan29_features/inferencetime.p', "rb"))
training_time = pickle.load(open('/home/felix/phd2/dec_automl/jan29_features/fairness.p', "rb"))

y = [1 for i in range(len(training_time))]
#inference_time = pickle.load(open('/home/felix/phd2/dec_automl/jan29_features/inferencetime.p', "rb"))
#inference_time = pickle.load(open('/home/felix/phd2/dec_automl/jan29_features/pipelinesize.p', "rb"))
#searchtime
inference_time = pickle.load(open('/home/felix/phd2/dec_automl/jan29_features/searchtime.p', "rb"))

y.extend([0 for i in range(len(inference_time))])

training_time.extend(inference_time)

print(np.vstack(training_time).shape)
#print(training_time)

X = np.vstack(training_time)

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
assert X.shape[1] == len(feature_names)


X[:, feature_names.index('training_time_constraint')] = 0
X[:, feature_names.index('inference_time_constraint')] = 0
X[:, feature_names.index('pipeline_size_constraint')] = 0
X[:, feature_names.index('fairness_constraint')] = 0

model_success = RandomForestClassifier(n_estimators=1000, n_jobs=-1).fit(X, y)

plot_most_important_features(model_success, feature_names, k=len(feature_names))