import numpy as np
import matplotlib.pyplot as plt
import pickle
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import get_feature_names
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

# Fixing random state for reproducibility
np.random.seed(19680801)


#X = pickle.load(open('/home/felix/phd2/dec_automl/okt31_random_2weeks/felix_X_compare_scaled.p', "rb"))
X = pickle.load(open('/home/felix/phd2/dec_automl/okt31_al_2weeks/felix_X_compare_scaled.p', "rb"))



# the histogram of the data
n, bins, patches = plt.hist(X[:, feature_names.index('global_search_time_constraint')], 50, density=True, facecolor='g', alpha=0.75)


plt.xlabel('Search Time')
plt.ylabel('Probability')
#plt.xlim(40, 160)
#plt.ylim(0, 0.03)
plt.grid(True)
plt.show()