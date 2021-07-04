import matplotlib.pyplot as plt
import numpy as np
import pickle

#cv = np.array(pickle.load(open('/home/felix/phd2/picture_progress/al_only/felix_cv_compare.p', "rb")))
#cv = np.array(pickle.load(open('/tmp/felix_cv_compare_scaled.p', "rb")))
#cv = np.array(pickle.load(open('/home/neutatz/phd2/picture_progress/al_only/felix_cv_success.p', "rb")))

#cv = np.array(pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/machine 4_squared/felix_cv_compare_scaled.p', "rb")))
#cv = np.array(pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/machine 2/felix_cv_compare_scaled.p', "rb")))

#cv = np.array(pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/1 week/machine4 cost_sensitive/felix_cv_compare_scaled.p', "rb")))
#cv = np.array(pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/1 week/more weeks cost sensitive - machine4/felix_cv_compare_scaled.p', "rb")))

#cv = np.array(pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/3 weeks/more_weeks_scaled_compare_5min_machine2/felix_cv_compare_scaled.p', "rb")))

#constraintsALCompare_MaxScaling_Cost
#X = np.array(pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/1 week/more weeks cost sensitive - machine4/felix_X_compare_scaled.p', "rb")))

#constraintsALCompare_MaxScaling_Cost
#X = np.array(pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/cost_sensitive_experiments/with_quantile_model_constraintsALCompare_MaxScaling_Cost/felix_X_compare_scaled.p', "rb")))


#X = np.array(pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/3 weeks/more_weeks_scaled_compare_5min_machine2//felix_X_compare_scaled.p', "rb")))


X = np.array(pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/cost_sensitive_experiments/Divison/felix_X_compare_scaled.p', "rb")))

#X = np.array(pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/cost_sensitive_experiments/fraction/felix_X_compare_scaled.p', "rb")))


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
                           'pipeline_size_constraint']

from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import get_feature_names
_, feature_names = get_feature_names(my_list_constraints)

print(feature_names)

id_time = feature_names.index('global_search_time_constraint')

x = range(len(X))
y = X[:,id_time].flatten()

plt.scatter(x, y)

z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x, p(x),"r--")

plt.title('Search Time Constraint Trend over Time')
plt.axhline(y=5*60)
plt.show()

print(np.max(y))

'''

cv = np.array(pickle.load(open('/home/felix/phd2/picture_progress/al_only/felix_cv_success.p', "rb")))
plt.plot(range(len(cv)), cv)
plt.axhline(y=np.nanmax(cv))
plt.ylim(0, 1)
plt.title('success prediction')
plt.ylabel('Cross-Validation R2 score')
plt.xlabel('Active Learning Iterations')
plt.show()
print(cv)


cv = np.array(pickle.load(open('/home/felix/phd2/picture_progress/weights/felix_cv_weights.p', "rb")))
plt.plot(range(len(cv)), cv)
plt.axhline(y=np.nanmax(cv))
#plt.ylim(0, 1)
plt.title('models weights')
plt.ylabel('Cross-Validation R2 score')
plt.xlabel('Active Learning Iterations')
plt.show()
print(cv)

'''