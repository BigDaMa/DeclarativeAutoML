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

#cv = np.array(pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/more_weeks/machine2_5min/felix_cv_compare_scaled.p', "rb")))

#cv = np.array(pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/cost_sensitive_experiments/machine1_model_constraintsALCompare_MaxScaling_Cost_Optimal/felix_cv_compare_scaled.p', "rb")))

cv = np.array(pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/cost_sensitive_experiments/Divison/felix_acquisition function value_scaled.p', "rb")))


plt.scatter(range(len(cv)), cv)
#plt.ylim(0, 1)
plt.axhline(y=np.nanmax(cv))
plt.title('comparison prediction')
plt.ylabel('Cross-Validation R2 score')
plt.xlabel('Active Learning Iterations')
plt.show()
print(cv)

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