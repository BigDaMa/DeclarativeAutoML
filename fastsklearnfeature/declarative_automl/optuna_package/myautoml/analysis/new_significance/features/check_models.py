import pickle
#model_success = pickle.load(open('/home/felix/phd2/dec_automl/dec25_2weeks_alternating/felix_X_compare_scaled.p', "rb"))


for day in range(1,15):
    model_success = pickle.load(open('/home/felix/phd2/dec_automl/dec09_al_sampling_daywise_models_classification/my_great_model_compare_scaled'+ str(day) +'_0.1.p', "rb"))


    print(model_success.classes_)