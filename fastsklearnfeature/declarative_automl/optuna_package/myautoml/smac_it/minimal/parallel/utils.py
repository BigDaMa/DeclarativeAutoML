from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.Space_GenerationTreeBalance import SpaceGenerator
import fastsklearnfeature.declarative_automl.optuna_package.myautoml.smac_it.minimal.parallel.my_global_vars as mp_global
import numpy as np
import copy

def query_model(trial, search_time=60, global_memory_constraint=None, privacy_constraint=None, training_time_constraint=None, inference_time_constraint=None, pipeline_size_constraint=None):
    gen = SpaceGenerator()
    space = gen.generate_params()
    #space.sample_parameters(trial)

    scaled_vector = np.zeros((1, len(mp_global.cs._hyperparameters)))
    scaled_vector[:] = np.nan

    for param_str, real_val in trial.params.items():
        scaled_val = mp_global.cs._hyperparameters[param_str]._inverse_transform(real_val)
        vector_id = mp_global.cs._hyperparameter_idx[param_str]

        scaled_vector[0, vector_id] = scaled_val

    param_dict = {}
    param_dict['global_search_time_constraint'] = search_time
    param_dict['use_evaluation_time_constraint'] = trial.suggest_categorical('use_evaluation_time_constraint', [False])

    evaluation_time = int(0.1 * search_time)
    if param_dict['use_evaluation_time_constraint']:
        evaluation_time = trial.suggest_int('global_evaluation_time_constraint', 10, search_time, log=False)
    param_dict['global_evaluation_time_constraint'] = evaluation_time
    trial.set_user_attr('evaluation_time', evaluation_time)

    # how many cvs should be used
    cv = 1
    number_of_cvs = 1
    hold_out_fraction = None
    param_dict['use_hold_out'] = trial.suggest_categorical('use_hold_out', [True])
    if param_dict['use_hold_out']:
        hold_out_fraction = trial.suggest_uniform('hold_out_fraction', 0, 1)
        param_dict['hold_out_fraction'] = hold_out_fraction
    else:
        cv = trial.suggest_int('global_cv', 2, 20, log=False)  # todo: calculate minimum number of splits based on y
        param_dict['global_cv'] = cv

        param_dict['use_multiple_cvs'] = trial.suggest_categorical('use_multiple_cvs', [True, False])

        number_of_cvs = 1
        if param_dict['use_multiple_cvs']:
            number_of_cvs = trial.suggest_int('global_number_cv', 2, 10, log=False)
        param_dict['global_number_cv'] = number_of_cvs

    sample_fraction = 1.0
    param_dict['use_sampling'] = trial.suggest_categorical('use_sampling', [False])
    if param_dict['use_sampling']:
        sample_fraction = trial.suggest_uniform('sample_fraction', 0, 1)
    param_dict['sample_fraction'] = sample_fraction

    #now the constraints
    if type(None) == type(global_memory_constraint):
        param_dict['use_search_memory_constraint'] = False
        param_dict['global_memory_constraint'] = 10
    else:
        param_dict['use_search_memory_constraint'] = True
        param_dict['global_memory_constraint'] = global_memory_constraint

    if type(None) == type(privacy_constraint):
        param_dict['use_privacy_constraint'] = False
        param_dict['privacy_constraint'] = 10
    else:
        param_dict['use_privacy_constraint'] = True
        param_dict['privacy_constraint'] = privacy_constraint

    if type(None) == type(training_time_constraint):
        param_dict['use_training_time_constraint'] = False
        param_dict['training_time_constraint'] = search_time
    else:
        param_dict['use_training_time_constraint'] = True
        param_dict['training_time_constraint'] = training_time_constraint

    if type(None) == type(inference_time_constraint):
        param_dict['use_inference_time_constraint'] = False
        param_dict['inference_time_constraint'] = 60
    else:
        param_dict['use_inference_time_constraint'] = True
        param_dict['inference_time_constraint'] = inference_time_constraint

    if type(None) == type(pipeline_size_constraint):
        param_dict['use_pipeline_size_constraint'] = False
        param_dict['pipeline_size_constraint'] = 350000000
    else:
        param_dict['use_pipeline_size_constraint'] = True
        param_dict['pipeline_size_constraint'] = pipeline_size_constraint

    for param_str, real_val in param_dict.items():
        if param_str in mp_global.cs._hyperparameter_idx:
            print(param_str)
            scaled_val = mp_global.cs._hyperparameters[param_str]._inverse_transform(real_val)
            vector_id = mp_global.cs._hyperparameter_idx[param_str]

            scaled_vector[0, vector_id] = scaled_val

    success = mp_global.new_model._predict(X=scaled_vector, metafeatures=mp_global.metafeature_values)[0][0][0]


    if trial.number == 0 or success < mp_global.study_prune.best_trial.value:
        trial.set_user_attr('space', copy.deepcopy(space))

    return success