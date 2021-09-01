from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.Space_GenerationTreeBalance import SpaceGenerator
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition

from smac.configspace import ConfigurationSpace

def get_space(total_search_time, my_openml_datasets):
    cs = ConfigurationSpace()
    cs.add_hyperparameter(UniformIntegerHyperparameter('global_search_time_constraint', 10, total_search_time, default_value=10))
    cs.add_hyperparameter(CategoricalHyperparameter('dataset_id', my_openml_datasets, default_value=my_openml_datasets[0]))

    use_evaluation_time_constraint_p = CategoricalHyperparameter('use_evaluation_time_constraint', [True, False],
                                                                 default_value=False)
    global_evaluation_time_constraint_p = UniformIntegerHyperparameter('global_evaluation_time_constraint', 10,
                                                                       total_search_time,
                                                                       default_value=total_search_time)
    cs.add_hyperparameters([use_evaluation_time_constraint_p, global_evaluation_time_constraint_p])
    cs.add_condition(EqualsCondition(global_evaluation_time_constraint_p, use_evaluation_time_constraint_p, True))

    use_search_memory_constraint_p = CategoricalHyperparameter('use_search_memory_constraint', [True, False], default_value=False)
    global_memory_constraint_p = UniformFloatHyperparameter('global_memory_constraint', 0.00000000000001, 10,
                                                            default_value=10, log=True)
    cs.add_hyperparameters([global_memory_constraint_p, use_search_memory_constraint_p])
    cs.add_condition(EqualsCondition(global_memory_constraint_p, use_search_memory_constraint_p, True))

    use_privacy_constraint_p = CategoricalHyperparameter('use_privacy_constraint', [True, False], default_value=False)
    privacy_constraint_p = UniformFloatHyperparameter('privacy_constraint', 0.0001, 10, default_value=10, log=True)
    cs.add_hyperparameters([privacy_constraint_p, use_privacy_constraint_p])
    cs.add_condition(EqualsCondition(privacy_constraint_p, use_privacy_constraint_p, True))

    use_training_time_constraint_p = CategoricalHyperparameter('use_training_time_constraint', [True, False], default_value=False)
    training_time_constraint_p = UniformFloatHyperparameter('training_time_constraint', 0.005, total_search_time,
                                                            default_value=total_search_time, log=True)
    cs.add_hyperparameters([training_time_constraint_p, use_training_time_constraint_p])
    cs.add_condition(EqualsCondition(training_time_constraint_p, use_training_time_constraint_p, True))

    use_inference_time_constraint_p = CategoricalHyperparameter('use_inference_time_constraint', [True, False], default_value=False)
    inference_time_constraint_p = UniformFloatHyperparameter('inference_time_constraint', 0.0004, 60, default_value=60,
                                                             log=True)
    cs.add_hyperparameters([inference_time_constraint_p, use_inference_time_constraint_p])
    cs.add_condition(EqualsCondition(inference_time_constraint_p, use_inference_time_constraint_p, True))

    use_pipeline_size_constraint_p = CategoricalHyperparameter('use_pipeline_size_constraint', [True, False],
                                                               default_value=False)
    pipeline_size_constraint_p = UniformIntegerHyperparameter('pipeline_size_constraint', 2000, 350000000,
                                                              default_value=350000000, log=True)
    cs.add_hyperparameters([pipeline_size_constraint_p, use_pipeline_size_constraint_p])
    cs.add_condition(EqualsCondition(pipeline_size_constraint_p, use_pipeline_size_constraint_p, True))

    use_sampling_p = CategoricalHyperparameter('use_sampling', [True, False], default_value=False)
    sample_fraction_p = UniformFloatHyperparameter('sample_fraction', 0, 1, default_value=1)
    cs.add_hyperparameters([sample_fraction_p, use_sampling_p])
    cs.add_condition(EqualsCondition(sample_fraction_p, use_sampling_p, True))

    use_hold_out_p = CategoricalHyperparameter('use_hold_out', [True, False], default_value=True)
    hold_out_fraction_p = UniformFloatHyperparameter('hold_out_fraction', 0, 1, default_value=1)

    global_cv_p = UniformIntegerHyperparameter('global_cv', 2, 20, default_value=2)
    use_multiple_cvs_p = CategoricalHyperparameter('use_multiple_cvs', [True, False], default_value=False)
    global_number_cv_p = UniformIntegerHyperparameter('global_number_cv', 2, 10, default_value=2)

    cs.add_hyperparameters([hold_out_fraction_p, use_hold_out_p, global_cv_p, use_multiple_cvs_p, global_number_cv_p])
    cs.add_condition(EqualsCondition(hold_out_fraction_p, use_hold_out_p, True))
    cs.add_condition(EqualsCondition(global_cv_p, use_hold_out_p, False))
    cs.add_condition(EqualsCondition(use_multiple_cvs_p, use_hold_out_p, False))
    cs.add_condition(EqualsCondition(global_number_cv_p, use_multiple_cvs_p, True))

    # random_feature_selection_p = UniformFloatHyperparameter('random_feature_selection', 0, 1, default_value=1)
    # cs.add_hyperparameters([random_feature_selection_p])

    unbalance_data_p = CategoricalHyperparameter('unbalance_data', [True, False], default_value=False)
    fraction_ids_class0_p = UniformFloatHyperparameter('fraction_ids_class0', 0, 1, default_value=1)
    fraction_ids_class1_p = UniformFloatHyperparameter('fraction_ids_class1', 0, 1, default_value=1)
    sampling_factor_train_only_p = UniformFloatHyperparameter('sampling_factor_train_only', 0, 1, default_value=1)
    cs.add_hyperparameters(
        [unbalance_data_p, fraction_ids_class0_p, fraction_ids_class1_p, sampling_factor_train_only_p])

    cs.add_condition(EqualsCondition(fraction_ids_class0_p, unbalance_data_p, True))
    cs.add_condition(EqualsCondition(fraction_ids_class1_p, unbalance_data_p, True))
    cs.add_condition(EqualsCondition(sampling_factor_train_only_p, unbalance_data_p, False))

    gen = SpaceGenerator()
    space = gen.generate_params()
    space.generate_SMAC(cs)

    return cs


def get_space_2_constraints(total_search_time, my_openml_datasets):
    cs = ConfigurationSpace()
    cs.add_hyperparameter(UniformIntegerHyperparameter('global_search_time_constraint', 10, total_search_time, default_value=10))
    cs.add_hyperparameter(CategoricalHyperparameter('dataset_id', my_openml_datasets, default_value=my_openml_datasets[0]))

    use_evaluation_time_constraint_p = CategoricalHyperparameter('use_evaluation_time_constraint', [True, False],
                                                                 default_value=False)
    global_evaluation_time_constraint_p = UniformIntegerHyperparameter('global_evaluation_time_constraint', 10,
                                                                       total_search_time,
                                                                       default_value=total_search_time)
    cs.add_hyperparameters([use_evaluation_time_constraint_p, global_evaluation_time_constraint_p])
    cs.add_condition(EqualsCondition(global_evaluation_time_constraint_p, use_evaluation_time_constraint_p, True))

    use_search_memory_constraint_p = CategoricalHyperparameter('use_search_memory_constraint', [False],
                                                               default_value=False)
    cs.add_hyperparameters([use_search_memory_constraint_p])

    use_privacy_constraint_p = CategoricalHyperparameter('use_privacy_constraint', [False], default_value=False)
    cs.add_hyperparameters([use_privacy_constraint_p])

    use_training_time_constraint_p = CategoricalHyperparameter('use_training_time_constraint', [False], default_value=False)
    cs.add_hyperparameters([use_training_time_constraint_p])

    use_inference_time_constraint_p = CategoricalHyperparameter('use_inference_time_constraint', [False],
                                                                default_value=False)
    cs.add_hyperparameters([use_inference_time_constraint_p])

    use_pipeline_size_constraint_p = CategoricalHyperparameter('use_pipeline_size_constraint', [True, False],
                                                               default_value=False)
    pipeline_size_constraint_p = UniformIntegerHyperparameter('pipeline_size_constraint', 2000, 350000000,
                                                              default_value=350000000, log=True)
    cs.add_hyperparameters([pipeline_size_constraint_p, use_pipeline_size_constraint_p])
    cs.add_condition(EqualsCondition(pipeline_size_constraint_p, use_pipeline_size_constraint_p, True))

    use_sampling_p = CategoricalHyperparameter('use_sampling', [True, False], default_value=False)
    sample_fraction_p = UniformFloatHyperparameter('sample_fraction', 0, 1, default_value=1)
    cs.add_hyperparameters([sample_fraction_p, use_sampling_p])
    cs.add_condition(EqualsCondition(sample_fraction_p, use_sampling_p, True))

    use_hold_out_p = CategoricalHyperparameter('use_hold_out', [True, False], default_value=True)
    hold_out_fraction_p = UniformFloatHyperparameter('hold_out_fraction', 0, 1, default_value=1)

    global_cv_p = UniformIntegerHyperparameter('global_cv', 2, 20, default_value=2)
    use_multiple_cvs_p = CategoricalHyperparameter('use_multiple_cvs', [True, False], default_value=False)
    global_number_cv_p = UniformIntegerHyperparameter('global_number_cv', 2, 10, default_value=2)

    cs.add_hyperparameters([hold_out_fraction_p, use_hold_out_p, global_cv_p, use_multiple_cvs_p, global_number_cv_p])
    cs.add_condition(EqualsCondition(hold_out_fraction_p, use_hold_out_p, True))
    cs.add_condition(EqualsCondition(global_cv_p, use_hold_out_p, False))
    cs.add_condition(EqualsCondition(use_multiple_cvs_p, use_hold_out_p, False))
    cs.add_condition(EqualsCondition(global_number_cv_p, use_multiple_cvs_p, True))

    # random_feature_selection_p = UniformFloatHyperparameter('random_feature_selection', 0, 1, default_value=1)
    # cs.add_hyperparameters([random_feature_selection_p])

    unbalance_data_p = CategoricalHyperparameter('unbalance_data', [True, False], default_value=False)
    fraction_ids_class0_p = UniformFloatHyperparameter('fraction_ids_class0', 0, 1, default_value=1)
    fraction_ids_class1_p = UniformFloatHyperparameter('fraction_ids_class1', 0, 1, default_value=1)
    sampling_factor_train_only_p = UniformFloatHyperparameter('sampling_factor_train_only', 0, 1, default_value=1)
    cs.add_hyperparameters(
        [unbalance_data_p, fraction_ids_class0_p, fraction_ids_class1_p, sampling_factor_train_only_p])

    cs.add_condition(EqualsCondition(fraction_ids_class0_p, unbalance_data_p, True))
    cs.add_condition(EqualsCondition(fraction_ids_class1_p, unbalance_data_p, True))
    cs.add_condition(EqualsCondition(sampling_factor_train_only_p, unbalance_data_p, False))

    gen = SpaceGenerator()
    space = gen.generate_params()
    space.generate_SMAC(cs)

    return cs