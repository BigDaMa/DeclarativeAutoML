if __name__ ==  '__main__':
    """
    ==========================
    scipy-style fmin interface
    ==========================
    """
    from smac.optimizer.acquisition import AbstractAcquisitionFunction
    from smac.epm.base_epm import AbstractEPM
    from smac.epm.rf_with_instances import RandomForestWithInstances
    import numpy as np
    from smac.intensification.simple_intensifier import SimpleIntensifier
    from ConfigSpace.hyperparameters import CategoricalHyperparameter
    import matplotlib.pyplot as plt
    import numpy as np
    from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.MyAutoMLProcessClassBalance import \
        MyAutoML
    import optuna
    import time
    from sklearn.metrics import make_scorer
    import numpy as np
    from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.Space_GenerationTreeBalance import \
        SpaceGenerator
    import copy
    from optuna.trial import FrozenTrial
    from anytree import RenderTree
    from sklearn.ensemble import RandomForestRegressor
    from optuna.samplers import RandomSampler
    import pickle
    import heapq
    import fastsklearnfeature.declarative_automl.optuna_package.myautoml.mp_global_vars as mp_glob
    from sklearn.metrics import f1_score
    from fastsklearnfeature.declarative_automl.optuna_package.myautoml.feature_transformation.FeatureTransformations import \
        FeatureTransformations
    from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import get_data
    from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import data2features
    from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import space2features
    from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import MyPool
    from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import get_feature_names
    from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import ifNull
    from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import generate_parameters
    from sklearn.model_selection import GroupKFold
    from sklearn.model_selection import GridSearchCV

    from smac.optimizer.acquisition import LogEI

    class UncertaintySampling(AbstractAcquisitionFunction):
        def __init__(self,
                     model: AbstractEPM):

            super(UncertaintySampling, self).__init__(model)
            self.long_name = 'Uncertainty Sampling'
            self.num_data = None
            self._required_updates = ('model', 'num_data')

        def _compute(self, X: np.ndarray) -> np.ndarray:
            if self.num_data is None:
                raise ValueError('No current number of Datapoints specified. Call update('
                                 'num_data=<int>) to inform the acquisition function '
                                 'about the number of datapoints.')
            if len(X.shape) == 1:
                X = X[:, np.newaxis]
            #m, var_ = self.model.predict_marginalized_over_instances(X)
            m, var_ = self.model.predict(X)
            return -var_

    class MinSampling(AbstractAcquisitionFunction):
        def __init__(self,
                     model: AbstractEPM):

            super(MinSampling, self).__init__(model)
            self.long_name = 'Uncertainty Sampling'
            self.num_data = None
            self._required_updates = ('model', 'num_data')
            self.count = 0

        def _compute(self, X: np.ndarray) -> np.ndarray:
            if self.num_data is None:
                raise ValueError('No current number of Datapoints specified. Call update('
                                 'num_data=<int>) to inform the acquisition function '
                                 'about the number of datapoints.')
            if len(X.shape) == 1:
                X = X[:, np.newaxis]
            #m, var_ = self.model.predict_marginalized_over_instances(X)
            m, var_ = self.model.predict(X)
            self.count += 1
            print(self.count)
            return m


    import logging

    import numpy as np
    from ConfigSpace.hyperparameters import UniformFloatHyperparameter

    # Import ConfigSpace and different types of parameters
    from smac.configspace import ConfigurationSpace
    from smac.facade.smac_hpo_facade import SMAC4HPO
    from smac.initial_design.latin_hypercube_design import LHDesign
    from smac.runhistory.runhistory2epm import RunHistory2EPM4InvScaledCost
    # Import SMAC-utilities
    from smac.scenario.scenario import Scenario


    def rosenbrock_2d(trial):
        """ The 2 dimensional Rosenbrock function as a toy model
        The Rosenbrock function is well know in the optimization community and
        often serves as a toy problem. It can be defined for arbitrary
        dimensions. The minimium is always at x_i = 1 with a function value of
        zero. All input parameters are continuous. The search domain for
        all x's is the interval [-5, 10].
        """
        x1 = trial.suggest_uniform('x1', -5, 10)

        trial.set_user_attr('features', [x1,])

        #val = 100. * (x2 - x1 ** 2.) ** 2. + (1 - x1) ** 2.
        val = (x1 - 2) ** 2.
        return val


    study = optuna.create_study(direction='minimize', sampler=RandomSampler(seed=42))
    study.optimize(rosenbrock_2d, n_trials=4, n_jobs=1)

    all_trials = study.get_trials()
    X_meta = []
    y_meta = []

    for t in range(len(study.get_trials())):
        current_trial = all_trials[t]
        if current_trial.value >= 0 and current_trial.value <= 1.0:
            y_meta.append(current_trial.value)
        else:
            y_meta.append(0.0)
        X_meta.append(current_trial.user_attrs['features'])

    for i in range(20):
        X_meta_d = np.vstack(X_meta)
        model = RandomForestRegressor(n_estimators=1000)
        model.fit(X_meta_d, y_meta)

        def predict_range(model, X):
            y_pred = model.predict(X)
            return y_pred


        def optimize_uncertainty(trial):
            features = np.array([[trial.suggest_uniform('x1', -5, 10),],])
            predictions = []
            for tree in range(model.n_estimators):
                predictions.append(predict_range(model.estimators_[tree], features))

            stddev_pred = np.std(np.matrix(predictions).transpose(), axis=1)
            return stddev_pred

        study_uncertainty = optuna.create_study(direction='maximize')
        study_uncertainty.optimize(optimize_uncertainty, n_trials=100, n_jobs=1)

        y_new = rosenbrock_2d(study_uncertainty.best_trial)
        x_new = np.array([study_uncertainty.best_params['x1'],])
        y_meta.append(y_new)
        X_meta.append(x_new)

        print(y_new)



    xx = [-3.0, -2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    estimated = []
    y_1 = []
    for i in xx:
        estimated.append(model.predict(np.array([[float(i)], ]))[0])
        y_1.append(float(i) ** 2.)

    print(estimated)
    plt.scatter(xx, estimated)
    plt.scatter(xx, y_1)
    plt.show()

    scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                         "runcount-limit": 2,
                         # max. number of function evaluations; for this example set to a low number
                         "cs": cs,  # configuration space
                         "deterministic": "true",
                         # "shared_model": True,
                         # "input_psmac_dirs": '/home/neutatz/phd2/smac/'
                         })

    smac2 = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42),
                    tae_runner=rosenbrock_2d,
                    runhistory2epm=RunHistory2EPM4InvScaledCost,
                    acquisition_function_optimizer_kwargs={'max_steps': 10000, 'n_sls_iterations': 10000},
                    acquisition_function=MinSampling,
                    initial_design=LHDesign,
                    initial_design_kwargs={'n_configs_x_params': 0, 'max_config_fracs': 0.0},
                    model=RandomForestWithInstances,
                    model_kwargs={'num_trees': 1000,
                                  'do_bootstrapping': True,
                                  'ratio_features': 1.0,
                                  'min_samples_split': 2,
                                  'min_samples_leaf': 1,
                                  }
                    )

    smac2.solver.epm_chooser.acq_optimizer.acquisition_function.model = smac.solver.epm_chooser.acq_optimizer.acquisition_function.model
    smac2.solver.scenario.acq_opt_challengers = 100000
    smac2.optimize()

    print('test')