from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time
from emukit.core import CategoricalParameter, ContinuousParameter,  DiscreteParameter, OneHotEncoding

X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model_reg = RandomForestRegressor(n_estimators=100)

def eval(parameters):
    print(parameters)
    result = np.zeros((len(parameters), 1))
    constraint = np.zeros((len(parameters), 1))
    for i in range(len(parameters)):
        start = time.time()
        model_reg.n_estimators = int(parameters[i,0])
        model_reg.fit(X_train, y_train)
        training_time = time.time() - start
        constraint[i, 0] = 0.5 - training_time
        result[i, 0] = -1 * model_reg.score(X_test, y_test)
        print('acc: ' + str(result[i, 0]), ' constraint: ' + str(constraint[i, 0]))
    return result, constraint

import GPy
import numpy as np

from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.bayesian_optimization.loops import UnknownConstraintBayesianOptimizationLoop
from emukit.core import ContinuousParameter, ParameterSpace
from emukit.core.loop import UserFunctionWrapper, FixedIterationsStoppingCondition
from emukit.model_wrappers import GPyModelWrapper

from fastsklearnfeature.declarative_automl.optuna_package.myautoml.cbays.emukit_version.EmuKitSpace import EmuKitSpace

n_iterations = 50


e =EmuKitSpace()
e.generate_number('n_estimators', default_element=100, low=1, high=1000, is_float=False, is_log=True)
e.generate_cat('tesadst', my_list=['test1', 'test2'], default_element='test1')

space = e.get_space()

x_init = np.array([[1,0,1], [100,1,0 ], [999,0,1]])

y_init, y_constraint_init = eval(x_init)

# Make GPy objective model
gpy_model = GPy.models.GPRegression(x_init, y_init)
model = GPyModelWrapper(gpy_model)

# Make GPy constraint model
gpy_constraint_model = GPy.models.GPRegression(x_init, y_init)
constraint_model = GPyModelWrapper(gpy_constraint_model)

acquisition = ExpectedImprovement(model)

# Make loop and collect points
bo = UnknownConstraintBayesianOptimizationLoop(model_objective=model, space=space, acquisition=acquisition, model_constraint=constraint_model)
bo.run_loop(UserFunctionWrapper(eval, extra_output_names=['Y_constraint']), FixedIterationsStoppingCondition(n_iterations))

print(bo.loop_state.Y)
print(bo.loop_state.X)
print(bo.loop_state.results)

print('avg: ' + str(np.average(bo.loop_state.Y[:,0])))

minimum_value = np.min(bo.loop_state.Y[:,0])
best_found_value_per_iteration = np.minimum.accumulate(bo.loop_state.Y[:,0]).flatten()

print(minimum_value)
print(best_found_value_per_iteration)