from emukit.core.loop import UserFunctionResult
from emukit.examples.gp_bayesian_optimization.unknown_constraint_bayesian_optimization import (
    UnknownConstraintGPBayesianOptimization,)
from emukit.core.initial_designs.random_design import RandomDesign
import numpy as np
from emukit.core import ContinuousParameter
from emukit.core import ParameterSpace

limit = 0.5

def objective1(parameterization):
    x = parameterization[0] * parameterization[1]
    c = (parameterization[1] - limit) *-1

    return x, c

def objective_multi(parameterization):
    results = np.zeros((len(parameterization), 1))
    constraints = np.zeros((len(parameterization), 1))
    for p in range(parameterization.shape[0]):
        results[p], constraints[p] = objective1(parameterization[p])
    return results * -1, constraints


parameters = [ContinuousParameter(name='a', min_value=0.0, max_value=1.0), ContinuousParameter(name='b', min_value=0.0, max_value=1.0)]

design = RandomDesign(ParameterSpace(parameters))
num_data_points = 5
X_init = design.get_samples(num_data_points)
y_init, y_constraint_init = objective_multi(X_init)

best_result = 0.0
bo = UnknownConstraintGPBayesianOptimization(variables_list=parameters, X=X_init, Y=y_init, Yc=y_constraint_init, batch_size=1)
results = []
for i in range(100):
    print(i)
    X_new = bo.get_next_points(results)
    Y_new, Yc_new = objective1(X_new[0])

    if best_result < Y_new and Yc_new > 0.0:
        best_result = Y_new
        print(best_result)

    Y_new *= -1
    results = [UserFunctionResult(X_new[0], np.array([Y_new]), Y_constraint=np.array([Yc_new]))]

print(best_result)