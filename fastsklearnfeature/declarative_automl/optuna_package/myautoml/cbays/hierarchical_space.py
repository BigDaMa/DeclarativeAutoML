from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time

X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model_reg = RandomForestRegressor(n_estimators=100)

def eval(parameterization):
    A = parameterization.get("A") > 0.5
    B = parameterization.get("B") > 0.5
    print('A: ' + str(A) + ' B: ' + str(B))
    return {"r2": (1.0, 0.0)}


from ax import *

branin_search_space = SearchSpace(
    parameters=[
        #ChoiceParameter('A', parameter_type=ParameterType.BOOL, values=[True, False], is_ordered=False),
        #ChoiceParameter('B', parameter_type=ParameterType.BOOL, values=[True, False], is_ordered=False)
        RangeParameter(name="A", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0),
        RangeParameter(name="B", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0)
    ],
    parameter_constraints=[ParameterConstraint(constraint_dict={'A': -1.0, 'B': 1.0}, bound=0.0)]
)
exp = SimpleExperiment(
    name="test_branin",
    search_space=branin_search_space,
    evaluation_function=eval,
    objective_name='r2',
    minimize=True
)

print(f"Running Sobol initialization trials...")
sobol = Models.SOBOL(exp.search_space)
for i in range(10):
    exp.new_trial(generator_run=sobol.gen(1))

for i in range(55):
    print(f"Running GP+EI optimization trial {i + 1}/25...")
    # Reinitialize GP+EI model at each step with updated data.
    gpei = Models.BOTORCH(experiment=exp, data=exp.eval())
    batch = exp.new_trial(generator_run=gpei.gen(1))