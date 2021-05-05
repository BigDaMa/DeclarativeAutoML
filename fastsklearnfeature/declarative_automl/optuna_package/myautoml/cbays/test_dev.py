from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time

X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model_reg = RandomForestRegressor(n_estimators=100)

def eval(parameterization):
    start = time.time()
    model_reg.n_estimators = parameterization.get("n_estimators")
    model_reg.fit(X_train, y_train)
    training_time = time.time() - start
    r2_score = -1.0 * model_reg.score(X_test, y_test)
    return {"r2": (r2_score, 0.0), "training_time": (training_time, 0.0)}


from ax import *

branin_search_space = SearchSpace(
    parameters=[
        RangeParameter(
            name="n_estimators", parameter_type=ParameterType.INT, lower=1, upper=1000
        )
    ]
)
exp = SimpleExperiment(
    name="test_branin",
    search_space=branin_search_space,
    evaluation_function=eval,
    objective_name='r2',
    minimize=True,
    outcome_constraints=[OutcomeConstraint(Metric('training_time', lower_is_better=True), op=ComparisonOp.LEQ, bound=0.15, relative=False)]
)

print(f"Running Sobol initialization trials...")
sobol = Models.SOBOL(exp.search_space)
for i in range(5):
    exp.new_trial(generator_run=sobol.gen(1))

for i in range(25):
    print(f"Running GP+EI optimization trial {i + 1}/25...")
    # Reinitialize GP+EI model at each step with updated data.
    gpei = Models.BOTORCH(experiment=exp, data=exp.eval())
    batch = exp.new_trial(generator_run=gpei.gen(1))

data = exp.fetch_data().df.values
cur_max = 0.0
arm_max = None
for i in range(len(data)):
    if data[i, 1] == 'r2' and data[i, 2] < cur_max:
        if data[i+1, 2] <= 0.15:
            cur_max = data[i, 2]
            arm_max = data[i, 0]

print(cur_max)