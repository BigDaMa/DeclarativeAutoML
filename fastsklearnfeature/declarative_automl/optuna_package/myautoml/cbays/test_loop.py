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
    print(str(model_reg.n_estimators) + ': time: ' + str(training_time) + ' score: ' + str(r2_score))
    return {"r2": (r2_score, 0.0), "training_time": (training_time, 0.0)}


from ax import *

best_parameters, values, experiment, model = optimize(
    parameters=[
        {
            "name": "n_estimators",
            "type": "range",
            "bounds": [1, 1000],
            "value_type": "int"
        }
    ],
    experiment_name="test",
    objective_name="r2",
    evaluation_function=eval,
    minimize=True,  # Optional, defaults to False.
    outcome_constraints=["training_time <= 0.15"],  # Optional.
    total_trials=30, # Optional.
)

data = experiment.fetch_data().df.values
cur_max = 0.0
arm_max = None
for i in range(len(data)):
    if data[i, 1] == 'r2' and data[i, 2] < cur_max:
        if data[i+1, 2] <= 0.15:
            cur_max = data[i, 2]
            arm_max = data[i, 0]

print(cur_max)