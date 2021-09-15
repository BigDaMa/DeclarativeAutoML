from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import MyPool
import time

start_time_tt = time.time()

def sample_and_evaluate(id):
    if time.time() - start_time_tt > 60:
        return None

    print(id)
    time.sleep(1)


topk = 2
with MyPool(processes=topk) as pool:
    results = pool.map(sample_and_evaluate, range(100000))