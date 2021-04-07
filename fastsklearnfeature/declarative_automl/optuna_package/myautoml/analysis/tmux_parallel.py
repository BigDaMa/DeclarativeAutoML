import libtmux
from pathlib import Path
import time
import multiprocessing

datasets = [1134, 1495, 41147, 316, 1085, 1046, 1111, 55, 1116, 448, 1458, 162, 1101, 1561, 1061, 1506, 1235, 4135, 151, 51, 41138, 40645, 1510, 1158, 312, 38, 52, 1216, 41007, 1130]


parallelism = multiprocessing.cpu_count()
server = libtmux.Server()

data_id = 0
running_ids = []
finished = []

session = server.new_session(session_name="install", kill_session=True, attach=False)
session.attached_pane.send_keys('conda activate dAutoML')
session.attached_pane.send_keys('cd /home/neutatz/Software/DeclarativeAutoML')
session.attached_pane.send_keys('git pull origin main')
session.attached_pane.send_keys('python -m pip install .')


time.sleep(60)


while len(finished) < len(datasets):
    if len(running_ids) < parallelism:
        session = server.new_session(session_name="data" + str(datasets[data_id]), kill_session=True, attach=False)
        running_ids.append(datasets[data_id])
        data_id += 1
        session.attached_pane.send_keys('conda activate dAutoML')
        session.attached_pane.send_keys('cd /home/neutatz/Software/DeclarativeAutoML')
        session.attached_pane.send_keys('python fastsklearnfeature/declarative_automl/optuna_package/myautoml/analysis/check_model_parallel_per_data.py -d ' + str(datasets[data_id]))


    #check if anything is done
    to_be_removed = []
    for r in running_ids:
        my_file = Path('/home/neutatz/data/automl_runs/all_results' + str(r) + '.p')
        if my_file.is_file():
            time.sleep(60)
            session = server.find_where({"session_name": "data" + str(r)})
            session.kill_session()
            to_be_removed.append(r)
            finished.append(r)
    for r in to_be_removed:
        running_ids.remove(r)

    time.sleep(5)

