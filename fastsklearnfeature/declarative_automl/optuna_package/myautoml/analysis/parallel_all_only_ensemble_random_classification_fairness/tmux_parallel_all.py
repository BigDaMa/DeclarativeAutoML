import libtmux
from pathlib import Path
import time
import argparse

datasets = [0,1,2,3,4]


program = '/home/neutatz/Software/DeclarativeAutoML/fastsklearnfeature/declarative_automl/optuna_package/myautoml/analysis/parallel_all_only_ensemble_random_classification_fairness/check_model_parallel_per_data_minimum_all1.py'
outputname = 'good_random_fairness'

conda_name = 'AutoMLD'

parallelism = 20#15#multiprocessing.cpu_count()
server = libtmux.Server()

data_id = 0
running_ids = []
finished = []

session = server.new_session(session_name="install", kill_session=True, attach=False)
session.attached_pane.send_keys('exec bash')
session.attached_pane.send_keys('conda activate ' + conda_name)
session.attached_pane.send_keys('cd /home/neutatz/Software/DeclarativeAutoML')
#session.attached_pane.send_keys('git pull origin main')
#session.attached_pane.send_keys('python -m pip install .')


time.sleep(60)


while len(finished) < len(datasets):
    if len(running_ids) < parallelism and data_id < len(datasets):
        session = server.new_session(session_name="data" + str(datasets[data_id]), kill_session=True, attach=False)
        running_ids.append(datasets[data_id])
        session.attached_pane.send_keys('exec bash')
        session.attached_pane.send_keys('conda activate ' + conda_name)
        session.attached_pane.send_keys('cd /home/neutatz/Software/DeclarativeAutoML')
        session.attached_pane.send_keys('python ' + program + ' -d ' + str(datasets[data_id]) + ' -o ' + str(outputname))
        data_id += 1


    #check if anything is done
    to_be_removed = []
    for r in running_ids:
        my_file = Path('/home/neutatz/data/automl_runs/' + outputname + '_' + str(r) + '.p')
        if my_file.is_file():
            time.sleep(60)
            session = server.find_where({"session_name": "data" + str(r)})
            session.kill_session()
            to_be_removed.append(r)
            finished.append(r)
    for r in to_be_removed:
        running_ids.remove(r)

    time.sleep(5)

