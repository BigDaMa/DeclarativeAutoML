import libtmux
import time

program = '/home/neutatz/Software/DeclarativeAutoML/fastsklearnfeature/declarative_automl/optuna_package/myautoml/smac_it/minimal/active_learning_smac_minimal.py'
my_folder = '/home/neutatz/data/pshare/'

conda_name = 'AutoMLD'

parallelism = 2#15#multiprocessing.cpu_count()
server = libtmux.Server()

session = server.new_session(session_name="install", kill_session=True, attach=False)
session.attached_pane.send_keys('exec bash')
session.attached_pane.send_keys('conda activate ' + conda_name)
session.attached_pane.send_keys('cd /home/neutatz/Software/DeclarativeAutoML')
#session.attached_pane.send_keys('git pull origin main')
session.attached_pane.send_keys('python -m pip install .')


time.sleep(60)

for i in range(parallelism):
    session = server.new_session(session_name="data" + str(i), kill_session=True, attach=False)
    session.attached_pane.send_keys('exec bash')
    session.attached_pane.send_keys('conda activate ' + conda_name)
    session.attached_pane.send_keys('cd /home/neutatz/Software/DeclarativeAutoML')
    session.attached_pane.send_keys('python ' + program + ' -r ' + str(i) + ' -p ' + str(my_folder))





