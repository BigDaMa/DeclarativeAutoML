import openml
import getpass

openml.config.apikey = '4384bd56dad8c3d2c0f6630c52ef5567'
openml.config.cache_directory = '/home/' + getpass.getuser() + '/phd2/cache_openml'

map_dataset = {}
map_dataset['31'] = 'foreign_worker@{yes,no}'
map_dataset['802'] = 'sex@{female,male}'
map_dataset['1590'] = 'sex@{Female,Male}'
map_dataset['1461'] = 'AGE@{True,False}'
map_dataset['42193'] = 'race_Caucasian@{0,1}'
map_dataset['1480'] = 'V2@{Female,Male}'
# map_dataset['804'] = 'Gender@{0,1}'
map_dataset['42178'] = 'gender@STRING'
map_dataset['981'] = 'Gender@{Female,Male}'
map_dataset['40536'] = 'samerace@{0,1}'
map_dataset['40945'] = 'sex@{female,male}'
map_dataset['451'] = 'Sex@{female,male}'
# map_dataset['945'] = 'sex@{female,male}'
map_dataset['446'] = 'sex@{Female,Male}'
map_dataset['1017'] = 'sex@{0,1}'
map_dataset['957'] = 'Sex@{0,1}'
map_dataset['41430'] = 'SEX@{True,False}'
map_dataset['1240'] = 'sex@{Female,Male}'
map_dataset['1018'] = 'sex@{Female,Male}'
# map_dataset['55'] = 'SEX@{male,female}'
map_dataset['38'] = 'sex@{F,M}'
map_dataset['1003'] = 'sex@{male,female}'
map_dataset['934'] = 'race@{black,white}'

my_openml_tasks = list(map_dataset.keys())

tasks = [168794, 168797, 168796, 189871, 189861, 167185, 189872, 189908, 75105, 167152, 168793, 189860, 189862, 126026, 189866, 189873, 168792, 75193, 168798, 167201, 167149, 167200, 189874, 167181, 167083, 167161, 189865, 189906, 167168, 126029, 167104, 126025, 75097, 168795, 75127, 189905, 189909, 167190, 167184]

print('hello')
for task_id in tasks:
    task = openml.tasks.get_task(task_id)
    data_id = task.get_dataset().dataset_id
    print(data_id)
    if str(data_id) in map_dataset:
        print('found: ' + str(data_id))