#overfit
import pickle

import pickle
import numpy as np
from anytree import RenderTree

class ConstraintRun(object):
    def __init__(self, space, best_trial, test_score, more=None):
        self.space = space
        self.best_trial = best_trial
        self.test_score = test_score
        self.more = more

    def print_space(self):
        for pre, _, node in RenderTree(self.space.parameter_tree):
            if node.status == True:
                print("%s%s" % (pre, node.name))

    def get_best_config(self):
        return self.best_trial.params

class ConstraintEvaluation(object):
    def __init__(self, dataset=None, constraint=None, system_def=None):
        self.runs = []
        self.dataset = dataset
        self.constraint = constraint
        self.system_def = system_def

    def append(self, constraint_run: ConstraintRun):
        self.runs.append(constraint_run)

    def get_best_run(self):
        max_score = -np.inf
        max_run = None
        for i in range(len(self.runs)):
            if max_score < self.runs[i].test_score:
                max_score = self.runs[i].test_score
                max_run = self.runs[i]
        return max_run

    def get_best_config(self):
        best_run = self.get_best_run()
        return best_run.get_best_config()

    def get_average(self):
        scores = []
        for i in range(len(self.runs)):
            scores.append(self.runs[i].test_score)
        return np.average(scores)

    def get_worst_run(self):
        min_score = np.inf
        min_run = None
        for i in range(len(self.runs)):
            if min_score > self.runs[i].test_score:
                min_score = self.runs[i].test_score
                min_run = self.runs[i]
        return min_run

#data = pickle.load(open('/home/neutatz/phd2/picture_progress/all_test_datasets/eval_dict_log_p2.p', 'rb'))

#data = pickle.load(open('/home/neutatz/phd2/picture_progress/all_test_datasets/eval_dict_log_new_sampling2dfixed#1046.p', 'rb'))

def test_average(myobject):
    scores = []
    for i in range(len(myobject.runs)):
        scores.append(myobject.runs[i].test_score)
    return np.average(scores)

import glob
file_list = glob.glob("/home/neutatz/data/automl_runs/log_mine_*.p")

for f in file_list:

    data = pickle.load(open(f, 'rb'))


    print(len(data['dynamic']))

    for d_i in range(len(data['dynamic'])):
        d = data['dynamic'][d_i]
        static = data['static'][d_i]



        if test_average(d) > test_average(static):
            best_run = d.get_best_run()
            #if best_run.more.params['use_hold_out'] == False:
            #if best_run.more.params['use_sampling'] == True:
            #if best_run.more.params['preprocessor'] == True:
            #if best_run.more.params['classifier'] == True:
            if best_run.more.params['scaler'] > 0:

                print('Dataset: ' + str(d.dataset))
                print('difference: ' + str(test_average(d) - test_average(static)))
                print('Constraint: ' + str(d.constraint))
                print('Best Score: ' + str(best_run.test_score))
                print('Best Space: ')
                best_run.print_space()
                print('Best Config: ')
                print(best_run.get_best_config())
                print('All paramas: ')
                print(best_run.more.params)
                print(best_run.more.value)
                print('########################################################')
