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

    def get_worst_run(self):
        min_score = np.inf
        min_run = None
        for i in range(len(self.runs)):
            if min_score > self.runs[i].test_score:
                min_score = self.runs[i].test_score
                min_run = self.runs[i]
        return min_run

    def get_average_estimate(self):
        estimates = []
        for i in range(len(self.runs)):
            estimates.append(self.runs[i].more.value)
        return np.average(estimates)

data = pickle.load(open('/home/neutatz/phd2/picture_progress/all_test_datasets/eval_dict_log_p2.p', 'rb'))

#data = pickle.load(open('/home/neutatz/phd2/picture_progress/all_test_datasets/machine2_5min_more_weeks_with_log_part1/eval_dict_log_part1.p', 'rb'))

print(len(data['dynamic']))

for d_i in range(len(data['dynamic'])):
    d = data['dynamic'][d_i]
    static = data['static'][d_i]

    if d.get_best_run().test_score < static.get_best_run().test_score:
        print('Dataset: ' + str(d.dataset))
        print('Constraint: ' + str(d.constraint))
        print('difference: ' + str(static.get_best_run().test_score - d.get_best_run().test_score))

        print(static.get_best_run().get_best_config())

        print('Best Space of dynamic: ')
        best_run = d.get_best_run()
        best_run.print_space()

        print('Avg Estimate: ' + str(d.get_average_estimate()))

        print('########################################################')
