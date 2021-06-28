#overfit
import pickle

import pickle
import numpy as np
from anytree import RenderTree
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

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

    def plot_correlation(self):
        estimates = []
        real_test_scores = []

        for i in range(len(self.runs)):
            estimates.append(self.runs[i].best_trial.value)
            real_test_scores.append(self.runs[i].test_score)

        estimates = MinMaxScaler().fit_transform(np.array(estimates).reshape(-1, 1))
        real_test_scores = MinMaxScaler().fit_transform(np.array(real_test_scores).reshape(-1, 1))

        plt.scatter(real_test_scores.flatten(), estimates.flatten())
        plt.xlabel('Real test score')
        #plt.yscale('log')
        plt.ylabel('Estimates')
        plt.show()

data = pickle.load(open('/home/neutatz/phd2/picture_progress/all_test_datasets/eval_dict_log.p', 'rb'))

print(len(data['dynamic']))

for d in data['dynamic']:
    print('Dataset: ' + str(d.dataset))
    print('Constraint: ' + str(d.constraint))
    best_run = d.get_best_run()
    print('Best Score: ' + str(best_run.test_score))
    print('Best Space: ')
    best_run.print_space()
    print('Best Config: ')
    print(best_run.get_best_config())
    print('All paramas: ')
    print(best_run.more.params)
    print(best_run.more.value)
    d.plot_correlation()
    print('########################################################')
