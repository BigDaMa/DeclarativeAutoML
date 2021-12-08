from anytree import RenderTree
import numpy as np

def space2str(parameter_tree):
    my_str = ''
    for pre, _, node in RenderTree(parameter_tree):
        if node.status == True:
            my_str += "%s%s" % (pre, node.name)
            my_str += '\n'
    return my_str

class ConstraintRun(object):
    def __init__(self, space_str, params, test_score, estimated_score):
        self.space_str = space_str
        self.params = params
        self.test_score = test_score
        self.estimated_score = estimated_score

    def print_space(self):
        print(self.space_str)

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