from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import categorical
import copy
from anytree import Node
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition


class MyAutoMLSpace:
    def __init__(self):
        self.parameter_tree = Node(name='root', parent=None, status=True)
        self.name2node = {}
        self.trial = None
        self.name2parameter = {}

    def generate_cat(self, name, my_list, default_element, depending_node=None):
        parent_node = self.parameter_tree
        if type(depending_node) != type(None):
            parent_node = depending_node

        categorical_node = Node(name=name, parent=parent_node, status=True, default_element=default_element, is_default=False)

        self.name2node[str(name)] = categorical_node
        for element in my_list:
            is_default = str(default_element) == str(element)
            self.name2node[str(name) + '##' + str(element)] = Node(name=str(name) + '##' + str(element), parent=categorical_node, status=True, is_default=is_default)

        return categorical_node.children


    def generate_number(self, name, default_element, depending_node=None, low=0.0, high=1.0 , is_float=True, is_log=False):
        parent_node = self.parameter_tree
        if type(depending_node) != type(None):
            parent_node = depending_node

        numeric_node = Node(name=name, parent=parent_node, status=True, default_element=default_element, is_default=False)
        self.name2node[str(name)] = numeric_node
        return numeric_node


    def recursive_sampling(self, node, trial):
        for child in node.children:
            if node.status:
                child.status = trial.suggest_categorical(child.name, [True, False])
            else:
                if child.is_default and node.parent.status:
                    child.status = True
                else:
                    child.status = False
            self.recursive_sampling(child, trial)

    def sample_parameters(self, x):
        # create dependency graph
        self.recursive_sampling(self.parameter_tree, x)

    def recursive_sampling_SMAC(self, node, x):
        for child in node.children:
            if node.status:
                child.status = x[child.name]
            else:
                if child.is_default and node.parent.status:
                    child.status = True
                else:
                    child.status = False
            self.recursive_sampling_SMAC(child, x)

    def sample_parameters_SMAC(self, x):
        #create dependency graph
        self.recursive_sampling_SMAC(self.parameter_tree, x)

    def recursive_generate_SMAC(self, node, cs):
        for child in node.children:
            param = CategoricalHyperparameter(child.name, [True, False], default_value=True)
            self.name2parameter[child.name] = param
            cs.add_hyperparameter(param)
            if node.name != 'root':
                cs.add_condition(EqualsCondition(param, self.name2parameter[node.name], True))
            self.recursive_generate_SMAC(child, cs)

    def generate_SMAC(self, cs):
        # create dependency graph
        self.recursive_generate_SMAC(self.parameter_tree, cs)






    def suggest_int(self, name, low, high, step=1, log=False):
        if self.name2node[str(name)].status:
            return self.trial.suggest_int(name, low, high, step=step, log=log)
        else:
            return self.name2node[str(name)].default_element

    def suggest_loguniform(self, name, low, high):
        if self.name2node[str(name)].status:
            return self.trial.suggest_loguniform(name, low, high)
        else:
            return self.name2node[str(name)].default_element

    def suggest_uniform(self, name, low, high):
        if self.name2node[str(name)].status:
            return self.trial.suggest_uniform(name, low, high)
        else:
            return self.name2node[str(name)].default_element



    def suggest_categorical(self, name, choices):
        new_list = []
        for element in choices:
            if self.name2node[str(name) + '##' + str(element)].status:
                new_list.append(element)

        if len(new_list) == 0:
            return copy.deepcopy(self.name2node[str(name)].default_element)

        return copy.deepcopy(categorical(self.trial, name, new_list))