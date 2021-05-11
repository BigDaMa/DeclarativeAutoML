import copy

from emukit.core import CategoricalParameter, ContinuousParameter,  DiscreteParameter, OneHotEncoding
from emukit.core import ParameterSpace
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.cbays.emukit_version.LogContinuousParameter import LogContinuousParameter

class EmuKitSpace:
    def __init__(self):
        self.parameters = []
        self.mapChoice2List = {}
        self.parameterization = None
        self.mapName2Id = {}
        self.counter = 0

    def generate_cat(self, name, my_list, default_element, depending_node=None):
        self.parameters.append(CategoricalParameter(name, OneHotEncoding(my_list)))
        self.mapChoice2List[name] = my_list
        self.mapName2Id[name] = copy.deepcopy(self.counter)
        self.counter += len(my_list)

        return range(len(my_list))

    def get_space(self):
        return ParameterSpace(self.parameters)


    def generate_number(self, name, default_element, depending_node=None, low=0.0, high=1.0 , is_float=True, is_log=False):
        if is_log:
            if is_float and low == 0.0:
                low = 0.0001
            if not is_float and low == 0.0:
                low = 0.1
            self.parameters.append(LogContinuousParameter(name=name, min_value=low, max_value=high))
        else:
            self.parameters.append(ContinuousParameter(name=name, min_value=low, max_value=high))
        self.mapName2Id[name] = copy.deepcopy(self.counter)
        self.counter += 1


    def suggest_int(self, name, low, high, step=1, log=False):
        return int(self.parameterization[self.mapName2Id[name]])

    def suggest_loguniform(self, name, low, high):
        return self.parameterization[self.mapName2Id[name]]

    def suggest_uniform(self, name, low, high):
        return self.parameterization[self.mapName2Id[name]]

    def suggest_categorical(self, name, choices):
        for i in range(len(self.mapChoice2List[name])):
            if self.parameterization[self.mapName2Id[name] + i] == True:
                return self.mapChoice2List[name][i]