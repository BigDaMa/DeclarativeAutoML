import copy
from ax import *

class MyConstrainedAutoMLSpace:
    def __init__(self):
        self.parameters = []
        self.mapChoice2List = {}
        self.parameterization = None

    def generate_cat(self, name, my_list, default_element, depending_node=None):
        self.parameters.append(ChoiceParameter(name, parameter_type=ParameterType.INT, values=range(len(my_list)), is_ordered=False))
        self.mapChoice2List[name] = my_list

        return range(len(my_list))

    def get_space(self):
        return SearchSpace(parameters=self.parameters)


    def generate_number(self, name, default_element, depending_node=None, low=0.0, high=1.0 , is_float=True, is_log=False):
        parameter_type = ParameterType.FLOAT
        if not is_float:
            parameter_type = ParameterType.INT
        self.parameters.append(RangeParameter(name=name, parameter_type=parameter_type, lower=low, upper=high, log_scale=is_log))

    def suggest_int(self, name, low, high, step=1, log=False):
        return self.parameterization.get(name)

    def suggest_loguniform(self, name, low, high):
        return self.parameterization.get(name)

    def suggest_uniform(self, name, low, high):
        return self.parameterization.get(name)

    def suggest_categorical(self, name, choices):
        return copy.deepcopy(self.mapChoice2List[name][self.parameterization.get(name)])