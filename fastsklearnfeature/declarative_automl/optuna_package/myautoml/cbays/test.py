from ax import *
space = SearchSpace(parameters=[
    ChoiceParameter('class_weighting', parameter_type=ParameterType.BOOL, values=[True, False], is_ordered=False),
    ChoiceParameter('custom_weighting', parameter_type=ParameterType.BOOL, values=[True, False], is_ordered=False),
    RangeParameter(name="custom_weight", parameter_type=ParameterType.FLOAT, lower=0.00000001, upper=1.0)])

exp = SimpleExperiment(
            name="test",
            search_space=space,
            evaluation_function=my_objective,
            objective_name='accuracy',
            minimize=False
        )

sobol = Models.SOBOL(exp.search_space)
for i in range(5):
     exp.new_trial(generator_run=sobol.gen(1))

for i in range(25):
     gpei = Models.GPEI(experiment=exp, data=exp.eval())
     batch = exp.new_trial(generator_run=gpei.gen(1))