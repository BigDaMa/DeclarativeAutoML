import fastsklearnfeature.declarative_automl.optuna_package.myautoml.define_space as myspace

from fastsklearnfeature.declarative_automl.optuna_package.myautoml.MyAutoMLTreeSpace import MyAutoMLSpace
from fastsklearnfeature.declarative_automl.optuna_package.data_preprocessing.SimpleImputerOptuna import SimpleImputerOptuna



from ax import *

class SpaceGenerator:
    def __init__(self):
        self.classifier_list = myspace.classifier_list
        self.private_classifier_list = myspace.private_classifier_list
        self.preprocessor_list = myspace.preprocessor_list
        self.scaling_list = myspace.scaling_list
        self.categorical_encoding_list = myspace.categorical_encoding_list
        self.augmentation_list = myspace.augmentation_list

        self.space = MyAutoMLSpace()

        #generate binary or mapping for each hyperparameter


    def generate_params(self):

        parameters = []

        class_weighting_p = ChoiceParameter('class_weighting', parameter_type=ParameterType.BOOL, values=[True, False], is_ordered=False)
        parameters.append(class_weighting_p)
        custom_weighting_p = ChoiceParameter('custom_weighting', parameter_type=ParameterType.BOOL, values=[True, False], is_ordered=False)
        parameters.append(custom_weighting_p)
        custom_weight_p = RangeParameter(name="custom_weight", parameter_type=ParameterType.FLOAT, lower=0.00000001, upper=1.0)
        parameters.append(custom_weight_p)

        '''
        augmentation_p = ChoiceParameter('augmentation', parameter_type=ParameterType.INT, values=range(len(self.augmentation_list)), is_ordered=False)
        parameters.append(augmentation_p)
        for au_i in range(len(self.augmentation_list)):
            augmentation = self.augmentation_list[au_i]
            #augmentation.generate_hyperparameters(self.space, category_aug[au_i])
        '''



        '''
        category_preprocessor = self.space.generate_cat('preprocessor', self.preprocessor_list, self.preprocessor_list[0])
        for p_i in range(len(self.preprocessor_list)):
            preprocessor = self.preprocessor_list[p_i]
            preprocessor.generate_hyperparameters(self.space, category_preprocessor[p_i])
        category_classifier = self.space.generate_cat('classifier', self.classifier_list, self.classifier_list[0])
        for c_i in range(len(self.classifier_list)):
            classifier = self.classifier_list[c_i]
            classifier.generate_hyperparameters(self.space, category_classifier[c_i])

        category_private_classifier = self.space.generate_cat('private_classifier', self.private_classifier_list, self.private_classifier_list[0])
        for c_i in range(len(self.private_classifier_list)):
            private_classifier = self.private_classifier_list[c_i]
            private_classifier.generate_hyperparameters(self.space, category_private_classifier[c_i])

        category_scaler = self.space.generate_cat('scaler', self.scaling_list, self.scaling_list[0])
        for s_i in range(len(self.scaling_list)):
            scaler = self.scaling_list[s_i]
            scaler.generate_hyperparameters(self.space, category_scaler[s_i])

        imputer = SimpleImputerOptuna()
        imputer.generate_hyperparameters(self.space)

        category_categorical_encoding = self.space.generate_cat('categorical_encoding', self.categorical_encoding_list, self.categorical_encoding_list[0])
        for cat_i in range(len(self.categorical_encoding_list)):
            categorical_encoding = self.categorical_encoding_list[cat_i]
            categorical_encoding.generate_hyperparameters(self.space, category_categorical_encoding[cat_i])
        
        return self.space
        '''
        return SearchSpace(parameters=parameters)

