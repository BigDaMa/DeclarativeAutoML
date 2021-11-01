import fastsklearnfeature.declarative_automl.optuna_package.myautoml.define_space as myspace

from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.MyAutoMLTreeSpace import MyAutoMLSpace

from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.classifiers.adaboost import AdaBoostClassifierOptuna
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.classifiers.bernoulli_nb import BernoulliNBOptuna
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.classifiers.decision_tree import DecisionTreeClassifierOptuna
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.classifiers.extra_trees import ExtraTreesClassifierOptuna
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.classifiers.gaussian_nb import GaussianNBOptuna
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.classifiers.gradient_boosting import HistGradientBoostingClassifierOptuna
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.classifiers.k_nearest_neighbors import KNeighborsClassifierOptuna
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.classifiers.lda import LDAOptuna
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.classifiers.liblinear_svc import LinearSVCOptuna
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.classifiers.libsvm_svc import SVCOptuna
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.classifiers.mlp import MLPOptuna
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.classifiers.multinomial_nb import MultinomialNBOptuna
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.classifiers.passive_aggressive import PassiveAggressiveOptuna
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.classifiers.qda import QuadraticDiscriminantAnalysisOptuna
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.classifiers.random_forest import RandomForestClassifierOptuna
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.classifiers.sgd import SGDClassifierOptuna

from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.feature_preprocessers.densifier import Densifier
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.feature_preprocessers.extra_trees_preproc_for_classification import ExtraTreesPreprocessorClassification
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.feature_preprocessers.extra_trees_preproc_for_regression import ExtraTreesPreprocessorRegression
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.feature_preprocessers.fast_ica import FastICAOptuna
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.feature_preprocessers.feature_agglomeration import FeatureAgglomerationOptuna
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.feature_preprocessers.kernel_pca import KernelPCAOptuna
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.feature_preprocessers.kitchen_sinks import RBFSamplerOptuna
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.feature_preprocessers.liblinear_svc_preprocessor import LibLinear_Preprocessor
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.feature_preprocessers.no_preprocessing import NoPreprocessing
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.feature_preprocessers.nystroem_sampler import NystroemOptuna
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.feature_preprocessers.pca import PCAOptuna
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.feature_preprocessers.polynomial import PolynomialFeaturesOptuna
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.feature_preprocessers.random_trees_embedding import RandomTreesEmbeddingOptuna
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.feature_preprocessers.select_percentile_classification import SelectPercentileClassification
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.feature_preprocessers.select_percentile_regression import SelectPercentileRegression
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.feature_preprocessers.select_rates_classification import SelectClassificationRates
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.feature_preprocessers.select_rates_regression import SelectRegressionRates
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.feature_preprocessers.truncatedSVD import TruncatedSVDOptuna

from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.data_preprocessors.balancing import Balancing
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.data_preprocessors.numerical_imputation import NumericalImputation
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.data_preprocessors.minority_coalescense.minority_coalescer import MinorityCoalescer
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.data_preprocessors.minority_coalescense.no_coalescense import NoCoalescence

from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.data_preprocessors.rescaling.minmax import MinMaxScalerComponent
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.data_preprocessors.rescaling.none import NoRescalingComponent
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.data_preprocessors.rescaling.normalize import NormalizerComponent
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.data_preprocessors.rescaling.power_transformer import PowerTransformerComponent
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.data_preprocessors.rescaling.quantile_transformer import QuantileTransformerComponent
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.data_preprocessors.rescaling.robust_scaler import RobustScalerComponent
from fastsklearnfeature.declarative_automl.optuna_package.autosklearn_flex.components.data_preprocessors.rescaling.standardize import StandardScalerComponent

import pickle

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

        my_classifiers = [AdaBoostClassifierOptuna(),
                          BernoulliNBOptuna(),
                          DecisionTreeClassifierOptuna(),
                          ExtraTreesClassifierOptuna(),
                          GaussianNBOptuna(),
                          HistGradientBoostingClassifierOptuna(),
                          KNeighborsClassifierOptuna(),
                          LDAOptuna(),
                          LinearSVCOptuna(),
                          SVCOptuna(),
                          MLPOptuna(),
                          MultinomialNBOptuna(),
                          PassiveAggressiveOptuna(),
                          QuadraticDiscriminantAnalysisOptuna(),
                          RandomForestClassifierOptuna(),
                          SGDClassifierOptuna()]

        my_preprocessors = [Densifier(),
                            ExtraTreesPreprocessorClassification(),
                            ExtraTreesPreprocessorRegression(),
                            FastICAOptuna(),
                            FeatureAgglomerationOptuna(),
                            KernelPCAOptuna(),
                            RBFSamplerOptuna(),
                            LibLinear_Preprocessor(),
                            NoPreprocessing(),
                            NystroemOptuna(),
                            PCAOptuna(),
                            PolynomialFeaturesOptuna(),
                            RandomTreesEmbeddingOptuna(),
                            SelectPercentileClassification(),
                            SelectPercentileRegression(),
                            SelectClassificationRates(),
                            SelectRegressionRates(),
                            TruncatedSVDOptuna()]

        coalescense = [NoCoalescence(),
                       MinorityCoalescer()]

        rescalers = [MinMaxScalerComponent(),
                     NoRescalingComponent(),
                     NormalizerComponent(),
                     PowerTransformerComponent(),
                     QuantileTransformerComponent(),
                     RobustScalerComponent(),
                     StandardScalerComponent()]

        #data preprocessing
        balancer = Balancing()
        balancer.generate_hyperparameters(self.space)

        imputer = NumericalImputation()
        imputer.generate_hyperparameters(self.space)

        for module_name in ['encoding', 'no_encoding', 'one_hot_encoding']:
            self.space.generate_cat(module_name, ['True', 'False'], 'True')

        for coalescenser in coalescense:
            module_name = str(coalescenser.__class__.__module__).split('.')[-1]
            use_coalescenser = self.space.generate_cat(module_name, ['True', 'False'], 'True')
            coalescenser.generate_hyperparameters(self.space, depending_node=use_coalescenser[0])

        for rescaler in rescalers:
            module_name = str(rescaler.__class__.__module__).split('.')[-1]
            use_rescaler = self.space.generate_cat(module_name, ['True', 'False'], 'True')
            rescaler.generate_hyperparameters(self.space, depending_node=use_rescaler[0])

        for preprocessor in my_preprocessors:
            module_name = str(preprocessor.__class__.__module__).split('.')[-1]
            use_preprocessor = self.space.generate_cat(module_name, ['True', 'False'], 'True')
            preprocessor.generate_hyperparameters(self.space, depending_node=use_preprocessor[0])

        for classifier in my_classifiers:
            module_name = str(classifier.__class__.__module__).split('.')[-1]
            use_classifier = self.space.generate_cat(module_name, ['True', 'False'], 'True')
            classifier.generate_hyperparameters(self.space, depending_node=use_classifier[0])


        return self.space



from anytree import RenderTree
import optuna

def objective(trial):
    gen = SpaceGenerator()
    space = gen.generate_params()
    space.sample_parameters(trial)

    for pre, _, node in RenderTree(space.parameter_tree):
        if node.status:
            print("%s%s" % (pre, node.name))


    pickle.dump(space.name2node, open('/tmp/space.p', 'wb+'))

    return 0

study = optuna.create_study()
study.optimize(objective, n_trials=2)

