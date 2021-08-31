import typing

import numpy as np
from pyrfr import regression

from smac.configspace import ConfigurationSpace
from smac.epm.base_rf import BaseModel
from smac.utils.constants import N_TREES, VERY_SMALL_NUMBER
import copy
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.smac_it.SMACFeatureTransformations import FeatureTransformations
import pickle

import copy
import typing
import warnings

import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import NotFittedError

__author__ = "Aaron Klein"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Aaron Klein"
__email__ = "kleinaa@cs.uni-freiburg.de"
__version__ = "0.0.1"


class CustomRandomForest(BaseModel):

    """Random forest that takes instance features into account.

    Attributes
    ----------
    rf_opts : regression.rf_opts
        Random forest hyperparameter
    n_points_per_tree : int
    rf : regression.binary_rss_forest
        Only available after training
    hypers: list
        List of random forest hyperparameters
    unlog_y: bool
    seed : int
    types : np.ndarray
    bounds : list
    rng : np.random.RandomState
    logger : logging.logger
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        types: typing.List[int],
        bounds: typing.List[typing.Tuple[float, float]],
        seed: int,
        log_y: bool = False,
        num_trees: int = N_TREES,
        do_bootstrapping: bool = True,
        n_points_per_tree: int = -1,
        ratio_features: float = 5. / 6.,
        min_samples_split: int = 3,
        min_samples_leaf: int = 3,
        max_depth: int = 2**20,
        eps_purity: float = 1e-8,
        max_num_nodes: int = 2**20,
        instance_features: typing.Optional[np.ndarray] = None,
        pca_components: typing.Optional[int] = None
    ) -> None:
        """
        Parameters
        ----------
        types : List[int]
            Specifies the number of categorical values of an input dimension where
            the i-th entry corresponds to the i-th input dimension. Let's say we
            have 2 dimension where the first dimension consists of 3 different
            categorical choices and the second dimension is continuous than we
            have to pass [3, 0]. Note that we count starting from 0.
        bounds : List[Tuple[float, float]]
            bounds of input dimensions: (lower, uppper) for continuous dims; (n_cat, np.nan) for categorical dims
        seed : int
            The seed that is passed to the random_forest_run library.
        log_y: bool
            y values (passed to this RF) are expected to be log(y) transformed;
            this will be considered during predicting
        num_trees : int
            The number of trees in the random forest.
        do_bootstrapping : bool
            Turns on / off bootstrapping in the random forest.
        n_points_per_tree : int
            Number of points per tree. If <= 0 X.shape[0] will be used
            in _train(X, y) instead
        ratio_features : float
            The ratio of features that are considered for splitting.
        min_samples_split : int
            The minimum number of data points to perform a split.
        min_samples_leaf : int
            The minimum number of data points in a leaf.
        max_depth : int
            The maximum depth of a single tree.
        eps_purity : float
            The minimum difference between two target values to be considered
            different
        max_num_nodes : int
            The maxmimum total number of nodes in a tree
        instance_features : np.ndarray (I, K)
            Contains the K dimensional instance features of the I different instances
        pca_components : float
            Number of components to keep when using PCA to reduce dimensionality of instance features. Requires to
            set n_feats (> pca_dims).
        """

        list_of_params_to_remove = ['dataset_id', 'unbalance_data', 'fraction_ids_class0', 'fraction_ids_class1',
                                    'sampling_factor_train_only']
        list_of_params_to_remove_ids = []
        for param_name in list_of_params_to_remove:
            list_of_params_to_remove_ids.append(configspace._hyperparameter_idx[param_name])
        print(list_of_params_to_remove_ids)

        self.mask_remove = np.ones(len(configspace._hyperparameters) + 34, dtype=bool)
        self.mask_remove[list_of_params_to_remove_ids] = False

        new_bounds = copy.deepcopy(bounds)
        for nb_new in range(34):
            new_bounds.append((0, np.nan))
        nn_bounds = []
        for bi in range(len(new_bounds)):
            if self.mask_remove[bi]:
                nn_bounds.append(new_bounds[bi])
        new_bounds = nn_bounds

        new_types = copy.deepcopy(types)
        for nb_new in range(34):
            new_types.append(0)

        nn_types = []
        for ti in range(len(new_types)):
            if self.mask_remove[ti]:
                nn_types.append(new_types[ti])
        new_types = nn_types

        save_dict = {}
        save_dict['configspace'] = configspace
        save_dict['types'] = types
        save_dict['bounds'] = bounds
        save_dict['seed'] = seed
        save_dict['log_y'] = log_y
        save_dict['num_trees'] = num_trees
        save_dict['do_bootstrapping'] = do_bootstrapping
        save_dict['n_points_per_tree'] = n_points_per_tree
        save_dict['ratio_features'] = ratio_features
        save_dict['min_samples_split'] = min_samples_split
        save_dict['min_samples_leaf'] = min_samples_leaf
        save_dict['max_depth'] = max_depth
        save_dict['eps_purity'] = eps_purity
        save_dict['max_num_nodes'] = max_num_nodes
        save_dict['instance_features'] = instance_features
        save_dict['pca_components'] = pca_components

        with open('/tmp/smac_conf.p', "wb") as pickle_model_file:
            pickle.dump(save_dict, pickle_model_file)



        super().__init__(
            configspace=configspace,
            types=new_types,
            bounds=new_bounds,
            seed=seed,
            instance_features=instance_features,
            pca_components=pca_components,
        )

        self.log_y = log_y
        self.rng = regression.default_random_engine(seed)

        self.rf_opts = regression.forest_opts()
        self.rf_opts.num_trees = num_trees
        self.rf_opts.do_bootstrapping = do_bootstrapping
        max_features = 0 if ratio_features > 1.0 else \
            max(1, int(len(types) * ratio_features))
        self.rf_opts.tree_opts.max_features = max_features
        self.rf_opts.tree_opts.min_samples_to_split = min_samples_split
        self.rf_opts.tree_opts.min_samples_in_leaf = min_samples_leaf
        self.rf_opts.tree_opts.max_depth = max_depth
        self.rf_opts.tree_opts.epsilon_purity = eps_purity
        self.rf_opts.tree_opts.max_num_nodes = max_num_nodes
        self.rf_opts.compute_law_of_total_variance = False

        self.n_points_per_tree = n_points_per_tree
        self.rf = None  # type: regression.binary_rss_forest

        # This list well be read out by save_iteration() in the solver
        self.hypers = [num_trees, max_num_nodes, do_bootstrapping,
                       n_points_per_tree, ratio_features, min_samples_split,
                       min_samples_leaf, max_depth, eps_purity, self.seed]

        self.t = FeatureTransformations()

    def _train(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestWithInstances':
        """Trains the random forest on X and y.

        Parameters
        ----------
        X : np.ndarray [n_samples, n_features (config + instance features)]
            Input data points.
        y : np.ndarray [n_samples, ]
            The corresponding target values.

        Returns
        -------
        self
        """


        X = self._impute_inactive(X)
        if X.shape[1] != len(self.bounds):
            X = self.featureprocessing(X, catch_exception=True)

        self.X = X
        self.y = y.flatten()

        print('train shape: ' + str(X.shape))

        if self.n_points_per_tree <= 0:
            self.rf_opts.num_data_points_per_tree = self.X.shape[0]
        else:
            self.rf_opts.num_data_points_per_tree = self.n_points_per_tree
        self.rf = regression.binary_rss_forest()
        self.rf.options = self.rf_opts
        data = self._init_data_container(self.X, self.y)
        self.rf.fit(data, rng=self.rng)

        with open('/tmp/smac_model.p', "wb") as pickle_model_file:
            pickle.dump(self.rf, pickle_model_file)

        return self

    def _init_data_container(self, X: np.ndarray, y: np.ndarray) -> regression.default_data_container:
        """Fills a pyrfr default data container, s.t. the forest knows
        categoricals and bounds for continous data

        Parameters
        ----------
        X : np.ndarray [n_samples, n_features]
            Input data points
        y : np.ndarray [n_samples, ]
            Corresponding target values

        Returns
        -------
        data : regression.default_data_container
            The filled data container that pyrfr can interpret
        """
        # retrieve the types and the bounds from the ConfigSpace

        data = regression.default_data_container(X.shape[1])

        for i, (mn, mx) in enumerate(self.bounds):
            if np.isnan(mx):
                data.set_type_of_feature(i, mn)
            else:
                data.set_bounds_of_feature(i, mn, mx)

        for row_X, row_y in zip(X, y):
            data.add_data_point(row_X, row_y)
        return data


    def featureprocessing(self, X, metafeatures=None, catch_exception=False):
        print(X.shape)
        self.t.fit(X)
        X = self.t.transform(X, cs=self.configspace, metafeatures_pre=metafeatures, catch_exception=catch_exception)
        print(X.shape)
        print(len(self.mask_remove))
        X = X[:, self.mask_remove]
        return X

    def _predict(self, X: np.ndarray,
                 cov_return_type: typing.Optional[str] = 'diagonal_cov', metafeatures=None) \
            -> typing.Tuple[np.ndarray, np.ndarray]:
        """Predict means and variances for given X.

        Parameters
        ----------
        X : np.ndarray of shape = [n_samples,
                                   n_features (config + instance features)]
        cov_return_type: typing.Optional[str]
            Specifies what to return along with the mean. Refer ``predict()`` for more information.

        Returns
        -------
        means : np.ndarray of shape = [n_samples, 1]
            Predictive mean
        vars : np.ndarray  of shape = [n_samples, 1]
            Predictive variance
        """
        if len(X.shape) != 2:
            raise ValueError(
                'Expected 2d array, got %dd array!' % len(X.shape))
        #if X.shape[1] != len(self.types):
        #    raise ValueError('Rows in X should have %d entries but have %d!' % (len(self.types), X.shape[1]))
        if cov_return_type != 'diagonal_cov':
            raise ValueError("'cov_return_type' can only take 'diagonal_cov' for this model")

        X = self._impute_inactive(X)

        if X.shape[1] != len(self.bounds):
            X = self.featureprocessing(X, metafeatures)

        if self.log_y:
            all_preds = []
            third_dimension = 0

            # Gather data in a list of 2d arrays and get statistics about the required size of the 3d array
            for row_X in X:
                preds_per_tree = self.rf.all_leaf_values(row_X)
                all_preds.append(preds_per_tree)
                max_num_leaf_data = max(map(len, preds_per_tree))
                third_dimension = max(max_num_leaf_data, third_dimension)

            # Transform list of 2d arrays into a 3d array
            preds_as_array = np.zeros((X.shape[0], self.rf_opts.num_trees, third_dimension)) * np.NaN
            for i, preds_per_tree in enumerate(all_preds):
                for j, pred in enumerate(preds_per_tree):
                    preds_as_array[i, j, :len(pred)] = pred

            # Do all necessary computation with vectorized functions
            preds_as_array = np.log(np.nanmean(np.exp(preds_as_array), axis=2) + VERY_SMALL_NUMBER)

            # Compute the mean and the variance across the different trees
            means = preds_as_array.mean(axis=1)
            vars_ = preds_as_array.var(axis=1)
        else:
            means, vars_ = [], []
            for row_X in X:
                mean_, var = self.rf.predict_mean_var(row_X)
                means.append(mean_)
                vars_.append(var)

        means = np.array(means)
        vars_ = np.array(vars_)

        return means.reshape((-1, 1)), vars_.reshape((-1, 1))

    def predict(self, X: np.ndarray,
                cov_return_type: typing.Optional[str] = 'diagonal_cov') \
            -> typing.Tuple[np.ndarray, typing.Optional[np.ndarray]]:
        """
        Predict means and variances for given X.

        Parameters
        ----------
        X : np.ndarray of shape = [n_samples, n_features (config + instance features)]
            Training samples
        cov_return_type: typing.Optional[str]
            Specifies what to return along with the mean. (Applies to only Gaussian Process for now)
            Can take 4 values: [None, diagonal_std, diagonal_cov, full_cov]
            * None - only mean is returned
            * diagonal_std - standard deviation at test points is returned
            * diagonal_cov - diagonal of the covariance matrix is returned
            * full_cov - whole covariance matrix between the test points is returned

        Returns
        -------
        means : np.ndarray of shape = [n_samples, n_objectives]
            Predictive mean
        vars : None or np.ndarray of shape = [n_samples, n_objectives] or [n_samples, n_samples]
            Predictive variance or standard deviation
        """
        if len(X.shape) != 2:
            raise ValueError('Expected 2d array, got %dd array!' % len(X.shape))
        #if X.shape[1] != self.n_params + self.n_feats:
        #    raise ValueError('Rows in X should have %d entries but have %d!' %
        #                     (self.n_params + self.n_feats, X.shape[1]))

        if self._apply_pca:
            try:
                X_feats = X[:, -self.n_feats:]
                X_feats = self.scaler.transform(X_feats)
                X_feats = self.pca.transform(X_feats)
                X = np.hstack((X[:, :self.n_params], X_feats))
            except NotFittedError:
                pass  # PCA not fitted if only one training sample

        #if X.shape[1] != len(self.types):
        #    raise ValueError('Rows in X should have %d entries but have %d!' % (len(self.types), X.shape[1]))

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Predicted variances smaller than 0. Setting those variances to 0.')
            mean, var = self._predict(X, cov_return_type)

        if len(mean.shape) == 1:
            mean = mean.reshape((-1, 1))
        if var is not None and len(var.shape) == 1:
            var = var.reshape((-1, 1))

        return mean, var

    def predict_marginalized_over_instances(self, X: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance marginalized over all instances.

        Returns the predictive mean and variance marginalised over all
        instances for a set of configurations.

        Note
        ----
        This method overwrites the same method of ~smac.epm.base_epm.AbstractEPM;
        the following method is random forest specific
        and follows the SMAC2 implementation;
        it requires no distribution assumption
        to marginalize the uncertainty estimates

        Parameters
        ----------
        X : np.ndarray
            [n_samples, n_features (config)]

        Returns
        -------
        means : np.ndarray of shape = [n_samples, 1]
            Predictive mean
        vars : np.ndarray  of shape = [n_samples, 1]
            Predictive variance
        """

        X = self._impute_inactive(X)
        if X.shape[1] != len(self.bounds):
            X = self.featureprocessing(X, catch_exception=True)

        if self.instance_features is None or \
                len(self.instance_features) == 0:
            mean_, var = self.predict(X)
            assert var is not None  # please mypy

            var[var < self.var_threshold] = self.var_threshold
            var[np.isnan(var)] = self.var_threshold
            return mean_, var

        if len(X.shape) != 2:
            raise ValueError(
                'Expected 2d array, got %dd array!' % len(X.shape))
        if X.shape[1] != len(self.bounds):
            raise ValueError('Rows in X should have %d entries but have %d!' %
                             (len(self.bounds),
                              X.shape[1]))

        #X = self._impute_inactive(X)


        dat_ = np.zeros((X.shape[0], self.rf_opts.num_trees))  # marginalized predictions for each tree
        for i, x in enumerate(X):

            # marginalize over instances
            # 1. get all leaf values for each tree
            preds_trees = [[] for i in range(self.rf_opts.num_trees)]  # type: typing.List[typing.List[float]]

            for feat in self.instance_features:
                x_ = np.concatenate([x, feat])
                preds_per_tree = self.rf.all_leaf_values(x_)
                for tree_id, preds in enumerate(preds_per_tree):
                    preds_trees[tree_id] += preds

            # 2. average in each tree
            if self.log_y:
                for tree_id in range(self.rf_opts.num_trees):
                    dat_[i, tree_id] = \
                        np.log(np.exp(np.array(preds_trees[tree_id])).mean())
            else:
                for tree_id in range(self.rf_opts.num_trees):
                    dat_[i, tree_id] = np.array(preds_trees[tree_id]).mean()

        # 3. compute statistics across trees
        mean_ = dat_.mean(axis=1)
        var = dat_.var(axis=1)

        var[var < self.var_threshold] = self.var_threshold

        if len(mean_.shape) == 1:
            mean_ = mean_.reshape((-1, 1))
        if len(var.shape) == 1:
            var = var.reshape((-1, 1))

        return mean_, var
