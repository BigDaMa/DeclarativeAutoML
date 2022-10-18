import pickle
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import get_feature_names
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import get_data
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import data2features
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.Space_GenerationTreeBalance import SpaceGenerator
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.feature_transformation.FeatureTransformations import FeatureTransformations
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import space2features
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import ifNull
from anytree import Node

test_holdout_dataset_id = 75097
X_train_hold, X_test_hold, y_train_hold, y_test_hold, categorical_indicator_hold, attribute_names_hold = get_data('data', randomstate=42, task_id=test_holdout_dataset_id)
metafeature_values_hold = data2features(X_train_hold, y_train_hold, categorical_indicator_hold)

gen = SpaceGenerator()
space = gen.generate_params()


my_list = list(space.name2node.keys())
my_list.sort()
print('len list:' + str(len(my_list)))

def generate_constraints(node):
    all_constraints = ''
    for child in node.children:
        if node.name in my_list:
            all_constraints += 'm.Equation(x[' + str(my_list.index(node.name)) + ']>=x[' + str(my_list.index(child.name)) + '])\n'
        all_constraints += generate_constraints(child)
    return all_constraints

all_constraints = ''
print(generate_constraints(space.parameter_tree))

X = pickle.load(open('/home/felix/phd2/dec_automl/oct16_1day_al/felix_X_compare_scaled.p', "rb"))
y = pickle.load(open('/home/felix/phd2/dec_automl/oct16_1day_al/felix_y_compare_scaled.p', "rb"))
groups = pickle.load(open('/home/felix/phd2/dec_automl/oct16_1day_al/felix_group_compare_scaled.p', "rb"))

my_list_constraints = ['global_search_time_constraint',
                           'global_evaluation_time_constraint',
                           'global_memory_constraint',
                           'global_cv',
                           'global_number_cv',
                           'privacy',
                           'hold_out_fraction',
                           'sample_fraction',
                           'training_time_constraint',
                           'inference_time_constraint',
                           'pipeline_size_constraint',
                           'fairness_constraint',
                           'use_ensemble',
                           'use_incremental_data',
                           'shuffle_validation'
                       ]


static_names = ['global_search_time_constraint',
                           'global_evaluation_time_constraint',
                           'global_memory_constraint',
                           'global_cv',
                           'global_number_cv',
                           'privacy',
                           'sample_fraction',
                           'training_time_constraint',
                           'inference_time_constraint',
                           'pipeline_size_constraint',
                           'fairness_constraint',
                           'product_cvs',
                           'sampled_instances',
                           'number_of_evaluations',
                           'log_search_time_constraint',
                           'log_evaluation_time_constraint',
                           'log_search_memory_constraint',
                           'log_privacy_constraint',
                           'log_sampled_instances',
                           'has_privacy_constraint',
                           'has_evaluation_time_constraint']



_, feature_names = get_feature_names(my_list_constraints)

search_time = 60 * 5
evaluation_time = 0.1 * search_time
memory_limit = 10
cv = 1
number_of_cvs = 1
privacy_limit = None
sample_fraction = 1.0


training_time_limit = None
inference_time_limit = None
pipeline_size_limit = None
fairness_limit = None


hold_out_fraction = -1
use_ensemble = -1
use_incremental_data = -1
shuffle_validation = -1


my_list_constraints_values = [search_time,
                                      evaluation_time,
                                      memory_limit,
                                      cv,
                                      number_of_cvs,
                                      ifNull(privacy_limit, constant_value=1000),
                                      hold_out_fraction,
                                      sample_fraction,
                                      ifNull(training_time_limit, constant_value=search_time),
                                      ifNull(inference_time_limit, constant_value=60),
                                      ifNull(pipeline_size_limit, constant_value=350000000),
                                      ifNull(fairness_limit, constant_value=0.0),
                                      int(use_ensemble),
                                      int(use_incremental_data),
                                      int(shuffle_validation)
                                      ]

features = space2features(space, my_list_constraints_values, metafeature_values_hold)
_, feature_names = get_feature_names(my_list_constraints)
features = FeatureTransformations().fit(features).transform(features, feature_names=feature_names)

hold_out_test_instances = 1.0 * features[0, feature_names.index('NumberOfInstances')] * features[0, feature_names.index('sample_fraction')]
print(hold_out_test_instances)


assert len(feature_names) == X.shape[1]


print(X.shape)

model_success = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1, max_depth=None)
model_success.fit(X, y)
clf = model_success.estimators_[0]

def build_equation(clf):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    value = clf.tree_.value

    #find best value with BO

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    print(
        "The binary tree structure has {n} nodes and has "
        "the following tree structure:\n".format(n=n_nodes)
    )
    child_id_2_parent_id = {}
    for i in range(n_nodes):
        if is_leaves[i]:
            print(
                "{space}node={node} is a leaf node: {value}.".format(
                    space=node_depth[i] * "\t", node=i, value=value[i][0][0]
                )
            )
        else:
            child_id_2_parent_id[children_left[i]] = i
            child_id_2_parent_id[children_right[i]] = i

            print(
                "{space}node={node} is a split node: "
                "go to node {left} if X[:, {feature}] {feature_name} <= {threshold} "
                "else to node {right}.".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature=feature[i],
                    threshold=threshold[i],
                    right=children_right[i],
                    feature_name=feature_names[feature[i]]
                )
            )

    prune_node = np.zeros(n_nodes, dtype=bool)

    print(features[0,311])

    for i in range(n_nodes):
        if prune_node[i] == True and not is_leaves[i]:
            prune_node[children_right[i]] = True
            prune_node[children_left[i]] = True
        else:
            if is_leaves[i]:
                pass
            else:
                if feature_names[feature[i]] in static_names:
                    prune_node[i] = True
                    if features[0, feature[i]] <= threshold[i]:
                        prune_node[children_right[i]] = True
                    else:
                        prune_node[children_left[i]] = True


    for i in range(n_nodes):
        print('node'+str(i) + ': ' + str(prune_node[i]))


    print(prune_node)

    def get_path(child_id):
        my_str = ''
        if child_id in child_id_2_parent_id:
            parent_id = child_id_2_parent_id[child_id]
            #if hyperparameter space var
            if feature_names[feature[parent_id]] in my_list:
                if threshold[parent_id] == 0.5:
                    if children_right[parent_id] == child_id:
                        my_str += '*x[' + str(feature[parent_id]) + ']'
                    else:
                        my_str += '*(1-x[' + str(feature[parent_id]) + '])'

            if feature_names[feature[parent_id]] == 'use_ensemble':
                if threshold[parent_id] == 0.5:
                    if children_right[parent_id] == child_id:
                        my_str += '*x_ensemble'
                    else:
                        my_str += '*(1-x_ensemble)'

            if feature_names[feature[parent_id]] == 'use_incremental_data':
                if threshold[parent_id] == 0.5:
                    if children_right[parent_id] == child_id:
                        my_str += '*x_incremental'
                    else:
                        my_str += '*(1-x_incremental)'

            if feature_names[feature[parent_id]] == 'shuffle_validation':
                if threshold[parent_id] == 0.5:
                    if children_right[parent_id] == child_id:
                        my_str += '*x_shuffle'
                    else:
                        my_str += '*(1-x_shuffle)'

            if feature_names[feature[parent_id]] == 'hold_out_fraction':
                if children_right[parent_id] == child_id:
                    my_str += "*((m.sign3(x_holdout-" + str(threshold[parent_id]) + ") + 1) / 2)"
                else:
                    my_str += "*((m.sign3(" + str(threshold[parent_id]) + "-x_holdout)+1)/2)"

            if feature_names[feature[parent_id]] == 'hold_out_test_instances':
                if children_right[parent_id] == child_id:
                    my_str += "*((m.sign3(x_holdout*"+str(features[0, feature_names.index('NumberOfInstances')] * features[0, feature_names.index('sample_fraction')])+"-" + str(threshold[parent_id]) + ") + 1) / 2)"
                else:
                    my_str += "*((m.sign3(" + str(threshold[parent_id]) + "-x_holdout*"+str(features[0, feature_names.index('NumberOfInstances')] * features[0, feature_names.index('sample_fraction')])+")+1)/2)"

            if feature_names[feature[parent_id]] == 'hold_out_training_instances':
                if children_right[parent_id] == child_id:
                    my_str += "*((m.sign3((1.0-x_holdout)*"+str(features[0, feature_names.index('NumberOfInstances')] * features[0, feature_names.index('sample_fraction')])+"-" + str(threshold[parent_id]) + ") + 1) / 2)"
                else:
                    my_str += "*((m.sign3(" + str(threshold[parent_id]) + "-(1.0-x_holdout)*"+str(features[0, feature_names.index('NumberOfInstances')] * features[0, feature_names.index('sample_fraction')])+")+1)/2)"


            my_str += get_path(parent_id)
        return my_str

    objective_list = []
    for i in range(n_nodes):
        if is_leaves[i] and prune_node[i] == False:
            my_str = str(value[i][0][0])
            objective_list.append( my_str + get_path(i) )
    return objective_list

print('\n\n')
my_str_all = ''
for clf in model_success.estimators_:
    for summmand in build_equation(clf):
        my_str_all  += 'my_array.append('+ str(summmand) +')\n'

print('\n\n')
print(my_str_all)
