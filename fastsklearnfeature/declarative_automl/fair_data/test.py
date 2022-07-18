from fastsklearnfeature.declarative_automl.fair_data.arff2pandas import load
import copy
import numpy as np

map_dataset = {}
map_dataset['31'] = 'foreign_worker@{yes,no}'
map_dataset['802'] = 'sex@{female,male}'
map_dataset['1590'] = 'sex@{Female,Male}'
map_dataset['1461'] = 'AGE@{True,False}'
map_dataset['42193'] = 'race_Caucasian@{0,1}'
map_dataset['1480'] = 'V2@{Female,Male}'
# map_dataset['804'] = 'Gender@{0,1}'
map_dataset['42178'] = 'gender@STRING'
map_dataset['981'] = 'Gender@{Female,Male}'
map_dataset['40536'] = 'samerace@{0,1}'
map_dataset['40945'] = 'sex@{female,male}'
map_dataset['451'] = 'Sex@{female,male}'
# map_dataset['945'] = 'sex@{female,male}'
map_dataset['446'] = 'sex@{Female,Male}'
map_dataset['1017'] = 'sex@{0,1}'
map_dataset['957'] = 'Sex@{0,1}'
map_dataset['41430'] = 'SEX@{True,False}'
map_dataset['1240'] = 'sex@{Female,Male}'
map_dataset['1018'] = 'sex@{Female,Male}'
# map_dataset['55'] = 'SEX@{male,female}'
map_dataset['38'] = 'sex@{F,M}'
map_dataset['1003'] = 'sex@{male,female}'
map_dataset['934'] = 'race@{black,white}'

def get_X_y_id(key='31'):

    def get_class_attribute_name(df):
        for i in range(len(df.columns)):
            if str(df.columns[i]).startswith('class'):
                return str(df.columns[i])

    def get_sensitive_attribute_id(df, sensitive_attribute_name):
        for i in range(len(df.columns)):
            if str(df.columns[i]) == sensitive_attribute_name:
                return i

    with open("/home/neutatz/phd2/fair_data" + "/downloaded_arff/" + str(key) + ".arff") as f:
        df = load(f)

        y = copy.deepcopy(df[get_class_attribute_name(df)])
        X = df.drop(columns=[get_class_attribute_name(df)])

        categorical_features = []
        continuous_columns = []
        for type_i in range(len(X.columns)):
            if X.dtypes[type_i] == object:
                categorical_features.append(type_i)
            else:
                continuous_columns.append(type_i)

        value = map_dataset[key]
        sensitive_attribute_id = get_sensitive_attribute_id(X, value)

        X_datat = X.values
        for x_i in range(X_datat.shape[0]):
            for y_i in range(X_datat.shape[1]):
                if type(X_datat[x_i][y_i]) == type(None):
                    if X.dtypes[y_i] == object:
                        X_datat[x_i][y_i] = 'missing'
                    else:
                        X_datat[x_i][y_i] = np.nan


        missing_mask = X_datat[:,sensitive_attribute_id] != 'missing'
        X_datat_new = X_datat[missing_mask]
        y_new = y[missing_mask]

        categorical_indicator = []
        for type_i in range(len(X.columns)):
            categorical_indicator.append(X.dtypes[type_i] == object)

        from sklearn import preprocessing
        y_new = preprocessing.LabelEncoder().fit_transform(y_new)
        y_new = y_new == 1

        return X_datat_new, y_new, sensitive_attribute_id, categorical_indicator

'''
for key in list(map_dataset.keys()):
    X_datat, y, sensitive_attribute_id, categorical_indicator = get_X_y_id(key=key)
    print('key: ' + str(key) + ' ' + str(sensitive_attribute_id) + str(np.unique(X_datat[:,sensitive_attribute_id])))
'''