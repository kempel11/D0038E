import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
import plot_generator
import table_generator
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

RANDOM = 42

table_generator.prep()

data_colums = ['day', 'month', 'year', 'Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'Classes', 'Region']
raw_data = pd.read_csv("data/Algerian_forest_fires_dataset_FIXED.csv",sep=";",header=0, names=data_colums)

raw_data["Classes"] = raw_data["Classes"].astype("category")
raw_data["Fire"] = raw_data["Classes"].map({"fire": 1, "not fire": 0})
raw_data["Region"] = raw_data["Region"].map({"Bejaia": 1, "Sidi-Bel Abbes": 0})

# Remove unused data
raw_data.drop(['Classes','year'], axis=1, inplace=True)

MIN_MAX_VALUES = {
    "Temperature": (22, 42),
    "RH": (21,90),
    "Ws": (6,29),
    "Rain":(0,16.8),
    "FFMC":(28.6,92.5),
    "DMC":(1.1,65.9),
    "DC":(7,220.4),
    "ISI":(0,18.5),
    "BUI":(1.1,68),
    "FWI":(0,31.1)
}

def normalize_min_max_scaler():
    data_to_check = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC','ISI','BUI','FWI']
    for a in data_to_check:
        minmax=preprocessing.MinMaxScaler(feature_range=(MIN_MAX_VALUES[a][0],MIN_MAX_VALUES[a][1]))
        raw_data[a] = minmax.fit_transform(raw_data[[a]])

# Split data into 70,30
# Stratify is used so there is an close to equal ammount of fire/not fire in each dataset
def split_data_70_30():
    train_data, test_data = train_test_split(raw_data, train_size=0.7, random_state=RANDOM, stratify=raw_data['Fire'])

    # Seperate label from data
    train_label = train_data['Fire'].copy()
    test_label = test_data['Fire'].copy()

    # Drop label from data
    train_data = train_data.drop(['Fire'], axis=1)
    test_data = test_data.drop(['Fire'], axis=1)

    return train_data,train_label, test_data,test_label

def train_decision_tree():
    clf = tree.DecisionTreeClassifier(random_state=RANDOM)
    params = {
        'criterion':['gini', 'entropy', 'log_loss'],
        'splitter':['best','random'],
        'max_features':['sqrt','log2',None]
    }
    grid_search = GridSearchCV(estimator=clf, param_grid=params, scoring='recall', verbose=1)
    grid_search.fit(train_data, train_label)

    final_model = grid_search.best_estimator_
    print(grid_search.best_params_)
    predict =  final_model.predict(test_data)

    report = classification_report(test_label,predict, target_names=['Not Fire','Fire'], output_dict=True)
    matrix = confusion_matrix(test_label,predict)
    
    table_generator.classification_report_table(report, "decision tree")
    table_generator.confusion_table(matrix, "decision tree")
    plot_generator.display_confusion_matrix(matrix, "decision_tree")

    return report, matrix

def train_SVM():
    clf = SVC(random_state = RANDOM)
    params = {
        'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma':['scale'],
        'decision_function_shape':['ovo','ovr'],
    }
    grid_search = GridSearchCV(estimator=clf, param_grid=params, scoring='recall', verbose=1)
    grid_search.fit(train_data, train_label)

    final_model = grid_search.best_estimator_
    print(grid_search.best_params_)
    predict =  final_model.predict(test_data)

    report = classification_report(test_label, predict, target_names=['Not Fire','Fire'], output_dict=True)
    matrix = confusion_matrix(test_label,predict)

    table_generator.classification_report_table(report, "SVM")
    table_generator.confusion_table(matrix, "SVM")
    plot_generator.display_confusion_matrix(matrix, "SVM")

    return report,matrix

def train_random_forest():
    clf = RandomForestClassifier(n_estimators=10, random_state = RANDOM, oob_score=True)
    clf.fit(train_data, train_label)

    predict = clf.predict(test_data)

    report = classification_report(test_label, predict, target_names=['Not Fire','Fire'], output_dict=True)
    matrix = confusion_matrix(test_label,predict)

    report = classification_report(test_label, predict, target_names=['Not Fire','Fire'], output_dict=True)
    matrix = confusion_matrix(test_label,predict)

    table_generator.classification_report_table(report, "Random Forest")
    table_generator.confusion_table(matrix, "Random Forest")
    plot_generator.display_confusion_matrix(matrix, "Random Forest")

    return report, matrix

def train_knn():
    clf = KNeighborsClassifier(n_neighbors = 42)
    params = {
        'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
        'weights':['uniform','distance',None]
    }
    grid_search = GridSearchCV(estimator=clf, param_grid=params, scoring='recall', verbose=1)
    grid_search.fit(train_data, train_label)

    final_model = grid_search.best_estimator_
    print(grid_search.best_params_)
    predict =  final_model.predict(test_data)

    report = classification_report(test_label, predict, target_names=['Not Fire','Fire'], output_dict=True)
    matrix = confusion_matrix(test_label,predict)

    table_generator.classification_report_table(report, "kNN")
    table_generator.confusion_table(matrix, "kNN")
    plot_generator.display_confusion_matrix(matrix, "kNN")

    return report, matrix

def train_mlp():
    clf = MLPClassifier(max_iter=2000, random_state=RANDOM)
    params = {
        'activation':['identity', 'logistic', 'tanh', 'relu'],
        'solver':['lbfgs','sgd','adam'],
    }
    grid_search = GridSearchCV(estimator=clf, param_grid=params, scoring='recall', verbose=1)
    grid_search.fit(train_data, train_label)

    final_model = grid_search.best_estimator_
    print(grid_search.best_params_)
    predict =  final_model.predict(test_data)

    report = classification_report(test_label, predict, target_names=['Not Fire','Fire'], output_dict=True)
    matrix = confusion_matrix(test_label,predict)

    table_generator.classification_report_table(report, "MLP")
    table_generator.confusion_table(matrix, "MLP")
    plot_generator.display_confusion_matrix(matrix, "MLP")

    return report, matrix

def train_ada_boost():
    clf = AdaBoostClassifier(random_state=RANDOM)
    clf.fit(train_data, train_label)

    predict = clf.predict(test_data)

    report = classification_report(test_label, predict, target_names=['Not Fire','Fire'], output_dict=True)
    matrix = confusion_matrix(test_label,predict)

    table_generator.classification_report_table(report, "AdaBoost")
    table_generator.confusion_table(matrix, "AdaBoost")
    plot_generator.display_confusion_matrix(matrix, "AdaBoost")

    return report, matrix

def train_gradient_boost():
    clf = GradientBoostingClassifier(random_state=RANDOM)
    params = {
        'loss':['log_loss', 'exponential'],
        'criterion':['friedman_mse','squared_error'],
        'max_features':['sqrt', 'log2', None]
    }
    grid_search = GridSearchCV(estimator=clf, param_grid=params, scoring='recall', verbose=1)
    grid_search.fit(train_data, train_label)

    final_model = grid_search.best_estimator_
    print(grid_search.best_params_)
    predict =  final_model.predict(test_data)

    report = classification_report(test_label, predict, target_names=['Not Fire','Fire'], output_dict=True)
    matrix = confusion_matrix(test_label,predict)

    table_generator.classification_report_table(report, "Gradient Boost")
    table_generator.confusion_table(matrix, "Gradient Boost")
    plot_generator.display_confusion_matrix(matrix, "Gradient Boost")

    return report, matrix

def evaluation_plots():
    reports = [report_decision_tree, report_svm, report_random_forest, report_knn, report_mlp, report_ada, report_gradient]
    names = ["Decision Tree", "SVM", "Random Forest", "k-NN", "MLP", "AdaBoost", "Gradient Boost"]
    plot_generator.accuracy_comparison(reports, names)
    plot_generator.f1_score_comparison(reports, names)
    plot_generator.recall_score_comparison(reports, names)

normalize_min_max_scaler()
train_data, train_label, test_data, test_label = split_data_70_30()

report_decision_tree, matrix_decision_tree = train_decision_tree()
report_svm, matrix_svm = train_SVM()
report_random_forest, matrix_random_forest = train_random_forest()
report_mlp, matrix_mlp = train_mlp()
report_ada, matrix_ada = train_ada_boost()
report_gradient, matrix_gradient = train_gradient_boost()
report_knn, matrix_knn = train_knn()

evaluation_plots()

print("Done")
