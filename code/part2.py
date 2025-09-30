import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
import plot_generator
import table_generator
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

RANDOM = 1

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

# Split data into 70,15,15 
# Stratify is used so there is an close to equal ammount of fire/not fire in each dataset
def split_data():
    train_data, temp_data = train_test_split(raw_data, train_size=0.7, random_state=RANDOM, stratify=raw_data['Fire'])
    vali_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=RANDOM, stratify=temp_data['Fire'])

    # Seperate label from data
    train_label = train_data['Fire'].copy()
    vali_label = vali_data['Fire'].copy()
    test_label = test_data['Fire'].copy()

    train_data = train_data.drop(['Fire'], axis=1)
    vali_data = vali_data.drop(['Fire'], axis=1)
    test_data = test_data.drop(['Fire'], axis=1)

    return train_data,train_label, vali_data,vali_label, test_data,test_label

#print(train_data.describe(), "\n")
#print(vali_data.describe(), "\n")
#print(test_data.describe(), "\n")

#print(train_data['Fire'].describe(), "\n")
#print(vali_data['Fire'].describe(), "\n")
#print(test_data['Fire'].describe(), "\n")

def train_decision_tree():
    clf = tree.DecisionTreeClassifier(random_state=42)
    clf.fit(train_data,train_label)

    predict = clf.predict(test_data)

    report = classification_report(test_label,predict, target_names=['Not Fire','Fire'], output_dict=True)
    matrix = confusion_matrix(test_label,predict)
    
    table_generator.classification_report_table(report, "decision tree")
    table_generator.confusion_table(matrix, "decision tree")
    plot_generator.display_confusion_matrix(matrix, "decision_tree")

normalize_min_max_scaler()
train_data, train_label, vali_data, vali_label, test_data, test_label = split_data()

train_decision_tree()


def train_SVM():
    clf = SVC(random_state = 42)
    clf.fit(train_data, train_label)

    predict = clf.predict(test_data)

    report = classification_report(test_label, predict, target_names=['Not Fire','Fire'], output_dict=True)
    matrix = confusion_matrix(test_label,predict)

    table_generator.classification_report_table(report, "SVM")
    table_generator.confusion_table(matrix, "SVM")
    plot_generator.display_confusion_matrix(matrix, "SVM")

train_SVM()


def train_random_forest():
    clf = RandomForestClassifier(n_estimators=10, random_state = 42, oob_score=True)
    clf.fit(train_data, train_label)

    predict = clf.predict(test_data)

    report = classification_report(test_label, predict, target_names=['Not Fire','Fire'], output_dict=True)
    matrix = confusion_matrix(test_label,predict)

    table_generator.classification_report_table(report, "Random Forest")
    table_generator.confusion_table(matrix, "Random Forest")
    plot_generator.display_confusion_matrix(matrix, "Random Forest")

train_random_forest()


def train_knn():
    clf = KNeighborsClassifier(n_neighbors = 42)
    clf.fit(train_data, train_label)

    predict = clf.predict(test_data)

    report = classification_report(test_label, predict, target_names=['Not Fire','Fire'], output_dict=True)
    matrix = confusion_matrix(test_label,predict)

    table_generator.classification_report_table(report, "kNN")
    table_generator.confusion_table(matrix, "kNN")
    plot_generator.display_confusion_matrix(matrix, "kNN")

train_knn()