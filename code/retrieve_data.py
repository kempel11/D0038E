import pandas as pd

data_colums = ['day', 'month', 'year', 'Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'Classes', 'Region']
data_set = pd.read_csv("data/Algerian_forest_fires_dataset_FIXED.csv",sep=";",header=0, names=data_colums)

print(data_set.head(15), "\n")
print(data_set.describe())