import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Create columns and import data
def getData():
    data_colums = ['day', 'month', 'year', 'Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'Classes', 'Region']
    data_set = pd.read_csv("data/Algerian_forest_fires_dataset_FIXED.csv",sep=";",header=0, names=data_colums)
    return data_set

data_set = getData()

# Create a datetime column
data_set["date"] = pd.to_datetime(data_set[["year", "month", "day"]])
data_set["Classes"] = data_set["Classes"].astype("category")
data_set["Fire"] = data_set["Classes"].map({"fire": 1, "not fire": 0})
data_set["Region"] = data_set["Region"].map({"Bejaia": 1, "Sidi-Bel Abbes": 0})
data_set["month"] = pd.to_datetime(data_set[["year","month","day"]]).dt.month

# Remove unused data
data_set.drop(['Classes','year'], axis=1, inplace=True)

# K-means
features = ["Temperature", "RH", "Ws", "Rain", "FFMC", "DMC", "DC", "ISI", "BUI", "FWI", "Fire"]
X = data_set[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels
data_set["Cluster"] = clusters
print(data_set["Cluster"].value_counts())

# minimum and maximum allowed values 
min_max_values = {
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

# prints out current values and their limits
def check_max_min_values():
    data_to_check = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC','ISI','BUI','FWI']
    for a in data_to_check:
        print(a)
        print("Curr min", min(data_set[a]) ,"min:", min_max_values[a][0]) 
        print("Curr max", max(data_set[a]) ,"max:", min_max_values[a][1], '\n')

#check_max_min_values()

# Transforms data to be inside given ranges
def normalize():
    data_to_check = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC','ISI','BUI','FWI']
    for a in data_to_check:
        #print(a)
        #print("Curr min", min(data_set[a]) ,"min:", min_max_values[a][0]) 
        #print("Curr max", max(data_set[a]) ,"max:", min_max_values[a][1], '\n')
        minmax=preprocessing.MinMaxScaler(feature_range=(min_max_values[a][0],min_max_values[a][1]))
        data_set[a] = minmax.fit_transform(data_set[[a]])
        #print("Norm min", min(data_set[a]) ,"min:", min_max_values[a][0]) 
        #print("Norm max", max(data_set[a]) ,"max:", min_max_values[a][1], '\n')

normalize()

# Generates histograms of the given columms
def generate_histogram(column):
    for c in column:
        plt.figure()
        data_set[c].hist(color='r')
        plt.title(c)
        plt.ylabel("frequency")
        plt.xlabel(c)
        plt.grid(False)
        plt.savefig("figure/"+c+"_histogram.png", dpi=300, bbox_inches="tight")
        #plt.close()

#generate_histogram(['Region','Fire', 'Temperature' ,'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI'])

# Plot temperature over time
def temp_over_time():
    plt.figure(figsize=(12,6))
    plt.plot(data_set["date"], data_set["Temperature"], label="Temperature (°C)", color="red")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.title("Daily Temperature in Algerian Forest Fires Dataset")
    plt.legend()
    plt.grid(True)
    plt.savefig("temperature_plot.png", dpi=300, bbox_inches="tight")
    #plt.show()
    #plt.close()

#temp_over_time()

#Heatmap
def heatmap():
    for cluster_id in sorted(data_set["Cluster"].unique()):
        subset = data_set[data_set["Cluster"] == cluster_id]
        corr = subset[features].corr()
    
        plt.figure(figsize=(12,8))
        sns.heatmap(corr, annot=False, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title(f"Correlation Heatmap for Cluster {cluster_id}")
        plt.savefig("heatmap_kmeans.png", dpi=300, bbox_inches="tight")
        plt.close()

    plt.figure(figsize=(12,8))
    corr = subset[features].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Fire-related Variables")
    plt.savefig("heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

#heatmap()

#Scatterplot Temp/RH
def scatterplot_temp_RH():
    plt.figure(figsize=(10,6))
    sns.scatterplot(x="Temperature", y="RH", hue="Classes", data=data_set, alpha=0.7, palette={"fire": "red", "not fire": "blue"})
    plt.title("K-Means Clusters (k=2) on Temperature vs Humidity")
    plt.savefig("scatterplot_kmeans.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10,6))
    sns.scatterplot(x="Temperature", y="RH", hue="Classes", data=data_set, alpha=0.7, palette={"fire": "red", "not fire": "blue"})
    plt.title("Temperature vs Relative Humidity (Fire vs No Fire)")
    plt.savefig("scatterplot.png", dpi=300, bbox_inches="tight")
    plt.close()

#scatterplot_temp_RH()

# Boxplot Fire/no fire
def boxplot_fnof():
    plt.figure(figsize=(12,6))
    sns.boxplot(x="Fire", y="Temperature", data=data_set, palette="coolwarm")
    plt.title("Temperature Distribution (Fire vs No Fire)")
    plt.xlabel("Fire (1=Yes, 0=No)")
    plt.ylabel("Temperature")
    plt.savefig("boxplot_temp.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12,6))
    sns.boxplot(x="Fire", y="RH", data=data_set, palette="coolwarm")
    plt.title("Relative Humidity Distribution (Fire vs No Fire)")
    plt.xlabel("Fire (1=Yes, 0=No)")
    plt.ylabel("Relative Humidity (%)")
    plt.savefig("boxplot_hum.png", dpi=300, bbox_inches="tight")
    plt.close()
#boxplot_fnof()

# FWI Distribution plot
def displot_FWI():
    plt.figure(figsize=(12,6))
    sns.kdeplot(data=data_set, x="FWI", hue="Fire", fill=True, common_norm=False, palette="coolwarm", alpha=0.6)
    plt.title("FWI Distribution for Fire vs No Fire")
    plt.xlabel("FWI")
    plt.ylabel("Density")
    plt.savefig("displot_FWI.png", dpi=300, bbox_inches="tight")
    plt.close()
#displot_FWI()