import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_colums = ['day', 'month', 'year', 'Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'Classes', 'Region']
data_set = pd.read_csv("data/Algerian_forest_fires_dataset_FIXED.csv",sep=";",header=0, names=data_colums)

#print(data_set.head(15), "\n")
#print(data_set.describe())

# Create a datetime column
data_set["date"] = pd.to_datetime(data_set[["year", "month", "day"]])
data_set["Classes"] = data_set["Classes"].astype("category")
data_set["Fire"] = data_set["Classes"].map({"fire": 1, "not fire": 0})
data_set["Region"] = data_set["Region"].map({"Bejaia": 1, "Sidi-Bel Abbes": 0})
data_set["month"] = pd.to_datetime(data_set[["year","month","day"]]).dt.month

# Plot temperature over time
plt.figure(figsize=(12,6))
plt.plot(data_set["date"], data_set["Temperature"], label="Temperature (°C)", color="red")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.title("Daily Temperature in Algerian Forest Fires Dataset")
plt.legend()
plt.grid(True)
plt.savefig("temperature_plot.png", dpi=300, bbox_inches="tight")

#Heatmap
plt.figure(figsize=(12,8))
corr = data_set[['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI','Fire', 'Region', 'month']].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Fire-related Variables")
plt.savefig("heatmap.png", dpi=300, bbox_inches="tight")

#Scatterplot Temp/RH
plt.figure(figsize=(10,6))
sns.scatterplot(x="Temperature", y="RH", hue="Classes", data=data_set, alpha=0.7, palette={"fire": "red", "not fire": "blue"})
plt.title("Temperature vs Relative Humidity (Fire vs No Fire)")
plt.savefig("scatterplot.png", dpi=300, bbox_inches="tight")
