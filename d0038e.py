import csv

numbers = []

with open("Algerian_forest_fires_dataset_FIXED.csv", newline="") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
         numbers.append([int(x) for x in row])

         


