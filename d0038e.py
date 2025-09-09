from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
algerian_forest_fires = fetch_ucirepo(id=547) 
  
# data (as pandas dataframes) 
X = algerian_forest_fires.data.features 
y = algerian_forest_fires.data.targets 
  
# metadata 
print(algerian_forest_fires.metadata) 
  
# variable information 
print(algerian_forest_fires.variables) 
