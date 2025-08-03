import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

data = pd.read_csv('Travel_Interest.csv')

# Encode categorical variables
travel_type_map = {t: i for i, t in enumerate(data['Travel_Type'].unique())}
destination_map = {d: i for i, d in enumerate(data['Destination'].unique())}

data['Travel_Type_encoded'] = data['Travel_Type'].map(travel_type_map)
data['Destination_encoded'] = data['Destination'].map(destination_map)

X = data[['Age', 'Gender', 'Travel_Type_encoded']]
y = data['Destination_encoded']

model = DecisionTreeClassifier()
model.fit(X, y)

joblib.dump(model, 'travel_model.pkl')
joblib.dump(destination_map, 'destination_map.pkl')
joblib.dump(travel_type_map, 'travel_type_map.pkl')

print("Model and mappings saved successfully.")