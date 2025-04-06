#Sheen Kak 
#CSCI 184 
#Project code 


# libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_absolute_error, mean_squared_error
import pydot
from PIL import Image
import graphviz
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model


data = pd.read_csv( 'train.csv')

def euclidean_distance(lon1, lat1, lon2, lat2):
    return np.sqrt((lon2 - lon1) ** 2 + (lat2 - lat1) ** 2)

data.dropna(inplace=True)

data = data[(data.fare_amount > 0) & (data.passenger_count > 0) & (data.passenger_count <= 6)]

data['trip_distance'] = euclidean_distance(data['pickup_longitude'], data['pickup_latitude'],
                                           data['dropoff_longitude'], data['dropoff_latitude'])

data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
data['pickup_hour'] = data['pickup_datetime'].dt.hour
data['pickup_day'] = data['pickup_datetime'].dt.dayofweek
data['pickup_month'] = data['pickup_datetime'].dt.month
data['pickup_year'] = data['pickup_datetime'].dt.year

features = ['trip_distance', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
            'passenger_count', 'pickup_hour', 'pickup_day', 'pickup_month', 'pickup_year']
X = data[features]
y = data['fare_amount']

y = pd.cut(y, bins=[0, 10, 20, 30, 40, 50, np.inf], labels=[0, 1, 2, 3, 4, 5]).astype(int)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
split_indices = list(kf.split(X))

accuracies = []
precisions = []
recalls = []

for i, (train_index, test_index) in enumerate(split_indices):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    precision_rf = precision_score(y_test, y_pred_rf, average='macro')
    recall_rf = recall_score(y_test, y_pred_rf, average='macro')

    accuracies.append(accuracy_rf)
    precisions.append(precision_rf)
    recalls.append(recall_rf)
    
    print(f'Random Forest {i+1} Accuracy: {accuracy_rf}, Precision: {precision_rf}, Recall: {recall_rf}')
    
    tree = rf_model.estimators_[0]
    export_graphviz(tree, out_file=f'tree_{i+1}.dot', feature_names=features, filled=True, rounded=True, special_characters=True)
    (graph,) = pydot.graph_from_dot_file(f'tree_{i+1}.dot')
    graph.write_png(f'tree_{i+1}.png')

    img = Image.open(f'tree_{i+1}.png')
    img.show()

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.bar(range(1, 6), accuracies, color='skyblue')
plt.xlabel('Random Forest Model')
plt.ylabel('Accuracy')
plt.title('Accuracy of Random Forest Models')

plt.subplot(1, 3, 2)
plt.bar(range(1, 6), precisions, color='lightgreen')
plt.xlabel('Random Forest Model')
plt.ylabel('Precision')
plt.title('Precision of Random Forest Models')

plt.subplot(1, 3, 3)
plt.bar(range(1, 6), recalls, color='lightcoral')
plt.xlabel('Random Forest Model')
plt.ylabel('Recall')
plt.title('Recall of Random Forest Models')

plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
nn_model = Sequential([
    Dense(128, input_dim=len(features), activation='relu'),
    Dense(64, activation='sigmoid'),
    Dense(32, activation='tanh'),
    Dense(1)
])
nn_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
nn_model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=0)
y_pred_nn = nn_model.predict(X_test)
mae_nn = mean_absolute_error(y_test, y_pred_nn)
mse_nn = mean_squared_error(y_test, y_pred_nn)
print(f'Neural Network MAE: {mae_nn}, MSE: {mse_nn}')

plot_model(nn_model, to_file='neural_network.png', show_shapes=True, show_layer_names=True)

plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred_nn, alpha=0.3)
plt.plot([0, max(y_test)], [0, max(y_test)], 'r--')
plt.xlabel('Actual Fare Amount')
plt.ylabel('Predicted Fare Amount')
plt.title('Neural Network: Actual vs Predicted Fare Amount')
plt.show()





