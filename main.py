#using logistic regression to tell if the image is vehicle or not
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from skimage.io import imread_collection
import numpy as np
import os
import joblib

vehicle_path="D:/Codes/computer_vision/logistic/archive/data/vehicles"
non_vehicle_path="D:/Codes/computer_vision/logistic/archive/data/non-vehicles"
# Load images
vehicle_images = imread_collection(os.path.join(vehicle_path, '*.png'))
non_vehicle_images = imread_collection(os.path.join(non_vehicle_path, '*.png'))

# Create labels
vehicle_labels = np.ones(len(vehicle_images))
non_vehicle_labels = np.zeros(len(non_vehicle_images))

# Concatenate images and labels
X = np.concatenate([vehicle_images, non_vehicle_images])
y = np.concatenate([vehicle_labels, non_vehicle_labels])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Flatten the images
X_train_flatten = X_train.reshape(X_train.shape[0], -1)
X_test_flatten = X_test.reshape(X_test.shape[0], -1)

# Create and train the logistic regression model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_flatten, y_train)

# Make predictions
y_pred = logreg.predict(X_test_flatten)

joblib.dump(logreg,"vehicle_mode.pkl")
# Evaluate the model
accuracy = metrics.accuracy_score(y_test, y_pred)
logreg.export
print("Accuracy:", accuracy)