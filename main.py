# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 12:24:39 2024

@author: lenovo
"""
import tensorflow as tf
print(tf.__version__)

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Check TensorFlow version
print("TensorFlow Version:", tf.__version__)

# Part 1 - Data Preprocessing
# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values  # Selecting relevant columns
y = dataset.iloc[:, -1].values    # Target variable

# Encoding categorical data
# Label Encoding the "Gender" column
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])  # Female=0, Male=1

# One Hot Encoding the "Geography" column
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Feature Scaling
sc = StandardScaler()
X = sc.fit_transform(X)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Part 2 - Building the ANN
# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
# Units: Number of neurons, Activation: ReLU (Rectified Linear Unit)
ann.add(tf.keras.layers.Dense(units=12, activation='relu', input_dim=X.shape[1]))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=8, activation='relu'))

# Adding the output layer
# Units=1 for binary classification, Activation=sigmoid for probability output
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the ANN
# Compiling the ANN
# Optimizer: Adam (Adaptive Moment Estimation)
# Loss: Binary Crossentropy for binary classification
# Metrics: Accuracy to evaluate during training
ann.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
            loss='binary_crossentropy', 
            metrics=['accuracy'])

# Training the ANN on the Training set
# Epochs: Number of times the model will iterate over the dataset
history = ann.fit(X_train, y_train, batch_size=32, epochs=20, validation_split=0.1, verbose=1)

# Part 4 - Making predictions and evaluating the model
# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)  # Convert probabilities to binary output

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Additional metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Accuracy on Test Set:", accuracy)
print("\nClassification Report:\n", report)

# Optional: Visualize Training Performance
import matplotlib.pyplot as plt

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
