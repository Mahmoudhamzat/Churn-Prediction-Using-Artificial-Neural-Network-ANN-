# Churn-Prediction-Using-Artificial-Neural-Network-ANN-
# Churn Prediction Using Artificial Neural Network (ANN)

This project focuses on predicting customer churn using an Artificial Neural Network (ANN). The model is trained to predict whether a customer will leave a service (churn) based on various customer attributes such as their demographic information, account information, and interaction with the company.

## Project Overview

The dataset used in this project contains information about customers such as their credit score, geography, gender, age, tenure, balance, number of products, whether they have a credit card, if they are active members, estimated salary, and whether they have exited (churned). The goal is to build a predictive model using an Artificial Neural Network to classify customers who will churn.

## Contents

### 1. **Data Preprocessing**
- Loading and cleaning the dataset.
- Encoding categorical features such as "Geography" and "Gender".
- Scaling features to standardize the data.
- Splitting the dataset into training and test sets.

### 2. **Building the Artificial Neural Network (ANN)**
- Using TensorFlow and Keras to build the ANN.
- Adding input, hidden, and output layers.
- Using ReLU activation function for hidden layers and sigmoid for the output layer.

### 3. **Training the Model**
- Compiling the model with the Adam optimizer and binary cross-entropy loss function.
- Training the model on the training dataset with validation.
- Evaluating the model performance on the test dataset.

### 4. **Model Evaluation**
- Predicting churn outcomes using the trained model.
- Evaluating model performance with metrics such as confusion matrix, accuracy, and classification report.

### 5. **Visualization**
- Visualizing the training and validation accuracy and loss curves.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- Scikit-learn
- NumPy
- Pandas
- Matplotlib

## Setup

To set up the project environment, follow these steps:

1. Clone the repository:
    ```bash
    git clone <repository-link>
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Jupyter notebook or Python script to train and evaluate the model.

## Dataset

The dataset `Churn_Modelling.csv` contains the following columns:

- **CustomerId**: Unique ID for each customer.
- **Surname**: Last name of the customer.
- **CreditScore**: The credit score of the customer.
- **Geography**: The country of the customer.
- **Gender**: Gender of the customer (Male or Female).
- **Age**: The age of the customer.
- **Tenure**: The number of years the customer has been with the company.
- **Balance**: The balance in the customer's account.
- **NumOfProducts**: The number of products the customer has with the company.
- **HasCrCard**: Whether the customer has a credit card (1 or 0).
- **IsActiveMember**: Whether the customer is an active member (1 or 0).
- **EstimatedSalary**: The estimated salary of the customer.
- **Exited**: Whether the customer churned (1 if the customer exited, 0 otherwise).

## How to Run the Code

1. Make sure all necessary libraries are installed.
2. Place the dataset file `Churn_Modelling.csv` in the project directory.
3. Run the provided script to train the model and evaluate its performance.
