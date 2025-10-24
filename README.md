# KNN (K Nearest Neighbors) Classification: Machine Tutorial Using Python Sklearn

This notebook demonstrates how to perform K Nearest Neighbors (KNN) classification using the scikit-learn library in Python. It covers loading and exploring datasets, implementing KNN from scratch (for illustrative purposes), using scikit-learn's `KNeighborsClassifier`, hyperparameter tuning with `GridSearchCV`, and evaluating the model using confusion matrices and classification reports.

## Datasets Used

- **Iris Dataset:** Used for initial exploration and visualization of KNN concepts.
- **Digits Dataset:** Used for a complete KNN classification task, including hyperparameter tuning and evaluation.

## Project Steps

The notebook follows these steps:

1.  **Loading and Exploring the Iris Dataset:**
    -   Loading the dataset using `sklearn.datasets.load_iris`.
    -   Examining feature names and target names.
    -   Creating a pandas DataFrame for easier manipulation.
    -   Adding target and flower name columns to the DataFrame.
    -   Visualizing Sepal Length vs Sepal Width and Petal Length vs Petal Width for different iris species.

2.  **Implementing KNN from Scratch (Illustrative):**
    -   Defining a function to calculate Euclidean distance.
    -   Implementing a basic KNN prediction function.
    -   Evaluating the custom KNN implementation.
    -   Visualizing the decision boundary for the custom KNN.
    -   Plotting Accuracy vs k for the custom KNN.

3.  **KNN Classification using Scikit-learn (Digits Dataset):**
    -   Loading the digits dataset using `sklearn.datasets.load_digits`.
    -   Splitting the dataset into training and testing sets.
    -   Using `GridSearchCV` to find the optimal number of neighbors (`k`).
    -   Training the `KNeighborsClassifier` with the best `k`.
    -   Evaluating the model's performance.

4.  **Model Evaluation:**
    -   Generating and plotting a Confusion Matrix to visualize the model's predictions against the actual values.
    -   Printing a Classification Report to show precision, recall, f1-score, and support for each class.

