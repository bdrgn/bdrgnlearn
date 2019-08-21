import numpy as np
from collections import Counter


class KNeighborsClassifier:

    def __init__(self, n_neighbors=5):
        self.n_neighbors_ = n_neighbors

    def fit(self, X, y):
        self.X_train_ = X
        self.y_train_ = y

    def predict(self, X):
        # Initialize an empty numpy array to store predictions
        predictions = np.empty(len(X))

        for i in range(len(X)):
            # Select an observation of features array
            obs = X[i, :]

            # Calculate distances from observation data to every data point in the training set features arrray
            distances = np.sqrt(np.sum(np.power(self.X_train_ - obs, 2), axis=1))

            # Get the classes of nearest neighbors
            nearest_neighbors = [x[1] for x in sorted(zip(distances, self.y_train_), key=lambda x: x[0])][:self.n_neighbors_]

            # Find the most frequent class
            mean_label = sorted(Counter(nearest_neighbors).items(), key=lambda x: x[1])[0][0]

            # Add the prediction to predictions array
            predictions[i] = mean_label

        # Return the predicted values
        return predictions


class KNeighborsRegressor:

    def __init__(self, n_neighbors=5):
        self.n_neighbors_ = n_neighbors

    def fit(self, X, y):
        self.X_train_ = X
        self.y_train_ = y

    def predict(self, X):
        # Initialize an empty numpy array to store predictions
        predictions = np.empty(len(X))

        for i in range(len(X)):
            # Select an observation of features array
            obs = X[i, :]

            # Calculate distances from observation data to every data point in the training set features arrray
            distances = np.sqrt(np.sum(np.power(self.X_train_ - obs, 2), axis=1))

            # Get the classes of nearest neighbors
            nearest_neighbors = [x[1] for x in sorted(zip(distances, self.y_train_), key=lambda x: x[0])][:self.n_neighbors_]

            # Find the most frequent class
            mean_label = np.mean(nearest_neighbors)

            # Add the prediction to predictions array
            predictions[i] = mean_label

        # Return the predicted values
        return predictions