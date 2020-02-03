import numpy as np

from bdrgnlearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

class RandomForestClassifier:
    # Initialize the object entity
    def __init__(self, min_samples_leaf=1, max_depth=None, n_trees=10, threshold=0.5):

        # Store the input parameters of the class
        self.max_depth_ = max_depth
        self.min_samples_leaf_ = min_samples_leaf
        self.n_trees_ = n_trees
        self.threshold_ = threshold

    # Create a method to fit the model
    def fit(self, X, y):

        # Initialize a list to store tree objects
        self.trees = []

        for i in range(self.n_trees_):
            # Initialize single tree object
            tree = DecisionTreeClassifier(max_depth=self.max_depth_, min_samples_leaf=self.min_samples_leaf_,
                                          bootstrap=True, \
                                          bootstrap_coeff=1 / self.n_trees_)

            # Fit the tree to bootstrapped data
            tree.fit(X, y)

            # Append tree to a list
            self.trees.append(tree)

    # Create a method to predict the target variable for the features array
    def predict(self, X):

        # Initialize a list to store the prediction value for each tree
        trees_predictions = np.empty((len(X), len(self.trees)))

        # Iterate over all observations
        for i, tree in enumerate(self.trees):
            trees_predictions[:, i] = tree.predict(X)

            # Take the average of all trees for each observation in dataset
        predictions = np.mean(trees_predictions, axis=1)

        # Get 0 or 1 according to threshold
        predictions = np.float64(predictions >= self.threshold_)

        # Return an array with predictions for each observation
        return predictions


class RandomForestRegressor:
    # Initialize the object entity
    def __init__(self, min_samples_leaf=1, max_depth=None, n_trees=10, threshold=0.5):

        # Store the input parameters of the class
        self.max_depth_ = max_depth
        self.min_samples_leaf_ = min_samples_leaf
        self.n_trees_ = n_trees
        self.threshold_ = threshold

    # Create a method to fit the model
    def fit(self, X, y):

        # Initialize a list to store tree objects
        self.trees = []

        for i in range(self.n_trees_):
            # Initialize single tree object
            tree = DecisionTreeRegressor(max_depth=self.max_depth_, min_samples_leaf=self.min_samples_leaf_,
                                          bootstrap=True, \
                                          bootstrap_coeff=1 / self.n_trees_)

            # Fit the tree to bootstrapped data
            tree.fit(X, y)

            # Append tree to a list
            self.trees.append(tree)

    # Create a method to predict the target variable for the features array
    def predict(self, X):

        # Initialize a list to store the prediction value for each tree
        trees_predictions = np.empty((len(X), len(self.trees)))

        # Iterate over all observations
        for i, tree in enumerate(self.trees):
            trees_predictions[:, i] = tree.predict(X)

        # Take the average of all trees for each observation in dataset
        predictions = np.mean(trees_predictions, axis=1)

        # Return an array with predictions for each observation
        return predictions
