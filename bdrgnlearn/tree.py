import numpy as np

class DecisionTreeClassifier:
    # Initialize the object entity
    def __init__(self, min_samples_leaf=1, max_depth=None, threshold=0.5, max_features=None, bootstrap=False, \
                 bootstrap_coeff=1):

        # Store the input parameters of the class
        self.max_depth_ = max_depth
        self.min_samples_leaf_ = min_samples_leaf
        self.threshold_ = threshold
        self.max_features_ = max_features
        self.bootstrap_ = bootstrap
        self.bootstrap_coeff_ = bootstrap_coeff

        # Initialize a dict to mimic a binary tree structure
        self.tree = {}
        self.tree['0'] = {}

    # Initialize a method to calculate GINI
    def gini_impurity(self, obj_set):

        # Convert input array to numpy array if necessary
        obj_set = np.array(obj_set)

        # Store a set of unique objects in variable
        unique_objects = set(obj_set)

        # Save the number of unique objects in the input array
        n_unique_objects = len(unique_objects)

        # Produce a complementatry array with the input array times number unique objects
        obj_set_2d = np.array([list(obj_set)] * n_unique_objects)

        # Produce a second complementatry array with the input array times number unique objects
        obj_set_2d_comp = np.array([[x] * len(obj_set) for x in unique_objects])

        # Calculate gini for each class
        obj_set_bool = obj_set_2d == obj_set_2d_comp

        # Calculate final gini by subtracting sum of all elements from 1
        result = 1 - np.sum(np.power(np.sum(obj_set_bool, axis=1) / len(obj_set), 2))

        # Return the result
        return result

    # Initialize a method to find the best split for features array and target series
    def find_best_split(self, X, y):

        # Initialize a variable to store the best split data
        best_split = None

        # Save the length of the features array
        total_size = len(X)

        # Initialize an array of feature indices
        features = np.arange(X.shape[1])

        # Choose indices according to the max_features hyperparameter
        needed_features = np.random.choice(features, self.max_features_, replace=False)

        # Iterate over column and value combination
        for col in needed_features:
            possible_split = sorted(set(X[:, col]))
            possible_split = [np.mean((possible_split[x], possible_split[x + 1])) for x in
                              range(len(possible_split) - 1)]
            for split in possible_split:

                # Split target series to two parts
                target_part_1 = y[X[:, col] <= split]
                target_part_2 = y[X[:, col] > split]

                # Measure the length of the targets
                size_1 = len(target_part_1)
                size_2 = len(target_part_2)

                # Calculate the GINIs
                GINI1 = self.gini_impurity(target_part_1)
                GINI2 = self.gini_impurity(target_part_2)

                # Calculate the weighted GINI for the split
                split_result = (size_1 / total_size) * GINI1 + (size_2 / total_size) * GINI2
                # Update best split data if necessary
                if best_split is None:
                    best_split = {}
                    best_split['GINI'] = split_result
                    best_split['split'] = split
                    best_split['col'] = col
                elif best_split['GINI'] > split_result:
                    best_split['GINI'] = split_result
                    best_split['split'] = split
                    best_split['col'] = col

        # Return the best column-value pair
        return (best_split['col'], best_split['split'])

    def split(self, X, y, parent_node='0'):

        # Estimate the condition in which the tree growing stops
        if X.shape[0] <= self.min_samples_leaf_ or len(parent_node) == self.max_depth_ or \
                        np.unique(X, axis=0).shape[0] == 1 or len(np.unique(y)) == 1:
            # Add metadata to the leaf
            self.tree[parent_node]['COL'] = ''
            self.tree[parent_node]['VAL'] = ''
            self.tree[parent_node]['MEAN'] = np.mean(y)
            self.tree[parent_node]['OBS'] = X.shape[0]
            self.tree[parent_node]['GINI'] = self.gini_impurity(y)
        else:
            # Find the best split for data
            col, val = self.find_best_split(X, y)

            # Add split data to parent node metadata
            self.tree[parent_node]['COL'] = col
            self.tree[parent_node]['VAL'] = val
            self.tree[parent_node]['MEAN'] = np.mean(y)
            self.tree[parent_node]['OBS'] = X.shape[0]
            self.tree[parent_node]['GINI'] = self.gini_impurity(y)

            # Add the first node of the split
            new_node_1 = parent_node + '0'
            self.tree[new_node_1] = {}

            # Add the second node of the split
            new_node_2 = parent_node + '1'
            self.tree[new_node_2] = {}

            # Split features and targets array in two
            X1, y1 = X[X[:, col] <= val], y[X[:, col] <= val]
            X2, y2 = X[X[:, col] > val], y[X[:, col] > val]

            # Return the result in recursion
            return self.split(X1, y1, parent_node=new_node_1), \
                   self.split(X2, y2, parent_node=new_node_2)

    # Create a method to fit the model
    def fit(self, X, y):

        # Store the number of features in a variable
        n_features = X.shape[1]

        # Initialize a maximum number of features at each split
        if type(self.max_features_) == int:
            pass
        elif type(self.max_features_) == float:
            self.max_features_ = int(self.max_features_ * n_features)
        elif self.max_features_ == 'auto':
            self.max_features_ = int(np.sqrt(n_features))
        elif self.max_features_ == 'sqrt':
            self.max_features_ = int(np.sqrt(n_features))
        elif self.max_features_ == 'log2':
            self.max_features_ = np.log2(n_features)
        elif self.max_features_ == None:
            self.max_features_ = n_features

        # Store the original input in a special variables
        self.X_original_ = X
        self.y_original_ = y

        # Bootstrap observations for a tree if needed
        if self.bootstrap_:
            # Initialize an array of observations indices
            indices = np.arange(len(X))

            # Bootstrap observation indices
            bootstrapped_indices = np.random.choice(indices, int(len(indices) * self.bootstrap_coeff_))

            # Choose observations according to bootstrapped indices
            X = X[bootstrapped_indices, :]
            y = y[bootstrapped_indices]

        # Split the data recursively and build a tree
        self.split(X, y)

    # Initialize a method to find the right node for observation
    def right_node(self, X, parent_node='0'):

        # Estimate the condition in which the node finding stops
        if (parent_node + '0' not in self.tree) and (parent_node + '1' not in self.tree):

            # Return the right node's mean value
            return (lambda x: 0 if x < self.threshold_ else 1)(self.tree[parent_node]['MEAN'])
        else:
            # Get the optimal column and value from the current node metadata
            col = self.tree[parent_node]['COL']
            val = self.tree[parent_node]['VAL']

            # Go to the next node in recursion dependent on the condition
            if X[col] <= val:
                return self.right_node(X, parent_node=parent_node + '0')
            else:
                return self.right_node(X, parent_node=parent_node + '1')

    # Create a method to predict the target variable for the features array
    def predict(self, X):

        # Initialize a list to store the prediction value for each observation
        predictions = []

        # Iterate over all observations
        for obs in X:
            # Find the best node for observation (returns value)
            node = self.right_node(obs)

            # Add the prediction to the list
            predictions.append(node)

        # Return an array with predictions for each observation
        return predictions

class DecisionTreeRegressor:
    # Initialize the object entity
    def __init__(self, min_samples_leaf=1, max_depth=None, threshold=0.5, max_features=None, bootstrap=False, \
                 bootstrap_coeff=1):

        # Store the input parameters of the class
        self.max_depth_ = max_depth
        self.min_samples_leaf_ = min_samples_leaf
        self.threshold_ = threshold
        self.max_features_ = max_features
        self.bootstrap_ = bootstrap
        self.bootstrap_coeff_ = bootstrap_coeff

        # Initialize a dict to mimic a binary tree structure
        self.tree = {}
        self.tree['0'] = {}

    # Initialize a method to calculate MSE
    def calc_mse(self, x):
        return np.mean(np.power(x - np.mean(x), 2))

    # Initialize a method to find the best split for features array and target series
    def find_best_split(self, X, y):

        # Initialize a variable to store the best split data
        best_split = None

        # Save the length of the features array
        total_size = len(X)

        # Initialize an array of feature indices
        features = np.arange(X.shape[1])

        # Choose indices according to the max_features hyperparameter
        needed_features = np.random.choice(features, self.max_features_, replace=False)

        # Iterate over column and value combination
        for col in needed_features:
            possible_split = sorted(set(X[:, col]))
            possible_split = [np.mean((possible_split[x], possible_split[x + 1])) for x in
                              range(len(possible_split) - 1)]
            for split in possible_split:

                # Split target series to two parts
                target_part_1 = y[X[:, col] <= split]
                target_part_2 = y[X[:, col] > split]

                # Measure the length of the targets
                size_1 = len(target_part_1)
                size_2 = len(target_part_2)

                # Calculate the MSEs
                MSE1 = self.calc_mse(target_part_1)
                MSE2 = self.calc_mse(target_part_2)

                # Calculate the weighted MSE for the split
                split_result = (size_1 / total_size) * MSE1 + (size_2 / total_size) * MSE2
                # Update best split data if necessary
                if best_split is None:
                    best_split = {}
                    best_split['MSE'] = split_result
                    best_split['split'] = split
                    best_split['col'] = col
                elif best_split['MSE'] > split_result:
                    best_split['MSE'] = split_result
                    best_split['split'] = split
                    best_split['col'] = col

        # Return the best column-value pair
        return (best_split['col'], best_split['split'])

    def split(self, X, y, parent_node='0'):

        # Estimate the condition in which the tree growing stops
        if X.shape[0] <= self.min_samples_leaf_ or len(parent_node) == self.max_depth_ or \
                        np.unique(X, axis=0).shape[0] == 1 or len(np.unique(y)) == 1:
            # Add metadata to the leaf
            self.tree[parent_node]['COL'] = ''
            self.tree[parent_node]['VAL'] = ''
            self.tree[parent_node]['MEAN'] = np.mean(y)
            self.tree[parent_node]['OBS'] = X.shape[0]
            self.tree[parent_node]['MSE'] = self.calc_mse(y)
        else:
            # Find the best split for data
            col, val = self.find_best_split(X, y)

            # Add split data to parent node metadata
            self.tree[parent_node]['COL'] = col
            self.tree[parent_node]['VAL'] = val
            self.tree[parent_node]['MEAN'] = np.mean(y)
            self.tree[parent_node]['OBS'] = X.shape[0]
            self.tree[parent_node]['MSE'] = self.calc_mse(y)

            # Add the first node of the split
            new_node_1 = parent_node + '0'
            self.tree[new_node_1] = {}

            # Add the second node of the split
            new_node_2 = parent_node + '1'
            self.tree[new_node_2] = {}

            # Split features and targets array in two
            X1, y1 = X[X[:, col] <= val], y[X[:, col] <= val]
            X2, y2 = X[X[:, col] > val], y[X[:, col] > val]

            # Return the result in recursion
            return self.split(X1, y1, parent_node=new_node_1), \
                   self.split(X2, y2, parent_node=new_node_2)

    # Create a method to fit the model
    def fit(self, X, y):

        # Store the number of features in a variable
        n_features = X.shape[1]

        # Initialize a maximum number of features at each split
        if type(self.max_features_) == int:
            pass
        elif type(self.max_features_) == float:
            self.max_features_ = int(self.max_features_ * n_features)
        elif self.max_features_ == 'auto':
            self.max_features_ = int(np.sqrt(n_features))
        elif self.max_features_ == 'sqrt':
            self.max_features_ = int(np.sqrt(n_features))
        elif self.max_features_ == 'log2':
            self.max_features_ = np.log2(n_features)
        elif self.max_features_ == None:
            self.max_features_ = n_features

        # Store the original input in a special variables
        self.X_original_ = X
        self.y_original_ = y

        # Bootstrap observations for a tree if needed
        if self.bootstrap_:
            # Initialize an array of observations indices
            indices = np.arange(len(X))

            # Bootstrap observation indices
            bootstrapped_indices = np.random.choice(indices, int(len(indices) * self.bootstrap_coeff_))

            # Choose observations according to bootstrapped indices
            X = X[bootstrapped_indices, :]
            y = y[bootstrapped_indices]

        # Split the data recursively and build a tree
        self.split(X, y)

    # Initialize a method to find the right node for observation
    def right_node(self, X, parent_node='0'):

        # Estimate the condition in which the node finding stops
        if (parent_node + '0' not in self.tree) and (parent_node + '1' not in self.tree):

            # Return the right node's mean value
            return self.tree[parent_node]['MEAN']
        else:
            # Get the optimal column and value from the current node metadata
            col = self.tree[parent_node]['COL']
            val = self.tree[parent_node]['VAL']

            # Go to the next node in recursion dependent on the condition
            if X[col] <= val:
                return self.right_node(X, parent_node=parent_node + '0')
            else:
                return self.right_node(X, parent_node=parent_node + '1')

    # Create a method to predict the target variable for the features array
    def predict(self, X):

        # Initialize a list to store the prediction value for each observation
        predictions = []

        # Iterate over all observations
        for obs in X:
            # Find the best node for observation (returns value)
            node = self.right_node(obs)

            # Add the prediction to the list
            predictions.append(node)

        # Return an array with predictions for each observation
        return predictions
