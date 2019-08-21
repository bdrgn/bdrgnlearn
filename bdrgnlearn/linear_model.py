import numpy as np

class LinearRegression:
    def __init__(self):
        pass

    def mserror(self, y, y_pred):
        # Calculate the mean squared error by subtracting predicted values from the true then
        # getting second power and averaging
        mse = np.mean(np.power(y - y_pred, 2))

        # Return the mean squared error
        return mse

    def stochastic_gradient_step(self, X, y, weights, train_ind, eta=0.01):
        # Calculate the difference of predicted and actual values for a given observation index
        diff = np.dot(X[train_ind, :], weights) - y[train_ind]

        # Calculate the gradient by multiplying features by computed difference
        grad = X[train_ind, :] * diff

        # Measure the number of observations
        l = len(y)

        # Calculate the updated weights by subtracting a vector of partial derivatives times eta / l
        new_weights = weights - 2 * (eta / l) * grad

        # Return the updated weights
        return new_weights

    def stochastic_gradient_descent(self, X, y, w_init, eta=1e-2, max_iter=1e5,
                                    min_weight_dist=1e-8, seed=42):
        # Initialize the initial distance between weight vectors on neighbor iterations
        # as infinitely large number
        weight_dist = np.inf

        # Set the initial weights according to the keyword argument
        w = w_init

        # Initialize a list of errors at each iteration
        self.errors_ = []

        # Create an iteration counter
        iter_num = 0

        # Set the random seed
        np.random.seed(seed)

        # Find the needed weights in a while loop
        while weight_dist > min_weight_dist and iter_num < max_iter:
            # Choose the random index
            random_ind = np.random.randint(X.shape[0])

            # Make a step towards function minimum
            new_w = self.stochastic_gradient_step(X, y, w, random_ind)

            # Append the current error to the errors log
            self.errors_.append(self.mserror(y, np.dot(X, w)))

            # Increase a number of iterations by 1
            iter_num += 1

            # Calculate the weights distance between current point and the previous one
            weight_dist = np.sum(np.power(w - weight_dist, 2))

            # Assign the new point to the weights variable
            w = new_w

        # Return the optimal weights
        return w

    def fit(self, X, y):
        # Calculate means and standard deviations of each feature in the dataset
        means, stds = np.mean(X, axis=0), np.std(X, axis=0)

        # Save the means and stds to normalize the features when needed for predictions
        self.means_ = means
        self.stds_ = stds

        # Normalize features array
        X = (X - means) / stds

        # Add a column with 1s to mimic the w0
        X = np.concatenate((np.ones(len(X)).reshape(len(X), 1), X), axis=1)

        # Find the optimal weights with SGD and assign them to a class variable
        self.weights_ = self.stochastic_gradient_descent(X=X, y=y, w_init=np.random.random(X.shape[1]))

    def predict(self, X):
        # Normalize the features array
        X = (X - self.means_) / self.stds_

        # Add a column with 1s to mimic the w0
        X = np.concatenate((np.ones(len(X)).reshape(len(X), 1), X), axis=1)

        # Return the dot product of the features array and optimized weights
        return np.dot(X, self.weights_)


import numpy as np

class LogisticRegression:
    
    def __init__(self):    
        pass

    def mserror(self, y, y_pred):
        # Calculate the mean squared error by subtracting predicted values from the true then
        # getting second power and averaging
        mse = np.mean(np.power(y - y_pred, 2))

        # Return the mean squared error
        return mse
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))    

    def stochastic_gradient_step(self, X, y, weights, train_ind, eta=0.01):
        
        z = np.dot(X[train_ind, :], weights)
        
        y_pred = self.sigmoid(z)
        
        # Calculate the difference of predicted and actual values for a given observation index
        diff = y_pred - y[train_ind]

        # Calculate the gradient by multiplying features by computed difference
        grad = X[train_ind, :] * diff

        # Measure the number of observations
        l = len(y)

        # Calculate the updated weights by subtracting a vector of partial derivatives times eta / l
        new_weights = weights - 2 * (eta / l) * grad

        # Return the updated weights
        return new_weights

    def stochastic_gradient_descent(self, X, y, w_init, eta=1e-2, max_iter=1e5,
                                    min_weight_dist=1e-8, seed=42):
        # Initialize the initial distance between weight vectors on neighbor iterations
        # as infinitely large number
        weight_dist = np.inf

        # Set the initial weights according to the keyword argument
        w = w_init

        # Initialize a list of errors at each iteration
        self.errors_ = []

        # Create an iteration counter
        iter_num = 0

        # Set the random seed
        np.random.seed(seed)

        # Find the needed weights in a while loop
        while weight_dist > min_weight_dist and iter_num < max_iter:
            # Choose the random index
            random_ind = np.random.randint(X.shape[0])

            # Make a step towards function minimum
            new_w = self.stochastic_gradient_step(X, y, w, random_ind)

            # Append the current error to the errors log
            self.errors_.append(self.mserror(y, np.dot(X, w)))

            # Increase a number of iterations by 1
            iter_num += 1

            # Calculate the weights distance between current point and the previous one
            weight_dist = np.sum(np.power(w - weight_dist, 2))

            # Assign the new point to the weights variable
            w = new_w

        # Return the optimal weights
        return w

    def fit(self, X, y):
        # Calculate means and standard deviations of each feature in the dataset
        means, stds = np.mean(X, axis=0), np.std(X, axis=0)

        # Save the means and stds to normalize the features when needed for predictions
        self.means_ = means
        self.stds_ = stds

        # Normalize features array
        X = (X - means) / stds

        # Add a column with 1s to mimic the w0
        X = np.concatenate((np.ones(len(X)).reshape(len(X), 1), X), axis=1)

        # Find the optimal weights with SGD and assign them to a class variable
        self.weights_ = self.stochastic_gradient_descent(X=X, y=y, w_init=np.random.random(X.shape[1]))

    def predict(self, X):
        # Normalize the features array
        X = (X - self.means_) / self.stds_

        # Add a column with 1s to mimic the w0
        X = np.concatenate((np.ones(len(X)).reshape(len(X), 1), X), axis=1)

        # Return the dot product of the features array and optimized weights
        z = np.dot(X, self.weights_)
        
        z = np.int64(z >= 0.5)
        
        return z
    
    def predict_proba(self, X):
        # Normalize the features array
        X = (X - self.means_) / self.stds_

        # Add a column with 1s to mimic the w0
        X = np.concatenate((np.ones(len(X)).reshape(len(X), 1), X), axis=1)

        # Return the dot product of the features array and optimized weights
        z = np.dot(X, self.weights_)
        
        y_pred = self.sigmoid(z)
        
        return y_pred