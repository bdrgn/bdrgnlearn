import numpy as np

# Define the Suquential network class
class Sequential:
    
    def __init__(self, input_size, validation_X, validation_y, epochs):
        
        # Store the number of epochs
        self.epochs_ = epochs
        
        # Se the validation data
        self.validation_X_ = validation_X
        self.validation_y_ = validation_y
        self.validation_scores_ = []
        
        # Set the local variable to store the layers
        self.layers_ = []
        
        # Save the input size in local variable
        self.input_size_ = input_size
        
        # Define a local variable to store the losses in
        self.losses_ = []
        
    def add_layer(self, n):
        
        # Inialize the random weights
        if len(self.layers_) == 0:
            new_layer = np.random.random((self.input_size_, n))
        else:
            new_layer = np.random.random((self.layers_[-1].shape[1], n))
        
        # Add layer with random weights to local variable
        self.layers_.append(new_layer)
        
    def loss(self, y_true, y_pred):
        
        # Set the loss function to be MSE
        return np.mean(np.power(y_true - y_pred, 2))
        
    def predict(self, X):
        
        # Initialize the input matrix
        prev_result = X
        
        # Iterate over layers
        for i, layer in enumerate(self.layers_):
            
            # Forward propagate
            prev_result = np.matmul(prev_result, layer)
            
            # Apply relu
            prev_result = np.clip(prev_result, a_min=0, a_max=None)
            
        # Return the final result
        return prev_result
    
    def backward_propagate(self):

        # Choose the random index to calculate the gradient
        random_index = np.random.randint(len(self.X_))
        
        # Add current random_index to a set of all seen indices in this epoch
        self.seen_indices_.add(random_index)
        
        # Increase the counter of epochs if needed
        if len(self.seen_indices_) == len(self.X_):
            
            # Make predictions with current weights on validation data
            y_pred = self.predict(self.validation_X_)

            # Measure performance metric on validation data
            validation_score = self.loss(self.validation_y_, y_pred)

            # Append the validation score to a histor
            self.validation_scores_.append(validation_score)

            print(f"Epoch {self.passed_epochs_}. Validation loss: {int(validation_score)}.")
            
            # Increase the counter of passed epochs
            self.passed_epochs_ += 1
            
            # Clean the set of seen indices
            self.seen_indices_ = set()
            
        # Make predictions with current set of weights
        y_pred = self.predict(self.X_)

        # Calculate the difference of predicted and actual values for a given observation index
        loss = self.loss(self.y_[random_index], y_pred[random_index])
        
        # Save the loss in log
        self.losses_.append(loss)

        # Get a list of all layers in reverse order
        reversed_layers = [x for x in reversed(self.layers_)]

        # Initialize a variable to store backpropagated errors
        errors = []

        # Initialize the input matrix
        prev_result = np.array(loss)[..., None]

        # Iterate over layers
        for i, layer in enumerate(reversed_layers):

            # Forward propagate
            prev_result = np.matmul(prev_result, np.transpose(layer))

            # Add the backpropagated errors for layer to errors list
            errors.append(prev_result)

        # Reverse the list of backpropagated errors to match the order of layers
        self.errors_ = [x for x in reversed(errors)]
    
    def fit(self, X, y):
        
        # Record the variables on which model is trained
        self.X_ = X
        self.y_ = y
        
        # Initialize a set to store seen indices in SGD
        self.seen_indices_ = set()
        
        # Define a variable to store the number of passed epochs
        self.passed_epochs_ = 0

        # Iterate for a number of iterations for training
        while self.passed_epochs_ <= self.epochs_:

            # Propagate the errors backward
            self.backward_propagate()
            
            # Iterate over layers to update weights
            for i, layer in enumerate(self.layers_):
                
                # Preprocess the array of errors which is fed into the weights
                errors = self.errors_[i]
                errors = np.array([errors for i in range(layer.shape[1])])
                errors = np.transpose(errors)
            
                # Calculate the gradient for the weights of layer
                # as a fraction of total error explained by this weight
                grad = errors * layer
                
                # Apply the derivative of RELU to complete the gradient
                grad = np.clip(grad, a_min=0, a_max=1)
                
                # Make a step to the decreasing loss by subtracting gradient multiplied by step
                self.layers_[i] = self.layers_[i] - ((0.01 / len(X)) * grad)
