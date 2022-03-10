# BMI 203 Project 7: Neural Network


# Importing Dependencies
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike


# Neural Network Class Definition
class NeuralNetwork:
    """
    This is a neural network class that generates a fully connected Neural Network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            This list of dictionaries describes the fully connected layers of the artificial neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}] will generate a
            2 layer deep fully connected network with an input dimension of 64, a 32 dimension hidden layer
            and an 8 dimensional output.
        lr: float
            Learning Rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            This list of dictionaries describing the fully connected layers of the artificial neural network.
    """
    def __init__(self,
                 nn_arch: List[Dict[str, Union[int, str]]],
                 lr: float,
                 seed: int,
                 batch_size: int,
                 epochs: int,
                 loss_function: str):
        # Saving architecture
        self.arch = nn_arch
        # Saving hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size
        # Initializing the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD!! IT IS ALREADY COMPLETE!!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """
        # seeding numpy random
        np.random.seed(self._seed)
        # defining parameter dictionary
        param_dict = {}
        # initializing all layers in the NN
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            # initializing weight matrices
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            # initializing bias matrices
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1
        return param_dict

    def _single_forward(self,
                        W_curr: ArrayLike,
                        b_curr: ArrayLike,
                        A_prev: ArrayLike,
                        activation: str) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        self.W_curr = W_curr
        self.b_curr = b_curr 
        self.A_prev = A_prev 
        self.activation = activation 

        if A_prev.shape[1] != W_curr.shape[0]:
            W_curr = W_curr.T

        Z_curr = np.dot(A_prev, W_curr) + b_curr.T

        if activation == 'sigmoid':
            A_curr = self._sigmoid(Z_curr)
        elif activation == 'relu':
            A_curr = self._relu(Z_curr)

        return Z_curr, A_curr
        

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        self.X = X 

        # dictionary for Z & A matrices (named 'cache')
        cache = {}

		# initializing inputs -- 'A0' because we need to get back to 'A' matrices when we backpropagate
        cache['A0'] = X 
        A_prev = X

        for _index, _layer in enumerate(self.arch):
            W_curr = self._param_dict['W' + str(_index + 1)]
            b_curr = self._param_dict['b' + str(_index + 1)]
            activation = _layer['activation']

            Z_curr, A_curr = self._single_forward(W_curr, b_curr, A_prev, activation)

            cache['Z' + str(_index + 1)] = Z_curr
            cache['A' + str(_index + 1)] = A_curr 

            A_prev = A_curr 

        output = A_prev 

        return output, cache 


    def _single_backprop(self,
                         W_curr: ArrayLike,
                         b_curr: ArrayLike,
                         Z_curr: ArrayLike,
                         A_prev: ArrayLike,
                         dA_curr: ArrayLike,
                         activation_curr: str) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        self.W_curr = W_curr
        self.b_curr = b_curr 
        self.Z_curr = Z_curr 
        self.A_prev = A_prev 
        self.dA_curr = dA_curr
        self.activation_curr = activation_curr

        dZ_curr = []

        if activation_curr == 'sigmoid':
            dZ_curr = self._sigmoid_backprop(dA_curr, Z_curr)
        elif activation_curr == 'relu':
            dZ_curr = self._relu_backprop(dA_curr, Z_curr)

        datapoints = dZ_curr.shape[0]
        d_dA_prev = np.dot(dZ_curr, W_curr)
        d_dW_curr = np.dot(dZ_curr.T, A_prev)
        d_db_curr = np.dot(dZ_curr.T, np.ones((datapoints, 1)))
        
        dA_prev = d_dA_prev
        dW_curr = d_dW_curr
        db_curr = d_db_curr
        
        return dA_prev, dW_curr, db_curr

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        self.y = y 
        self.y_hat = y_hat 
        self.cache = cache 

        # Dictionary containing the gradient information from this pass of backprop
        grad_dict = {}

		# First derivative == d/dy_hat
        d_dy_hat = []

		# Choosing the loss function to use
        if self._loss_func == "mse":
            d_dy_hat = self._mean_squared_error_backprop(y, y_hat)
        elif self._loss_func == 'bce':
            d_dy_hat = self._binary_cross_entropy_backprop(y, y_hat)
            
        dA_curr = d_dy_hat

		# Backpropagating through the network
        for _index, _layer in reversed(list(enumerate(self.arch))):
            W_curr = self._param_dict['W' + str(_index + 1)]
            b_curr = self._param_dict['b' + str(_index + 1)]
            Z_curr = cache['Z' + str(_index + 1)]
            A_prev = cache['A' + str(_index)]
            activation_curr = _layer["activation"]
            
            dA_prev, dW_curr, db_curr = self._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, activation_curr)
            
            grad_dict['dW' + str(_index + 1)] = dW_curr
            grad_dict['db' + str(_index + 1)] = db_curr

			# Renaming variables
            dA_curr = dA_prev
            
        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and thus does not return anything

        Method: Gradient Descent

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.

        Returns:
            None
        """
        self.grad_dict = grad_dict 
        
        for _index, _layer in enumerate(self.arch):
            W_prev = self._param_dict['W' + str(_index + 1)]
            b_prev = self._param_dict['b' + str(_index + 1)]
            dW = grad_dict['dW' + str(_index + 1)]
            db = grad_dict['db' + str(_index + 1)]
            W_new = W_prev - self._lr * dW 
            b_new = b_prev - self._lr * db 
            
            self._param_dict['W' + str(_index + 1)] = W_new
            self._param_dict['b' + str(_index + 1)] = b_new



    def fit(self,
            X_train: ArrayLike,
            y_train: ArrayLike,
            X_val: ArrayLike,
            y_val: ArrayLike) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network via training for the number of epochs defined at
        the initialization of this class instance.
        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        self.X_train = X_train 
        self.y_train = y_train 
        self.X_val = X_val 
        self.y_val = y_val 

        # Lists of per epoch loss for training & validation sets
        per_epoch_loss_train = []
        per_epoch_loss_val = []

		# training the model
        # self._epochs = # of trainings (1 training = 1 forward & backprop?)
        for i in range(self._epochs):

			# for each round of training, concatenate and shuffle the data
            shuffle_data = np.concatenate([X_train, y_train], axis=1)
            np.random.shuffle(shuffle_data)

			# how many dimensions is y_train?
            y_train_dims = y_train.shape[1]
            shuffle_data_dims = shuffle_data.shape[1]
            cutoff = shuffle_data_dims - y_train_dims

			# determine batches
            X_train = shuffle_data[:, :cutoff]
            y_train = shuffle_data[:, cutoff:]
            num_batches = np.ceil(X_train.shape[0]/self._batch_size)
            X_batch = np.array_split(X_train, num_batches)
            y_batch = np.array_split(y_train, num_batches)

            batches = zip(X_batch, y_batch)

            # keeping track of loss history through training
            loss_history_train = []
            loss_history_val = []

			# iterating through all batches
            for X_train, y_train in batches:

				# forward pass
                output, cache = self.forward(X_train)
                
                if self._loss_func == "mse":
                    loss_train = self._mean_squared_error(y_train, output)
                elif self._loss_func == "bce":
                    loss_train = self._binary_cross_entropy(y_train, output)
                    
                loss_history_train.append(loss_train)

				# backward pass
                grad_dict = self.backprop(y_train, output, cache)
                
                self._update_params(grad_dict)
                
                output_val = self.predict(X_val)
                
                if self._loss_func == "mse":
                    loss_val = self._mean_squared_error(y_val, output_val)
                elif self._loss_func == "bce":
                    loss_val = self._binary_cross_entropy(y_val, output_val)
                    
                loss_history_val.append(loss_val)

			# average training & validation loss
            per_epoch_loss_train.append(sum(loss_history_train)/len(loss_history_train))
            per_epoch_loss_val.append(sum(loss_history_val)/len(loss_history_val))

        # End of training #
        return per_epoch_loss_train, per_epoch_loss_val

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network model.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        self.X = X 
        
        y_hat, cache = self.forward(X)
        return y_hat

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        self.Z = Z  
        nl_transform = 1.0/(1+np.exp(-Z))

        return nl_transform

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        self.Z = Z 
        
        nl_transform = np.maximum(0.0, Z)
        return nl_transform 

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        self.dA = dA 
        self.Z = Z 

        dZ = (1.0/(1+np.exp(-Z))) * (1-(1.0/(1+np.exp(-Z))))
        dZ = dZ * dA 

        return dZ

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        self.dA = dA 
        self.Z = Z 

        Z[Z <= 0.0] = 0
        Z[Z > 0.0] = 1

        dZ = Z * dA 

        return dZ 

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        self.y = y 
        self.y_hat = y_hat 

        y_hat[y_hat == 1] = 0.9999
        y_hat[y_hat == 0] = 0.0001

        loss = -np.mean(y * (np.log(y_hat)) - (1 - y) * np.log(1 - y_hat))

        return loss 

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        self.y = y 
        self.y_hat = y_hat 

        dA = np.mean((-y/y_hat) + (1-y/1-y_hat))

        return dA 

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        self.y = y 
        self.y_hat = y_hat 

        loss = np.mean((np.square(y - y_hat)))

        return loss 

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        self.y = y 
        self.y_hat = y_hat 

        dA = np.mean(-2 * (y - y_hat))

        return dA 

    def _loss_function(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Loss function, computes loss given y_hat and y. This function is
        here for the case where someone would want to write more loss
        functions than just binary cross entropy.

        _loss_function = categorical cross-entropy

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.
        Returns:
            loss: float
                Average loss of mini-batch.
        """
        self.y = y 
        self.y_hat = y_hat 

        samples = len(y_hat)
        y_hat_clipped = np.clip(y_hat, 1e-7, 1-1e-7)

        if len(y.shape) == 1:
            correct_conf = y_hat_clipped[range(samples), y]
        elif len(y.shape) == 2:
            correct_conf = np.sum(y_hat_clipped * y, axis=1)

        neg_log_lik = -np.log(correct_conf)

        loss = np.mean(neg_log_lik)

        return loss


    def _loss_function_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        This function performs the derivative of the loss function with respect
        to the loss itself.
        Args:
            y (array-like): Ground truth output.
            y_hat (array-like): Predicted output.
        Returns:
            dA (array-like): partial derivative of loss with respect
                to A matrix.
        """
        pass

    def pytest_params(self, pytest_parameters):
        """
        Method to set parameters for pytest functions (so that they won't be random everytime)
        """

        self._param_dict = pytest_parameters
        
