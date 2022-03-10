# BMI 203 Project 7: Neural Network

# Import necessary dependencies here
import numpy as np
from typing import List, Dict, Tuple, Union 
from numpy.typing import ArrayLike
from nn import NeuralNetwork, preprocess
from sklearn import metrics
from itertools import repeat 

# setting up testing neural network
testing_dict = [{'input_dim': 8, 'output_dim': 4, 'activation': 'relu'}, {'input_dim': 4, 'output_dim': 2, 'activation:': 'sigmoid'}]
nn_test = NeuralNetwork(testing_dict, lr = 0.6, seed = 10, batch_size = 1, epochs = 2, loss_function = 'mse')

# setting up my testing parameters
pytest_parameters = {}

pytest_parameters['W1'] = np.array([[1,2,3,4],
                                    [5,6,7,8],
                                    [4,3,2,1],
                                    [8,7,6,5]])
pytest_parameters['b1'] = np.array([[0.5],[1.5],[2.5],[3.5]])

pytest_parameters['W2'] = np.array([[1],[2],[3],[4]])
pytest_parameters['b2'] = np.array([[1],[2],[3],[4]])

nn_test.pytest_params(pytest_parameters)

A_prev_test = np.array([[4,3,2,1]])


# Testing Functions

def test_forward():
    """
    test the full forward pass through the neural network
    """
    output, cache = nn_test.forward(A_prev_test)

    # expected final layer outputs
    Z2_expected = np.array([[436.], [447.], [458.], [469.]])
    A2_expected = np.array([[1.], [1.], [1.], [1.]])

    assert Z2_expected == cache['Z2']
    assert A2_expected == cache['A2']


def test_single_forward():
    """
    Testing outputs of single forward method.
    """
    output, cache = nn_test.forward(A_prev_test)

    # test if _single_forward in cache of full forward test
    Z_curr, A_curr = nn_test._single_forward(W_curr = pytest_parameters['W1'], b_curr = pytest_parameters['b1'], A_prev = A_prev_test, activation = 'relu')

    assert cache['Z1'] == Z_curr 
    assert cache['A1'] == A_curr 


def test_single_backprop():
    """
    Testing a single backpropagation through a neural network.
    """
    # setting values for test
    W_curr = np.array([[1,2,3,4], [5,6,7,8]])
    b_curr = np.array([[0.5,1]])
    Z_curr = np.array([[2,6], [4,8]])
    dA_curr = np.array([[7,5],[9,1]])
    A_prev = np.array([[1,2,3,4], [5,6,7,8]])

    dA_prev, dW_curr, db_curr = nn_test._single_backprop(W_curr = W_curr, b_curr = b_curr, Z_curr = Z_curr, dA_curr = dA_curr, A_prev = A_prev, activation_curr = 'relu')

    # expected outcomes
    exp_dA_prev = np.array([[32 44 56 68],[14 24 34 44]])
    exp_dW_curr = np.array([[ 52  68  84 100],[ 10  16  22  28]])
    exp_db_curr = np.array([[16.], [ 6.]])

    assert exp_dA_prev[0] == dA_prev[0]
    assert exp_dW_curr[1] == dW_curr[1]
    assert db_curr[0][0] == 16.0


def test_predict():
    """
    Assert that predict outcome matches the expectation
    """
    X_train = np.array([[2,3,4,5]])
    y_train = np.array([[4]])
    X_val = np.array([[5,6,7,8]])
    y_val = np.array([[6]])

    per_epoch_loss_train, per_epoch_loss_val = nn_test.fit(X_train = X_train, y_train = y_train, X_val = X_val, y_val = y_val)

    predict_test = nn_test.predict(X_train)

    exp_predict_test = np.array([[1. 1. 1. 1.]])

    assert predict_test == exp_predict_test


def test_binary_cross_entropy():
    """
    Assert that the error is what we expect
    """
    array_1 = np.array([0.01,0.99])
    array_2 = np.array([0.099,0.001])

    bce_error = nn_test._binary_cross_entropy(array_1, array_2)
    assert bce_error < 3.40  # expection: bce_error = 3.379293277158697

def test_binary_cross_entropy_backprop():
    """
    Assert that error backprop is what we expect
    """
    array_3 = np.array([0.01,0.99])
    array_4 = np.array([0.099,0.001])

    bce_error_backprop = nn_test._binary_cross_entropy_backprop(array_3, array_4)
    assert bce_error_backprop > -495 # expection: bce_error_backprop = -494.60050505050503


def test_mean_squared_error():
    """
    Assert that mean squared error is the same as a sklearn package
    """
    array_5 = np.array([2,3])
    array_6 = np.array([4,5])

    mse_test = nn_test._mean_squared_error(array_5, array_6)
    sklearn_test = metrics.mean_squared_error(array_5, array_6)

    assert mse_test == sklearn_test


def test_mean_squared_error_backprop():
    """
    Assert that mean squared error is what we expect
    """
    array_7 = np.array([2,3])
    array_8 = np.array([4,5])

    mse_backprop_test = nn_test._mean_squared_error_backprop(array_7, array_8)

    assert mse_backprop_test == 4



def test_one_hot_encode():
    """
    Testing if the function replaces nucleotides correctly (is the length of encodings what is expected)
    """

    test_list = ['ACATCCGTGCACCTCCG', 'ACACCCAGACATCGGGC', 'CCACCCGTACCCATGAC']
    encodings = one_hot_encode_seqs(test_list)
    
    for i in range(len(encodings)):
        assert len(encodings[i]) == len(test_list[i]) * 4


def test_sample_seqs():
    """
    Assert that this method returns a balanced sample (equal number of seqs and labels).
    """
  
    # test dataset (what color is the grass?)
    test_dataset = [("Green", True), ("Red", False)]

    positive_dataset = [v for value in test_dataset if v[0]=="Green"]
    negative_dataset = [v for value in test_dataset if v[0]=="Red"]

    # create an imbalance
    positive_dataset = list(repeat(positive_dataset, 3))
    negative_dataset = list(repeat(negative_dataset, 6))

    # combine and shuffle data around
    test_dataset = positive_dataset + negative_dataset
    random.shuffle(test_dataset)

    test_seqs = [i[0] for i in test_dataset]
    test_labels = [i[1] for i in test_dataset]

    # testing sample_seqs
    sampled_seqs, sampled_labels = preprocess.sample_seqs(test_seqs, test_labels)

    assert len(sampled_seqs) == len(sampled_labels)
