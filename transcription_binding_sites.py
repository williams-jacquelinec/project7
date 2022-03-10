# Import necessary dependencies here
import numpy as np
from typing import List, Dict, Tuple, Union 
from numpy.typing import ArrayLike
from nn import NeuralNetwork, preprocess, io
from sklearn import metrics
from itertools import repeat

## Use the 'read_text_file' function from preprocess.py to read in the 137 positive Rap1 motif examples
pos_rap1_examples = io.read_text_file('./data/rap1-lieb-positive.txt')

## Use the 'read_fasta_file' function to read in all the negative examples from all 1kb upstream in yeast.
neg_yeast_examples = io.read_fasta_file('./data/yeast-upstream-1k-negative.fa')

## Implement a sampling scheme in the 'sample_seqs' function in the preprocess.py file
# positive = True, negative = False
joined_examples = pos_rap1_examples + neg_yeast_examples
pos_labels = [True] * len(pos_rap1_examples)
neg_labels = [False] * len(neg_yeast_examples)
joined_labels = pos_labels + neg_labels

sampled_seqs, sampled_labels = preprocess.sample_seqs(joined_examples, joined_labels, 0.75)

## Explain in your jupyter notebook why chose the sampling scheme that you did.
"""
I chose to sample ~75% of the examples from the smallest list. This way we could get a good spread of the data.
"""

## Generate a training and a validation set for training your classifier.
# 75% of samples can go in training, other 25% can go in validation

sample_portion = round(len(sampled_seqs)*0.75)
X_train = sampled_seqs[:sample_portion]
y_train = np.expand_dims(np.array(sampled_labels[:sample_portion], axis=1))
X_val = sampled_seqs[sample_portion:]
y_val = np.expand_dims(np.array(sampled_labels[sample_portion:], axis=1))


## One hot encode your training and validation sets using your implementation of the 'one_hot_encode_seqs' function in the preprocess.py file
X_train_encodings = preprocess.one_hot_encode_seqs(X_train)
X_val_encodings = preprocess.one_hot_encode_seqs(X_val)

## Train your neural network!
nn_arch = [{'input_dim': 64, 'output_dim': 16, 'activation': 'sigmoid'}, {'input_dim': 16, 'output_dim': 64, 'activation': 'sigmoid'}]
transcription_nn = NeuralNetwork(nn_arch, lr = 0.00001, seed = 10, batch_size = 2, epochs = 4, loss_function = 'bce')

per_epoch_loss_train, per_epoch_loss_val = transcription_nn.fit(X_train, y_train, X_val, y_val)

## Explain your choice of loss function in the jupyter notebook
"""
I chose the binary cross entropy loss function because the encoded sequences were binary values (0 or 1)
"""

## Explain your choice of hyperparameters in the jupyter notebook
"""
I chose sigmoid because the initial input values of the network are 0s and 1s.
I chose a low learning rate because I wanted to ensure that my losses would not take a long time to get closer to 0.
I chose a small batch size to avoid a RuntimeWarning error if the batch size was too large
I chose a small number of epochs because loss reached a low point after a few runs
"""

## Plot the training and validation loss curves per epoch
fig, axs = plt.subplots(2, figsize=(8,8))
fig.suptitle('Loss History')
axs[0].plot(np.arange(len(per_epoch_loss_train)), per_epoch_loss_train)
axs[0].set_title('Training Loss')
axs[1].plot(np.arange(len(per_epoch_loss_val)), per_epoch_loss_val)
axs[1].set_title('Validation Loss')
plt.xlabel('Number of Epochs')
axs[0].set_ylabel('Training Loss')
axs[1].set_ylabel('Validation Loss')
fig.tight_layout()
plt.show()

## Print out the accuracy of your classifier on your validation dataset