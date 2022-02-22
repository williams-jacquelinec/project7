# Project 7
Neuronal Network implementation


# Assigment

## Overview
In this assignment, you will try out your machine learning skills by first implementing a neural network
class from (almost) scratch. You will then apply your class to implement both

**(1) a simple 8x3x8 autoencoder**

**(2) a classifier for transcription factor binding sites**

In this assigment you will begin by finishing the API for generating fully connected neural networks from scratch.
Next, you will make a jupyter notebook where you will implement your 8x3x8 neural network and your 
classifier for transcription factor binding sites.


## Step 1: Finish the neural network API
TODO:
* Finish all incomplete methods that have a pass statement in the NN class in the nn.py file
* Finish the 'one_hot_encode_seqs' function in the preprocess.py file
* Finish the 'sample_seqs' function in the preprocess.py file


## Step 2: Generate your Autoencoder
### Background
Autoencoders are a type of neural network architecture that takes in an input, encodes that information
into a lower-dimensional latent space via 'encoding' layer(s) and then attempts to reconstruct the intial
input via 'decoding' layer(s). Autoencoders are most often used as a dimensionality reduction technique.

### Your task
Here you will train train an 8x3x8 autoencoder. All of the following work should be done in a jupyter notebook.

TODO:
* Generate an instance of your NN class for the 8x3x8 autoencoder
* Generate a training set and a validation set of 500 random 8-dimensional vectors each
* Train your autoencoder on the training and validation set
* Plot your training and validation loss per epoch
* Explain in your jupyter notebook why you chose the hyperparameter values that you did
* Show an example of your autoencoder accurately reconstrcuting a sinle input value


## Transcription Factor Classifier
### Background
Transcription factors are proteins that bind DNA at promoters to drive gene expression. 
Most preferentially bind specific patterns of sequence, while ignoring others. 
Traditional methods to determine this pattern (called a motif) have assumed that binding 
sites in the genome are all independent. However, there are cases where people have identified motifs where
positional interdependencies exist.

### Your task
Here you will implement a multilayer fully connected neural network using your NN class
capable of accurately predicting whether a short DNA sequence is a binding site for the 
yeast transcription factor Rap1. Note that the training data is incredibly imbalanced as
there are much fewer positive sequences than negative sequences. In order to overcome this
you will need to implement a sampling scheme to ensure that class imbalance does not effect
your training.

TODO:
* Use the 'read_text_file' function from preprocess.py to read in the 137 positive Rap1 motif examples
* Use the 'read_fasta_file' function to read in all the negative examples from all 1kb upstream in yeast.
* Implement a sampling scheme in the 'sample_seq' function in the preprocess.py file
* Explain in your jupyter notebook why chose the sampling scheme that you did.
* Generate a training a validation set for training your classifier.
* One hot encode your training and validation sets using your implementation of the 'one_hot_encode_seqs' function in the preprocess.py file
* Train your neural network
* Explain your choice of loss function in the jupyter notebook
* Explain your choice of hyperparameters in the jupyter notebook
* Plot the training and validation loss curves per epoch
* Print out the accuracy of your classifier on your validation dataset


# Grading

## Implementation of API (15 points)
* Proper implementation of NN class (13 points)
* Proper implementation of 'one_hot_encode_seqs' function (1 point)
* Proper implementation of 'sample_seqs' function (1 point)

## Autoencoder (10 points)
* Generating training and validation datasets (2 points)
* Successfully train your autoencoder (4 points)
* Plots of training and validation loss (2 points)
* Explanation of hyperparameters and exaple reconstruction (2)

## Transcription Factor Classifier (15 points)
* Correctly read in all data (2 points)
* Explanation of your sampling scheme (2 points)
* Proper generation of a training set and a validation set (2 point)
* Successfully train your classifeir (4 points)
* Explain the choice of your loss function in the jupyter notebook (2 points)
* Plots of training and validation loss (2 points)
* Print out accuracy of the classifier on the training set (2 point)




