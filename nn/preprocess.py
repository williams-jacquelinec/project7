# BMI 203 Project 7: Neural Network


# Importing Dependencies
import numpy as np
from typing import List
from numpy.typing import ArrayLike


# Defining preprocessing functions
def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one hot encoding of a list of nucleic acid sequences
    for use as input into a fully connected neural net.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence
            length due to the one hot encoding scheme for nucleic acids.

            For example, if we encode 
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    """
    pass


def sample_seqs(seqs: List[str]) -> List[seq]:
    """
    This function should sample your negative sequences to account
    for class imbalance.

    Args:
        seqs: List[str]
            List of all negative sequences.

    Returns:
        sampled_seqs: List[str]
            List of sampled negative sequences.
    """
    pass
