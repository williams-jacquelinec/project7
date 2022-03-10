# BMI 203 Project 7: Neural Network


# Importing Dependencies
import numpy as np
import random 
from typing import List, Tuple
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

    A_encoding = [1, 0, 0, 0]
    T_encoding = [0, 1, 0, 0]
    C_encoding = [0, 0, 1, 0]
    G_encoding = [0, 0, 0, 1]

    # store encodings as a numpy array 
    # multiple by 4 because of each nucleotide codes for 4 numbers
    encodings = np.zeros((len(seq_arr), (4 * len(max(seq_arr, key = len)))))

    # encodings nucleotides in the list
    for i in range(len(seq_arr)):

        # transform string seq to list (easier to modify)
        one_hot_encode_seq = []
        one_hot_encode_seq[:0] = seq_arr[i] 

        # replace A nucleotide & flatten list
        # flatten after every replacement for indexing purposes
        one_hot_encode_seq = [A_encoding if nuc == 'A' else [nuc] for nuc in one_hot_encode_seq]
        one_hot_encode_seq = [item for sublist in one_hot_encode_seq for item in sublist]

        # replace T nucleotide & flatten list
        one_hot_encode_seq = [T_encoding if nuc == 'T' else [nuc] for nuc in one_hot_encode_seq]
        one_hot_encode_seq = [item for sublist in one_hot_encode_seq for item in sublist]

        # replace C nucleotide & flatten list
        one_hot_encode_seq = [C_encoding if nuc == 'C' else [nuc] for nuc in one_hot_encode_seq]
        one_hot_encode_seq = [item for sublist in one_hot_encode_seq for item in sublist]

        # replace G nucleotide & flatten list
        one_hot_encode_seq = [G_encoding if nuc == 'G' else [nuc] for nuc in one_hot_encode_seq]
        one_hot_encode_seq = [item for sublist in one_hot_encode_seq for item in sublist]

        # appending to encodings array
        encodings[i, :len(one_hot_encode_seq)] = one_hot_encode_seq


    return encodings


def sample_seqs(seqs: List[str], labels: List[bool], sample_percent: int): #-> Tuple[List[seq], List[bool]]:
    """
    This function should sample your sequences to account for class imbalance. 
    Consider this as a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels
        sample_percent: integer
            Integer (decimal) of sample size
            eg. 0.50 == 50% of the (smallest) dataset will be sampled

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """

    # zipping seqs and labels together
    seqs_and_labels = list(zip(seqs, labels))

    # filtering out positive (i[1] == True) and negative (i[1] == False) data
    positive_data = [i[0] for i in seqs_and_labels if i[1]==True]
    negative_data = [i[0] for i in seqs_and_labels if i[1]==False]

	# sample_size will be the sample_percent * smallest list
    positive_len = len(positive_data)
    negative_len = len(negative_data)
    
    ss = int(min(positive_len, negative_len) * sample_percent)

    # choosing random samples
    pos_sample = random.choices(positive_data, k=ss) 
    neg_sample = random.choices(negative_data, k=ss) 
    
    # establishing labels for positive and negative samples
    pos_sample = [(i, True) for i in pos_sample]
    neg_sample = [(i, False) for i in neg_sample]
    final_sample = pos_sample + neg_sample

    # shuffling data before splitting (will be better for larger sample sizes)
    random.shuffle(final_sample)

	# add extra sample if len(final_sample) is unbalanced after filtering
    if len(final_sample) % 2 == 0:
        pass
    elif len(final_sample) % 2 == 1:
        final_sample.append(random.sample(seqs_and_labels), k=1)

	# re-splitting lists
    sampled_seqs = [i[0] for i in final_sample]
    sampled_labels = [i[1] for i in final_sample]
    
    return sampled_seqs, sampled_labels



