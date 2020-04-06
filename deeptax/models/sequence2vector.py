import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pickle

"""
Author: Carlos Garcia-Perez
Date: 13.06.2019 first version of the script


Convert sequences to one-hot-ecoding.


This script has these functions:

    * seq2vectorize
    * index2sequence
    * label2one_hot_encoding
"""


def seq2vectorize(sequences_index):
    s2v = []
    for sequence in sequences_index:
        s2v.append(to_categorical(sequence))
    return s2v


def index2sequence(vec_seq):
    amino_alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                      'Y']
    index_to_amino = dict((i, c) for i, c in enumerate(amino_alphabet))

    translated = ''
    for idx, v in enumerate(vec_seq):
        translated = translated + index_to_amino[np.argmax(vec_seq[idx])]

    return translated


def label2one_hot_encoding(labels):
    lb_encoder = LabelEncoder()
    lb_encoder.fit(labels)
    lb_trans = lb_encoder.transform(labels)

    lbl_ohe = to_categorical(lb_trans)

    pickle.dump(lb_encoder, open("/data/lbl_encoder.pkl", 'wb'), protocol=4)

    return lbl_ohe
