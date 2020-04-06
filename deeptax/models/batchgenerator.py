import numpy as np
from keras.preprocessing import sequence


def generator(fname, max_len, btch_size):
    while True:
        with open(fname) as fh:
            for dataset in read_fasta(fh, max_len, batch_size=btch_size):
                yield dataset


def read_fasta(f, max_len, batch_size=2000):
    X = []
    Y = []
    for line in f:
        line = line.strip()
        line = line.split('\t')
        y = line[0].split(',')
        y = [float(i) for i in y]
        sq = line[1].split(',')
        sq = [int(i) for i in sq]
        Y.append(y)
        X.append(sq)
        if len(X) == batch_size:
            X = np.array(X)
            X = sequence.pad_sequences(X, maxlen=max_len)
            Y = np.array(Y)
            yield X, Y
            X = []
            Y = []

    if X:
        X = sequence.pad_sequences(X, maxlen=max_len)
        Y = np.array(Y)
        yield X, Y

