import sys
#import deeptax.models.process_sequence_fasta as pro_seq_fasta
#import deeptax.models.sequence2vector as s2v_tools
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from keras.layers import Conv1D, Dense
from keras.layers import MaxPooling1D, GlobalMaxPooling1D
from keras.models import Sequential
from keras.layers import Embedding
from keras.preprocessing import sequence
import deeptax.deeptax.models.batchgenerator as bg


"""
Author: Carlos Garcia-Perez
Date: 26.06.2019 1D CNN for sequence classification
                 first version of the script
"""

def runtraining():
    print('loading data...')
    info = pickle.load(open("/srv/deeptax/deeptax/data/info.pkl", 'rb'))
    fname_train = '/srv/deeptax/deeptax/data/train.txt'
    fname_val = '/srv/deeptax/deeptax/data/val_train.txt'
    fname_test = '/srv/deeptax/deeptax/data/test.txt'
    print('defining model:')

    ly = 128  # layer
    btch_size = 250
    epch = 20

    features = 20
    num_classes = info[0]
    max_len = info[1]
    nsamples_train = info[2]
    nsamples_val = info[3]
    nsamples_test = info[4]

    train_steps_per_epoch = np.ceil(nsamples_train / btch_size)
    val_steps_per_epoch = np.ceil(nsamples_val / btch_size)
    test_steps_per_epoch = np.ceil(nsamples_test / btch_size)

    print('features: ', features)
    print('clases: ', num_classes)
    print('max_length', max_len)
    print('layer nodes: ', ly)  # 128
    print('bacth size: ', btch_size)  # 2000
    print('epochs: ', epch)
    print('train steps per epoch: ', train_steps_per_epoch)
    print('val steps per epoch: ', val_steps_per_epoch)
    print('test steps per epoch: ', test_steps_per_epoch)

    model = Sequential()
    model.add(Embedding(max_len, features, input_length=max_len))
    model.add(Conv1D(ly, 9, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(ly, 9, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(num_classes, activation='softmax'))

    print('compiling the model...')
    # compile the model
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print('training the model...')
    # metric
    train_generator = bg.generator(fname_train, max_len, btch_size)
    validation_generator = bg.generator(fname_val, max_len, btch_size)
    network = model.fit_generator(train_generator,
                                  steps_per_epoch=train_steps_per_epoch,
                                  epochs=epch,
                                  validation_data=validation_generator,
                                  validation_steps=val_steps_per_epoch)  # 0.2

    test_generator = bg.generator(fname_test, max_len, btch_size)
    results_eval = model.evaluate_generator(test_generator, steps=test_steps_per_epoch)

    print('training the model... done!!!')
    print('savinig the history...')
    pickle.dump(network, open("/srv/deeptax/deeptax/data/history.pkl", 'wb'), protocol=4)
    pickle.dump(results_eval, open("/srv/deeptax/deeptax/data/results_eval.pkl", 'wb'), protocol=4)
    print('done...')

"""
opt = int(sys.argv[1])
type = 'nuc'
print('running option = ', opt)


if opt == 1:
    print('loading data...')
    info = pickle.load(open("/data/info.pkl", 'rb'))
    fname_train = '/data/train.txt'
    fname_val = '/data/val_train.txt'
    fname_test = '/data/test.txt'
    print('defining model:')

    ly = 128  # layer
    btch_size = 250
    epch = 20

    features = 20
    num_classes = info[0]
    max_len = info[1]
    nsamples_train = info[2]
    nsamples_val = info[3]
    nsamples_test = info[4]

    train_steps_per_epoch = np.ceil(nsamples_train / btch_size)
    val_steps_per_epoch = np.ceil(nsamples_val / btch_size)
    test_steps_per_epoch = np.ceil(nsamples_test / btch_size)

    print('features: ', features)
    print('clases: ', num_classes)
    print('max_length', max_len)
    print('layer nodes: ', ly)  # 128
    print('bacth size: ', btch_size)  # 2000
    print('epochs: ', epch)
    print('train steps per epoch: ', train_steps_per_epoch)
    print('val steps per epoch: ', val_steps_per_epoch)
    print('test steps per epoch: ', test_steps_per_epoch)

    model = Sequential()
    model.add(Embedding(max_len, features, input_length=max_len))
    model.add(Conv1D(ly, 9, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(ly, 9, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(num_classes, activation='softmax'))

    print('compiling the model...')
    # compile the model
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print('training the model...')
    # metric
    train_generator = bg.generator(fname_train, max_len, btch_size)
    validation_generator = bg.generator(fname_val, max_len, btch_size)
    network = model.fit_generator(train_generator,
                                  steps_per_epoch=train_steps_per_epoch,
                                  epochs=epch,
                                  validation_data=validation_generator,
                                  validation_steps=val_steps_per_epoch)  # 0.2

    test_generator = bg.generator(fname_test, max_len, btch_size)
    results_eval = model.evaluate_generator(test_generator, steps=test_steps_per_epoch)

    print('training the model... done!!!')
    print('savinig the history...')
    pickle.dump(network, open("/data/history.pkl", 'wb'), protocol=4)
    pickle.dump(results_eval, open("/data/results_eval.pkl", 'wb'), protocol=4)
    print('done...')

elif opt == 2:

    """
#    Create training and testing shuffled datasets 
"""
    fname = '/data/subdataset_RDP_nucl.fasta'

    sequence_df = pro_seq_fasta.process_fasta(fname, type)

    Y = np.array(sequence_df['bacteria'])
    X = np.array(sequence_df['sequence'])

    max_len = max([len(s) for s in X])
    classes = len(np.unique(Y))

    Y = s2v_tools.label2one_hot_encoding(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.20)

    nsamples_train = len(X_train)
    nsamples_val = len(X_val)
    nsamples_test = len(X_test)

    with open('/data/train.txt', 'w') as f:
        for i in range(len(X_train)):
            line = str(list(Y_train[i])).strip('[]').replace(', ', ',') + '\t' + str(X_train[i]).strip('[]').replace(
                ', ', ',') + '\n'
            f.write(line)
        f.close()

    with open('/data/val_train.txt', 'w') as f:
        for i in range(len(X_val)):
            line = str(list(Y_val[i])).strip('[]').replace(', ', ',') + '\t' + str(X_val[i]).strip('[]').replace(', ',
                                                                                                                 ',') + '\n'
            f.write(line)
        f.close()

    with open('/data/test.txt', 'w') as f:
        for i in range(len(X_test)):
            line = str(list(Y_test[i])).strip('[]').replace(', ', ',') + '\t' + str(X_test[i]).strip('[]').replace(', ',
                                                                                                                   ',') + '\n'
            f.write(line)
        f.close()

    info = (classes, max_len, nsamples_train, nsamples_val, nsamples_test)
    pickle.dump(info, open("/data/info.pkl", 'wb'), protocol=4)

