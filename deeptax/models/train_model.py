import sys
import process_sequence_fasta as pro_seq_fasta
import sequence2vector as s2v_tools
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.layers import Embedding
from keras.preprocessing import sequence
import batchgenerator as bg

"""
Author: Carlos Garcia-Perez
Date: 25.06.2019 final version of model setting.
                 add new option to train all data set no splitting.
                 save the model and the weight separated
      17.06.2019 fix the model definition
      14.06.2019 split data set in train, validation and test sets in option 1
      13.06.2019 create data set in one-hot-encoding and save as object.pkl in option 2
                 first version of the script
"""
opt = int(sys.argv[1])
type = 'nuc' # aa

print('running option = ', opt)

if opt == 1:
    print('processing all...')
    x_data_name = '/data/sequence_dataset.pkl'
    y_data_name = '/data/label_dataset.pkl'

    X = pickle.load(open(x_data_name, 'rb'))
    Y = pickle.load(open(y_data_name, 'rb'))

    classes = pickle.load(open("/data/classes.pkl", 'rb'))

    print('defining model:')

    features = 20
    num_classes = classes

    print('features: ', features)
    print('clases: ', num_classes)
    print('nodes: ', 128)
    print('bacth size: ', 2000)
    print('epochs: ', 50)

    print('reshaping data...')
    max_len = max([len(s) for s in X])
    X_train = sequence.pad_sequences(X, maxlen=max_len)

    print('training dataset: ', X_train.shape)
    print('max_length:', max_len)

    model = Sequential()
    model.add(Embedding(len(X_train), features, input_length=max_len))
    model.add(LSTM(128))  # 32
    model.add(Dense(num_classes, activation='softmax'))

    print('compiling the model...')
    # compile the model
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print('training the model...')
    # metric
    model.fit(X_train, Y,
              epochs=50,
              batch_size=2000)

    results_eval = model.evaluate(X_train, Y, batch_size=2000)

    print("%s: %.2f%%" % (model.metrics_names[1], results_eval[1] * 100))

    pickle.dump(results_eval, open("/data/results_eval.pkl", 'wb'), protocol=4)

    # serialize model to JSON
    model_json = model.to_json()
    with open("/data/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("/data/model.h5")
    print("Saved model to disk...")

elif opt == 2:
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
    model.add(LSTM(ly))  # 128
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

elif opt == 3:


    """
    Create training and testing shuffled datasets 
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
            line = str(list(Y_train[i])).strip('[]').replace(', ', ',') + '\t' + str(X_train[i]).strip('[]').replace(', ', ',') + '\n'
            f.write(line)
        f.close()

    with open('/data/val_train.txt', 'w') as f:
        for i in range(len(X_val)):
            line = str(list(Y_val[i])).strip('[]').replace(', ', ',') + '\t' + str(X_val[i]).strip('[]').replace(', ', ',') + '\n'
            f.write(line)
        f.close()

    with open('/data/test.txt', 'w') as f:
        for i in range(len(X_test)):
            line = str(list(Y_test[i])).strip('[]').replace(', ', ',') + '\t' + str(X_test[i]).strip('[]').replace(', ', ',') + '\n'
            f.write(line)
        f.close()

    info = (classes, max_len, nsamples_train, nsamples_val, nsamples_test)
    pickle.dump(info, open("/data/info.pkl", 'wb'), protocol=4)

