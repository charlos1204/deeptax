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
import deeptax.models.batchgenerator as bg


"""
Author: Carlos Garcia-Perez
Date: 26.06.2019 1D CNN for sequence classification
                 first version of the script
"""

def runtraining():
    print('loading data...')
    info = pickle.load(open("/srv/deeptax/data/info.pkl", 'rb'))
    fname_train = '/srv/deeptax/data/train.txt'
    fname_val = '/srv/deeptax/data/val_train.txt'
    fname_test = '/srv/deeptax/data/test.txt'
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

    # serialize model to JSON
    model_json = model.to_json()
    with open("/srv/deeptax/models/model.json", "w") as json_file:
        json_file.write(model_json)
    
    # serialize weights to HDF5
    model.save_weights("/srv/deeptax/models/model.h5")
    print("Saved model to disk...")

    print('training the model... done!!!')
    print('savinig the history...')
    pickle.dump(network, open("/srv/deeptax/models/history.pkl", 'wb'), protocol=4)
    pickle.dump(results_eval, open("/srv/deeptax/models/results_eval.pkl", 'wb'), protocol=4)
    print('done...')


