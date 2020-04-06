import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from keras.models import model_from_json
from keras.preprocessing import sequence


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')
    
    fig = plt.figure(figsize=(15,10))
    ax = fig.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
             rotation_mode='anchor')

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),fontsize=8,
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()

	

fname = sys.argv[1]

lbl_encoder = pickle.load(open('/data/lbl_encoder.pkl', 'rb'))
info = pickle.load(open('/data/info.pkl', 'rb'))
num_classes = info[0]
max_len = info[1]

X_test = pickle.load(open('/data/x_test.pkl', 'rb'))
X_test = sequence.pad_sequences(X_test, maxlen=max_len)
Y_test = pickle.load(open('/data/y_test.pkl', 'rb'))

class_names = lbl_encoder.inverse_transform([i for i in range(num_classes)])

json_file = open('/data/'+fname+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
model.load_weights('/data/'+fname+'.h5')

Y_pred = model.predict(X_test)

plot_confusion_matrix(Y_test.argmax(1),Y_pred.argmax(1), classes=class_names, normalize=True, title='Confusion matrix')

plt.savefig('/data/confusion_mtx' + fname + '.png', dpi = (200))

