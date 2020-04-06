#import numpy as np
#import pickle
#import sequence2vector as s2v_tools

#y_data_name = '/data/label_dataset.pkl'#

#Y = pickle.load(open(y_data_name, 'rb'))

#print(Y.shape)

#Y = s2v_tools.seq2vectorize(Y)
#print(Y)


from keras.models import Sequential
import plotresults as pltrslts
import pickle

network = pickle.load(open("/data/history.pkl", 'rb'))

pltrslts.plot_acc_loss(network, 'save')
