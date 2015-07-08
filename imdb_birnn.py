from __future__ import absolute_import
from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.datasets import cifar10
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import *
from keras.datasets import imdb
import cPickle
import sys
from birnn import BiDirectionLSTM, Transform
'''
    Train a BiDirectionLSTM LSTM on the IMDB sentiment classification task.

    The dataset is actually too small for LSTM to be of any advantage
    compared to simpler, much faster methods such as TF-IDF+LogReg.

    Notes:

    - RNNs are tricky. Choice of batch size is important,
    choice of loss and optimizer is critical, etc.
    Most configurations won't converge.

    - LSTM loss decrease during training can be quite different
    from what you see with CNNs/MLPs/etc. It's more or less a sigmoid
    instead of an inverse exponential.

    GPU command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_lstm.py

    250s/epoch on GPU (GT 650M), vs. 400s/epoch on CPU (2.4Ghz Core i7).
'''

max_features=20000
maxseqlen = 100 # cut texts after this number of words (among top max_features most common words)
batch_size = 16
word_vec_len = 256

print("Loading data...")
(train_X, train_y), (test_X, test_y) = imdb.load_data(nb_words=max_features, test_split=0.2)
print(len(train_X), 'train sequences')
print(len(test_X), 'test sequences')

print("Pad sequences (samples x time)")
train_X = sequence.pad_sequences(train_X, maxlen=maxseqlen)
test_X = sequence.pad_sequences(test_X, maxlen=maxseqlen)

print('train_X shape:', train_X.shape)
print('test_X shape:', test_X.shape)


print('Build model...')
model = Sequential()
model.add(Embedding(max_features, word_vec_len))

# MLP layers
model.add(Transform((word_vec_len,))) # transform from 3d dimensional input to 2d input for mlp
model.add(Dense(word_vec_len, 100, activation='relu'))
model.add(BatchNormalization((100,)))
model.add(Dense(100,100,activation='relu'))
model.add(BatchNormalization((100,)))
model.add(Dense(100, word_vec_len, activation='relu'))
model.add(Transform((maxseqlen, word_vec_len))) # transform back from 2d to 3d for recurrent input

# Stacked up BiDirectionLSTM layers
model.add(BiDirectionLSTM(word_vec_len, 50, output_mode='concat'))
model.add(BiDirectionLSTM(100, 24, output_mode='sum'))

# MLP layers
model.add(Reshape(24 * maxseqlen))
model.add(BatchNormalization((24 * maxseqlen,)))
model.add(Dense(24 * maxseqlen, 50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, 1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='mean_squared_error', optimizer='adam', class_mode="binary")

print("Train...")
model.fit(train_X, train_y, batch_size=batch_size, nb_epoch=5, validation_data=(test_X, test_y), show_accuracy=True)

score = model.evaluate(test_X, test_y, batch_size=batch_size)
print('Test score:', score)

classes = model.predict_classes(test_X, batch_size=batch_size)
acc = np_utils.accuracy(classes, test_y)
print('Test accuracy:', acc)
