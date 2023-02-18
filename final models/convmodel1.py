import numpy as np
import boto3
import tensorflow as tf
import os
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback

s3 = boto3.resource('s3')
kfold = 0

#partition is used for k-fold validation, indicating the number of parts the
#training data should be split into. At the end of each epoch, the global k-fold
#is incremented which shifts the window. validate indicates whether the generator
#is a validation generator or not
def generator(qtrs, batch_size, steps, lookforward, partition = 1, validate = True):
    global kfold
    k = 0
    q = 0
    timesteps = lookforward//steps
    samples = np.zeros((batch_size, timesteps, 22))
    targets = np.zeros((batch_size, 3))

    while True:
        q += 1
        if q == len(qtrs):
            q = 0
        qtr = qtrs[q]
        data = np.load(qtr+'.npy', mmap_mode = 'r')

        length = len(data)
        starting_from = (kfold % partition) / partition
        ending_at =  ((kfold % partition)/ partition) + (1 /partition)
        min_index = int(length*starting_from)
        max_index = int(length*ending_at)
        if not validate:
            skip_from = min_index
            skip_to = max_index
            min_index = 0
            max_index = length

        j = min_index
        i = 0
        count = 0
        weights = np.array([0.13736523, 1, 0.0020757]) #weights for prepayment, default, and other
        while j < max_index:
            rand_sample = random.random()
            slice = data[j, i:i+lookforward+steps:steps]
            the_target = slice[-1,22:25]
            if rand_sample < np.dot(the_target, weights):
                samples[k] = slice[:-1, :22]
                targets[k] = the_target
                k += 1
            i += 1
            if not the_target.any() or i+lookforward+steps > 240:
                j += 1
                i = 0
            if k == batch_size:
                yield samples, targets
                k = 0
            if not validate and j == skip_from:
                j = skip_to

class KFoldCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        global kfold
        kfold += 1

j = 8
def make_model(qtrs_train, batch_size, lookforward, steps, epoch_steps, val_steps, testing_steps, partition, neurons, dropout, optimizer, epochs):
    global j

    train_gen = generator(qtrs_train, batch_size = batch_size, steps = steps, lookforward = lookforward, partition = partition, validate = False)
    validation_gen = generator(qtrs_train, batch_size = batch_size, steps = steps, lookforward = lookforward, partition = partition)

    model = Sequential()
    model.add(layers.Conv1D(32, 3, activation = 'relu', input_shape = (None, 22)))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Conv1D(32, 3, activation = 'relu'))
    model.add(layers.SpatialDropout1D(0.5))
    model.add(layers.LSTM(32, dropout = dropout, recurrent_dropout = dropout))
    model.add(layers.Dense(3, activation = tf.nn.softmax))

    model.compile(optimizer = optimizer, loss = keras.losses.CategoricalCrossentropy(), metrics = ['acc'])
    my_callbacks = [KFoldCallback()]
    model.fit(train_gen, steps_per_epoch = epoch_steps, epochs = epochs, validation_data = validation_gen, validation_steps = val_steps, callbacks = my_callbacks)

    model_name = 'model{}.h5'.format(j)
    j += 1
    model.save(model_name)
    s3.Bucket('loan-performance-data').upload_file(model_name, 'saved_models/conv_models/'+model_name)

qtrs_train = ['20{}Q{}'.format(str(i).zfill(2), j) for i in range(2) for j in range(1,5)]

#os.system('rm -f *.npy')
#for qtr in qtrs_train:
    #f = qtr+'.npy.gz'
    #os.system('aws s3 cp s3://loan-performance-data/numpy4/'+f+' '+f)
    #os.system('gunzip -f '+f)

batch_size = 64
dropout = 0.2
neurons = 20
learning_rates = [1e-3]

i = 0
for learning_rate in learning_rates:
  make_model(qtrs_train, batch_size, 12, 1, 5000, 500, 2500, 5, neurons, dropout, optimizers.RMSprop(learning_rate = learning_rate), 100)
