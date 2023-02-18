import numpy as np
import boto3
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback
import os
import random

s3 = boto3.resource('s3')
kfold = 0
#partition is used for k-fold validation, indicating the number of parts the
#training data should be split into. At the end of each epoch, the global k-fold
#is incremented which shifts the window. validate indicates whether the generator
#is a validation generator or not
def generator(qtrs, batch_size, steps, lookforward, partition = 1, validate = True):
    global kfold
    k = 0
    q = -1
    timesteps = lookforward//steps
    samples = np.zeros((batch_size, timesteps, 22))
    targets = np.zeros((batch_size, 3))
    while True:
        q += 1
        if q == len(qtrs):
            q = 0
        qtr = qtrs[q]
        os.system('rm -f *.npy')
        os.system('rm -f *.npy.gz')
        f = qtr+'.npy.gz'
        os.system('aws s3 cp s3://loan-performance-data/numpy4/'+f+' '+f)
        os.system('gunzip -f '+f)
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
        weights = np.array([0.13736523, 1, 0.0020757])
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

j = 0
def make_model(qtrs_train, batch_size, lookforward, steps, epoch_steps, val_steps, testing_steps, partition, num_layers, neurons, dropout, optimizer, epochs):
    global j

    train_gen = generator(qtrs_train, batch_size = batch_size, steps = steps, lookforward = lookforward, partition = partition, validate = False)
    validation_gen = generator(qtrs_train, batch_size = batch_size, steps = steps, lookforward = lookforward, partition = partition)

    model = Sequential()
    model.add(layers.LSTM(neurons, dropout = dropout, recurrent_dropout = dropout, input_shape = ((lookforward//steps), 22), return_sequences = num_layers > 1))
    for i in range(num_layers - 1):
        model.add(layers.LSTM(neurons, dropout = dropout, recurrent_dropout = dropout, return_sequences = i < num_layers - 2))
    model.add(layers.Dense(3, activation = tf.nn.softmax))

    model.compile(optimizer = optimizer, loss = keras.losses.CategoricalCrossentropy(), metrics = ['acc'])
    my_callbacks = [KFoldCallback()]
    model.fit(train_gen, steps_per_epoch = epoch_steps, epochs = epochs, validation_data = validation_gen, validation_steps = val_steps, callbacks = my_callbacks)

    model_name = 'model{}.h5'.format(j)
    j += 1
    model.save(model_name)
    s3.Bucket('loan-performance-data').upload_file(model_name, 'saved_models/full_models/'+model_name)

qtrs_train = ['20{}Q{}'.format(str(i).zfill(2), j) for i in range(19) for j in range(1,5)]

layer_count = 2
batch_size = 64
neurons = 20
dropout = 0.15
learning_rate = 1e-3

make_model(qtrs_train, batch_size, 12, 1, 32000, 8000, 2500, 5, layer_count, neurons, dropout, optimizers.RMSprop(learning_rate = learning_rate), 10)
