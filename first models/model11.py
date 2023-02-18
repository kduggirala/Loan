import numpy as np
import boto3
import tensorflow as tf
import os
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop

s3 = boto3.resource('s3')
q = 0
qtrs = ['20{}Q{}'.format(str(i).zfill(2), j) for i in range(3) for j in range(1,5)]
random.shuffle(qtrs)
data = None
min_index = 0
max_index = 0
j = 0
#starting from and ending at are measured as a fraction of the length of the
#data, which varies. So taking a sample from the lower quartile to the upper
#quartile would be starting from  0.25 and ending at 0.75
def generator(batch_size, steps, lookforward, starting_from = 0, ending_at = 1):
    global q
    global data
    global j
    global qtrs
    global min_index
    global max_index
    k = 0
    samples = np.zeros((batch_size, (lookforward//steps) - 1, 7))
    targets = np.zeros((batch_size,))
    while True:
        if data is None or j >= max_index:
            q += 1
            if q == len(qtrs):
                q = 0
            qtr = qtrs[q]
            f = qtr+'.npy.gz'
            os.system('aws s3 cp s3://loan-performance-data/numpy/'+f+' '+f)
            os.system('gunzip '+f)
            data = np.load(qtr+'.npy', mmap_mode = 'r')
            length = len(data)
            min_index = int(length*starting_from)
            max_index = int(length*ending_at)
            j = min_index
        i = 0
        while j < max_index:
            slice = np.nan_to_num(data[j, i:i+lookforward:steps])
            samples[k] = slice[:(lookforward//steps) - 1, :7]
            targets[k] = slice[-1, 7]
            if targets[k] == 0 or i+1+lookforward+steps > 215:
                j += 1
                i = 0
            else:
                i += 1
            k += 1
            if k == batch_size:
                yield samples, targets
                k = 0
        os.system('rm '+qtrs[q]+'.npy')

batch_size = 128
steps = 2
lookforward = 12
epoch_steps = 10000
val_steps = 2000
testing_steps = 2000
dropout = 0.2
epochs = 10


model = Sequential()
model.add(layers.LSTM(32, dropout = dropout, recurrent_dropout = dropout, input_shape = (lookforward//steps, 7)))
model.add(layers.Dense(1, activation = 'sigmoid'))

train_gen = generator(batch_size = batch_size, steps = steps, lookforward = lookforward)
validation_gen = generator(batch_size = batch_size, steps = steps, lookforward = lookforward)
testing_gen = generator(batch_size = batch_size, steps = steps, lookforward = lookforward)
model.compile(optimizer = RMSprop(), loss = 'binary_crossentropy', metrics = ['acc'])
model.fit(train_gen, steps_per_epoch = epoch_steps, epochs = epochs, validation_data = validation_gen, validation_steps = val_steps)

model.evaluate(testing_gen, steps = testing_steps)
model.save('model11.h5')
s3.Bucket('loan-performance-data').upload_file('model11.h5', 'saved_models/model11.h5')
