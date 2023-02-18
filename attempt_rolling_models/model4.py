import numpy as np
import boto3
import tensorflow as tf
import os
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import Callback



s3 = boto3.resource('s3')
kfold = 0

#partition is used for k-fold validation, indicating the number of parts the
#training data should be split into. At the end of each epoch, the global k-fold
#is incremented which shifts the window. validate indicates whether the generator
#is a validation generator or not
def generator(qtrs, batch_size, steps, lookforward, partition = 1, validate = False):
    global kfold
    k = 0
    q = 0
    timesteps = lookforward//steps
    samples = np.zeros((batch_size, timesteps, 21))
    pos = np.zeros((10000, timesteps + 1, 22))
    neg = np.zeros((batch_size // 2, timesteps + 1, 22))
    targets = np.zeros((batch_size,))
    while True:
        os.system('rm -f *.npy')
        q += 1
        if q == len(qtrs):
            q = 0
        qtr = qtrs[q]
        f = qtr+'.npy.gz'
        os.system('aws s3 cp s3://loan-performance-data/numpy3/'+f+' '+f)
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
        while j < max_index:
            slice = np.nan_to_num(data[j, i:i+lookforward+steps:steps])
            if slice[-1,0]:
                target = slice[-1, 21]
                if target:
                    neg[k] = slice
                    k += 1
                    j += 1
                    i = 0
                else:
                    pos[count] = slice
                    count += 1
                    i += 1
                if i+lookforward+steps > 240:
                    j += 1
                    i = 0
                if k == batch_size//2 or count == pos.shape[0]:
                    randselect = np.append(pos[np.random.randint(0, count, batch_size - k)], neg[:k], axis = 0)
                    np.random.shuffle(randselect)
                    samples = randselect[:,:timesteps,:21]
                    targets = randselect[:,-1,21]
                    yield samples, targets
                    k = 0
                    count = 0
            else:
                j += 1
                i = 0
            if not validate and j == skip_from:
                j = skip_to

qtrs_train = ['20{}Q{}'.format(str(i).zfill(2), j) for i in range(1) for j in range(1,5)]
qtrs_test = ['2001Q1']
epochs = 10
batch_size = 128
steps = 1
lookforward = 6
epoch_steps = 10000
val_steps = 2000
testing_steps = 5000
dropout = 0.2
partition = 5


model = Sequential()
model.add(layers.LSTM(64, dropout = dropout, recurrent_dropout = dropout, input_shape = ((lookforward//steps), 21), return_sequences = True))
model.add(layers.LSTM(64, dropout = dropout, recurrent_dropout = dropout))
model.add(layers.Dense(1, activation = 'sigmoid'))

train_gen = generator(qtrs_train, batch_size = batch_size, steps = steps, lookforward = lookforward, partition = partition)
validation_gen = generator(qtrs_train, batch_size = batch_size, steps = steps, lookforward = lookforward, partition = partition, validate = True)
testing_gen = generator(qtrs_test, batch_size = batch_size, steps = steps, lookforward = lookforward, validate = True)
model.compile(optimizer = RMSprop(), loss = 'binary_crossentropy', metrics = ['acc'])

class KFoldCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        global kfold
        kfold += 1

my_callbacks = [KFoldCallback()]
model.fit(train_gen, steps_per_epoch = epoch_steps, epochs = epochs, validation_data = validation_gen, validation_steps = val_steps, callbacks = my_callbacks)

model.evaluate(testing_gen, steps = testing_steps)
model.save('model2000v4.h5')
s3.Bucket('loan-performance-data').upload_file('model2000v4.h5', 'saved_models/rolling_models/model2000v4.h5')
