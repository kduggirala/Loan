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

#os.system('rm -f *.npy')
qtrs_test = ['2002Q1']
#for qtr in qtrs_test:
    #f = qtr+'.npy.gz'
    #os.system('aws s3 cp s3://loan-performance-data/numpy4/'+f+' '+f)
    #os.system('gunzip -f '+f)

testing_gen = generator(qtrs_test, batch_size = 64, steps = 1, lookforward = 12)
testing_steps = 5000


for j in range(8,13):
    f = 'model{}.h5'.format(j)
    os.system('aws s3 cp s3://loan-performance-data/saved_models/rolling_models3/'+f+' '+f)
    model = keras.models.load_model(f)
    model.evaluate(testing_gen, steps = testing_steps)
