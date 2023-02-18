import numpy as np
import boto3
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.optimizers import RMSprop
import os
import random

s3 = boto3.resource('s3')
kfold = 0
all = False
#partition is used for k-fold validation, indicating the number of parts the
#training data should be split into. At the end of each epoch, the global k-fold
#is incremented which shifts the window. validate indicates whether the generator
#is a validation generator or not
def generator(qtrs, batch_size, steps, lookforward, partition = 1, validate = True):
    global kfold
    global all
    k = 0
    q = -1
    timesteps = lookforward//steps
    samples = np.zeros((batch_size, timesteps, 22))
    targets = np.zeros((batch_size, 3))
    while True:
        q += 1
        if q == len(qtrs):
            all = True
            q = 0
        qtr = qtrs[q]
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
        os.system('rm *.npy')

qtrs = ['20{}Q{}'.format(str(i).zfill(2), j) for i in range(19) for j in range(1,5)]
train_gen = generator(qtrs, 128, 1, 12)

index = 0
for samples in train_gen:
    index += 1
    print('Update %d' % index, end='\r')
    if all:
        break

print(index)
#(12212271, 8704, 579025)
#(0.95285973203125, 0.0007948, 0.04634546796875)
#(0.9867170296875, 0.0002414, 0.0130415703125)
#(0.980742105, 0.001997, 0.017260895)
