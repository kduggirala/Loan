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
        os.system('rm *.npy')

qtrs = ['20{}Q{}'.format(str(i).zfill(2), j) for i in range(1) for j in range(1,5)]
train_gen = generator(qtrs, 128, 1, 6)
val_gen = generator(['2001Q1'], 128, 1, 6)
next(train_gen)
samples = next(val_gen)
sample = samples[0]
length = 0
for i, slise in enumerate(sample):
    if samples[1][i] == 0:
        print(slise)
        print('\n')
        length += 1
print(length)
