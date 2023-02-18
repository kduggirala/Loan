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
bad_list = ['2005Q4', '2000Q2','2002Q3','2004Q1','2008Q3','2009Q1','2009Q3', '2001Q1']
qtrs = ['20{}Q{}'.format(str(i).zfill(2), j) for i in range(17) for j in range(1,5)]
qtrs = [q for q in qtrs if q not in bad_list]
qtrs_used = []
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
    global qtrs_used
    global min_index
    global max_index
    k = 0
    timesteps = lookforward//steps
    samples = np.zeros((batch_size, timesteps, 7))
    pos = np.zeros((10000, timesteps + 1, 8))
    neg = np.zeros((batch_size // 2, timesteps + 1, 8))
    targets = np.zeros((batch_size,))
    while True:
        if data is None or j >= max_index:
            q += 1
            if q == len(qtrs):
                q = 0
            qtr = qtrs[q]
            qtrs_used.append(qtr)
            f = qtr+'.npy.gz'
            os.system('aws s3 cp s3://loan-performance-data/numpy/'+f+' '+f)
            os.system('gunzip '+f)
            data = np.load(qtr+'.npy', mmap_mode = 'r')
            length = len(data)
            min_index = int(length*starting_from)
            max_index = int(length*ending_at)
            j = min_index
        i = 0
        count = 0
        while j < max_index:
            slice = np.nan_to_num(data[j, i:i+lookforward+steps:steps])
            if not (slice==np.zeros((timesteps+1,8))).all():
                target = slice[-1, 7]
                if target == 0:
                    neg[k] = slice
                    k += 1
                    j += 1
                    i = 0
                else:
                    pos[count] = slice
                    count += 1
                    i += 1
                if i+lookforward+steps > 215:
                    j += 1
                    i = 0
                if k == batch_size//2 or count == pos.shape[0]:
                    randselect = np.append(pos[np.random.randint(0, count, batch_size - k)], neg[:k], axis = 0)
                    np.random.shuffle(randselect)
                    samples = randselect[:,:timesteps,:7]
                    targets = randselect[:,-1,7]
                    yield samples, targets
                    k = 0
                    count = 0
        os.system('rm '+qtrs[q]+'.npy')

gen = generator(128, 2, 12)
next(gen)
sample = next(gen)
samples = sample[0]
length = 0
for i, slice in enumerate(samples):
    if sample[1][i] == 0:
        print(slice)
        print('\n')
        length += 1
print(length)
