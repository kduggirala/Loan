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
    samples = np.zeros((batch_size, timesteps, 7))
    pos = np.zeros((10000, timesteps + 1, 8))
    neg = np.zeros((batch_size // 2, timesteps + 1, 8))
    targets = np.zeros((batch_size,))
    while True:
        q += 1
        if q == len(qtrs):
            q = 0
        qtr = qtrs[q]
        f = qtr+'.npy.gz'
        os.system('aws s3 cp s3://loan-performance-data/numpy/'+f+' '+f)
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
                target = slice[-1, 7]
                if not target:
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
                    samples = randselect[:,:timesteps,:7]
                    targets = randselect[:,-1,7]
                    yield samples, targets
                    k = 0
                    count = 0
            else:
                j += 1
                i = 0
            if not validate and j == skip_from:
                j = skip_to
        os.system('rm *.npy')

already_seen = ['2007Q4','2016Q4','2010Q3','2008Q4','2011Q2','2011Q4','2006Q4','2013Q2','2012Q1','2008Q2','2007Q2','2012Q4','2016Q2','2009Q4','2016Q3','2005Q2','2015Q2','2012Q2','2004Q4','2011Q1','2010Q4','2012Q3','2013Q4','2015Q1','2008Q1','2014Q4','2013Q1', '2005Q4', '2000Q2','2002Q3','2004Q1','2008Q3','2009Q1','2009Q3']
qtrs = ['20{}Q{}'.format(str(i).zfill(2), j) for i in range(4,17) for j in range(1,5)]
testing_gen = generator([q for q in qtrs if q not in already_seen], 128, 2, 12, validate = True)

os.system('aws s3 cp s3://loan-performance-data/saved_models/worse_models/model9.h5 model9.h5')
model = keras.models.load_model('model9.h5')
model.evaluate(testing_gen, steps = 5000)
