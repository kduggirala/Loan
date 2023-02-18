import numpy as np
import boto3
import os

s3 = boto3.resource('s3')
q = 0
bad_list = ['2005Q4', '2000Q2','2002Q3','2004Q1','2008Q3','2009Q1','2009Q3', '2001Q1']
qtrs = [q for q in ['20{}Q{}'.format(str(i).zfill(2), j) for i in range(16,17) for j in range(1,5)] if q not in bad_list]
count = 0
for qtr in qtrs:
    f = qtr+'.npy.gz'
    os.system('aws s3 cp s3://loan-performance-data/numpy/'+f+' '+f)
    os.system('gunzip '+f)
    data = np.load(qtr+'.npy', mmap_mode = 'r')
    count += data.shape[0]
    os.system('rm '+qtr+'.npy')


print(count)
