import numpy as np
import boto3
import os

s3 = boto3.resource('s3')
good_list = ['2004Q4', '2010Q1', '2005Q4', '2015Q1', '2005Q4', '2000Q2','2002Q3','2004Q1','2008Q3','2009Q1','2009Q3']
qtrs = ['20{}Q{}'.format(str(i).zfill(2), j) for i in range(18) for j in range(1,5)]
qtrs = [q for q in qtrs if q not in good_list]
bad_qtrs = []
for qtr in qtrs:
    f = qtr+'.npy.gz'
    os.system('aws s3 cp s3://loan-performance-data/numpy/'+f+' '+f)
    os.system('gunzip '+f)
    data = np.load(qtr+'.npy', mmap_mode = 'r')
    count = 0
    for i in range(10):
        slice = np.nan_to_num(data[i, 0:14:2])
        interest = slice[:,0]
        if (interest==np.zeros((7,))).all():
            count += 1
    if count > 1:
        bad_qtrs.append(qtr)
    os.system('rm '+qtr+'.npy')

for bad_qtr in bad_qtrs:
    print(bad_qtr)
