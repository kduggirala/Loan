qtrs = ['20{}Q{}'.format(str(i).zfill(2), j) for i in range(1,17) for j in range(1,5)]
s3 = boto3.resource('s3')
for qtr in qtrs:
    q = qtr+'.npy.gz'
    os.system('aws s3 cp s3://loan-performance-data/numpy/'+q+' '+q)
    os.system('gunzip '+q)
    data = np.nan_to_num(np.load(qtr+'.npy'))
    q = qtr+'new.npy.gz'
    f = gzip.GzipFile(q, "w")
    np.save(f, data)
    s3.Bucket('loan-performance-data').upload_file(q, 'numpy2/'+qtr+'.npy.gz')
    os.system('rm '+qtr+'.npy')
    os.system('rm '+qtr+'new.npy.gz')
