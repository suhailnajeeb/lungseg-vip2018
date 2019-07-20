import h5py
import numpy as np

# Path to dbs to merge
dbToMerge = ['../dbHdf5/dataset3_2d_onlyTumor_cropped_x-74-426_y-74-426_resized_224-224_clipped-0-1800_wavelet_scaled_val.hdf5',
             '../dbHdf5/dataset3_2d_onlyTumor_cropped_x-74-426_y-74-426_resized_224-224_clipped-0-1800_wavelet_scaled_val.hdf5']

mergeDbPath = './mergedDb.hdf5'

# height width
h = 224
w = 224

# geting total entry count
totalCount = 0
indiCount = []
for dbPath in dbToMerge:
    with h5py.File(dbPath, 'r') as db:
        totalCount += db['slice'].shape[0]
        indiCount.append(db['slice'].shape[0])

# creating new db
dbMerged = h5py.File(mergeDbPath, mode='w')
dbMerged.create_dataset("slice", (totalCount, h, w,), np.float32)
dbMerged.create_dataset("case",  (totalCount,), np.dtype('|S16'))
dbMerged.create_dataset("tumor", (totalCount,), np.uint8)

idx = 0
for didx, dbPath in enumerate(dbToMerge):
    with h5py.File(dbPath, 'r') as db:
        for jdx in range(indiCount[didx]):
            dbMerged['slice'][idx, ...] = db['slice'][jdx,...]
            dbMerged['tumor'][idx, ...] = db['tumor'][jdx, ...]
            dbMerged['case'][idx, ...] = db['case'][jdx, ...]

            idx +=1
            print("Copied %d / %d" % (idx,totalCount))
            
dbMerged.close()
