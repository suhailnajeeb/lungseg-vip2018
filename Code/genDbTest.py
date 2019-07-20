import trainUtils as ut
import pickle
import os
import numpy as np

dbMetaDataPath = 'nsclc_metadata.p'

onlyTumor = True

# cropping
cx1, cx2 = (75, 425)
cy1, cy2 = (75, 425)

# resizing
resizeFactor = 1.0

# mask expanion
expandMasks = False  # mask expansion through dialation by elliptical kernel
minArea = 300       # pixel squared
applyGaussianFilter = False

# agumentations : 'rot90', 'rot180', 'rot270' 'horFlip', 'verFlip', 'elasticTransform', 'compound' (for compounding augmentations)
augments = []

dbName = 'dataset1_3d_n-2_mask-1_onlyTumor_cropped_x-75-425_y-75-425.hdf5'

dbSavePath = os.path.join('..', 'dbHdf5', dbName + '.hdf5')

with open(dbMetaDataPath, 'rb') as df:
    dbMetaData = pickle.load(df)

cases = dbMetaData.keys()

from trainList import *
cases_test = [c for c in cases if c not in trainList]
np.random.seed(1234)
random_cases = np.random.choice(cases_test, 100, replace=False)
dbMetaData_test = {c: dbMetaData[c] for c in random_cases}


ut.createDatabase2D(dbMetaData_test, dbSavePath, onlyTumor=onlyTumor, crop=[cx1, cy1, cx2, cy2], resizeFactor=resizeFactor,
                    expandMasks=expandMasks, minArea=minArea, applyGaussianFilter=applyGaussianFilter, augments=augments)

# print(ut.getTotalEntryCount(dbMetaData, onlyTumor, augments))


