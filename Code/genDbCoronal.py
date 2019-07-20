import trainUtils as ut
import pickle
import os
import numpy as np

dbMetaDataPath = 'dataset3_train_metadata.p'

onlyTumor = True

# HU Clipping
upperClip = 1800
lowerClip = 0

# cropping
cx1, cx2 = (74, 426)
cy1, cy2 = (0, 352)

# resizing
resizeTo = (224,224)

# rescale
rescale = True

# perform wavelet transform
includeWavelet = False

# mask expanion
expandMasks = False  # mask expansion through dialation by elliptical kernel
minArea = 300       # pixel squared
applyGaussianFilter = False

# agumentations : 'rot90', 'rot180', 'rot270' 'horFlip', 'verFlip', 'elasticTransform', 'compound' (for compounding augmentations)
augments = ['rot90', 'rot180', 'rot270', 'horFlip', 'verFlip', 'elasticTransform']
# augments = []

# no. of stacks to include for 3d stacks
nSlices = 2
sliceStep = 1
maskType = '3D'

#aug-R90-R180-R270-Hf-Vf-Et_
dbName = 'dataset3_2d_onlyTumor_coronal_cropped_x-74-426_y-0-352_clipped-0-1800_resized-224-224_aug-R90-R180-R270-Hf-Vf-Et_scaled_train'

dbSavePath = os.path.join('..','dbHdf5', dbName + '.hdf5')
with open(dbMetaDataPath,'rb') as df:
    dbMetaData = pickle.load(df)

ut.createDatabase2D_Coronal(dbMetaData, dbSavePath, wavelet=includeWavelet, onlyTumor=onlyTumor, crop=[cx1,cy1,cx2,cy2], clipHU=(lowerClip,upperClip), resizeTo=resizeTo,
                    rescale=rescale, expandMasks=expandMasks, minArea=minArea, applyGaussianFilter=applyGaussianFilter, augments=augments)

