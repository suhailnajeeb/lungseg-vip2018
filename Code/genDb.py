import trainUtils as ut
import pickle
import os
import numpy as np

dbMetaDataPath = 'dataset3_train_metadata.p'

onlyTumor = True
nonTumorSlices = 1

# mask regions to include
maskRegions = ['roi', 'Lungs', 'Patient']

# HU Clipping
upperClip = 1800
lowerClip = 0

# cropping
cx1, cx2 = (74, 426)
cy1, cy2 = (74, 426)

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
# augments = ['rot90', 'rot180', 'rot270', 'horFlip', 'verFlip', 'elasticTransform']
augments = []

# no. of stacks to include for 3d stacks
nSlices = 4
sliceStep = 1
maskType = '3D'

#aug-R90-R180-R270-Hf-Vf-Et_
# dbName = 'dataset3_2d_non-tumor-1_cropped_x-74-426_y-74-426_clipped-0-1800_resized-224-224_aug-R90-R180-R270-Hf-Vf-Et_scaled_train'
# dbName = 'dataset3_2d_non-tumor-10_cropped_x-74-426_y-74-426_clipped-0-1800_resized-224-224_aug-R90-R180-R270-Hf-Vf-Et_scaled_allregions_train'
dbName = 'dataset3_3d_nSlice-9_cropped_x-74-426_y-74-426_clipped-0-1800_resized-224-224_aug-R90-R180-R270-Hf-Vf-Et_scaled_train'
# dbName = 'dataset3_3d_nSlice-9_cropped_x-74-426_y-74-426_clipped-0-1800_resized-128-128_scaled_val'

dbSavePath = os.path.join('..','dbHdf5', dbName + '.hdf5')
with open(dbMetaDataPath,'rb') as df:
    dbMetaData = pickle.load(df)

# ut.createDatabase2D(dbMetaData, dbSavePath, maskRegions=maskRegions, wavelet=includeWavelet, onlyTumor=onlyTumor, nonTumorSlices=nonTumorSlices, crop=[cx1, cy1, cx2, cy2], clipHU=(lowerClip, upperClip), resizeTo=resizeTo,rescale=rescale, expandMasks=expandMasks, minArea=minArea, applyGaussianFilter=applyGaussianFilter, augments=augments)

ut.createDatabase3D(dbMetaData, dbSavePath, nSlices, sliceStep=sliceStep, maskType=maskType, onlyTumor=onlyTumor, crop=[cx1, cy1, cx2, cy2], clipHU=(lowerClip,upperClip), 
                    rescale=rescale, resizeTo=resizeTo, expandMasks=expandMasks, minArea=minArea, applyGaussianFilter=applyGaussianFilter, augments=augments)

# print(ut.getTotalEntryCount(dbMetaData, onlyTumor, augments))