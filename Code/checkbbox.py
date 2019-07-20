import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import trainUtils as ut

dbpath = '../dbHdf5/dataset3_2d_onlyTumor_cropped_x-74-426_y-74-426_clipped-0-1800_wavelet_scaled_aug-R90-R180-R270-Hf-Vf-Et_train.hdf5'
with h5py.File(dbpath,'r') as db:

    cases = db['case'][...]
    bbox = db['bbox'][...]

    idx = 0
    k = None
    while k != 'q':
        
        fig, ax = plt.subplots(1)

        s = db['slice'][idx, ...]
        m = db['mask'][idx, ...]
        ax.imshow(m)
 
        y1, x1, y2, x2 = np.int16(bbox[idx, ...] * m.shape[0])
        h = y2-y1
        w = x2-x1

        print( "%d,%d to %d,%d" %(x1,y1,x2,y2))
        rect = patches.Rectangle( (x1,y1), w,h, linewidth = 1, edgecolor='r',facecolor='none' )
        ax.add_patch(rect)

        plt.show()

        ut.plotScan(np.expand_dims(s,-1), np.expand_dims(m,-1), [0])
        k = input("q for quit")
        idx = int(k)
        idx = (len(cases)-1) if idx >= len(cases) else idx
