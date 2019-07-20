import numpy as np
import trainUtils as ut
from progressbar import ProgressBar

dbTest = ut.loadMetaData('dataset3_val_metadata.p')
dbAll = ut.loadMetaData('nsclc_metadata.p')

kTest = sorted(list(dbTest.keys()))
kAll = sorted(list(dbAll.keys()))

sTest = {}
sAll = {}

print("Getting test slices")
bar = ProgressBar(max_value=len(kTest))
for idx, k in enumerate(kTest):
    bar.update(idx+1)
    sTest[k] = ut.stackSlices(dbTest[k])[...,50]

print("\nGetting nsclc slices")
bar = ProgressBar(max_value=len(kAll))
for idx, k in enumerate(kAll):
    bar.update(idx+1)
    sAll[k] = ut.stackSlices(dbAll[k])[...,50]


mapping = {}
print("\nComparing")
bar = ProgressBar(max_value=len(kTest))
for idx, k1 in enumerate(kTest):

    bar.update(idx+1)

    matched = False
    for k2 in kAll:

        if (np.sum(sTest[k1] - sAll[k2]) == 0):
            mapping[k1] = k2
            matched = True
            break
        else:
            continue
    
    if not matched:
        mapping[k1] = None

print(mapping)
