import os
import sys
import cv2
import numpy as np
import glob
import trainUtils as ut
from pprint import pprint


resultFolder = sys.argv[1]

imgPaths = sorted(glob.glob(resultFolder + '/Masks/*'))

cases = sorted([os.path.basename(p) for p in imgPaths])

data = {}

print("Getting files ...")
for c in cases:
    imgs = glob.glob(resultFolder + '/Masks/' + c + '/*')
    data[c] = {}
    data[c]['slice'] = sorted([p for p in imgs if 'Slice' in p])
    data[c]['gt'] = sorted([p for p in imgs if 'GT' in p])
    data[c]['pred'] = sorted([p for p in imgs if (('GT' not in p) and ('Slice' not in p))])

while True:

    for i,c in enumerate(cases):
        print(c, end='\t')
        
        if not (i % 5):
            print("")
    
    if sys.version[0] == '3':
        k = input("\n\nEnter Patient ID <q to quit> : ")
        if k == 'q':
            break
    else:
        k = raw_input("\nEnter Patient ID <q to quit>: ")
        if k == ord('q'):
            break
    
    k = int(k)
    
    try:
        c = data['LUNG1-%03d' % k]
    except Exception as err:
        print(err)
        print("Patient ID not found in results folder")
        c = data[cases[0]]
        print("Showing %s\n" % cases[0])

    x = np.array([cv2.imread(s, -1) for s in c['slice']])
    yp = np.array([cv2.imread(s,0) for s in c['pred']])
    ygt = np.array([cv2.imread(s,0) for s in c['gt']])
    x = np.moveaxis(x,0,-1)
    ygt = np.moveaxis(ygt,0,-1)
    yp = np.moveaxis(yp,0,-1)

    if len(ygt) > 0 and len(x) > 0:
        ut.plotScan(x, [ygt, yp], colors=['r', 'b'], title=('LUNG1-%03d' % k))
    elif len(x) > 0:
        ut.plotScan(x, yp, colors=['r', 'b'], title=('LUNG1-%03d' % k))
    