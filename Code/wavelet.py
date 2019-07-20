import pywt
import pickle
import numpy as np
import trainUtils as ut
import matplotlib.pyplot as plt
from matplotlib import animation

with open('dataset1_meta.p') as df:
    db = pickle.load(df)

cases = sorted(db.keys())

def plotWavlet(wav,s):

    fig = plt.figure()
    
    def update(idx):
        w = wav[idx]
        ax = plt.subplot(1,6,1)
        ax.clear()
        ax.imshow(s[..., idx])
        ax = plt.subplot(1, 6, 2)
        ax.clear()
        ax.imshow(w[0][0])
        for i in range(3):
            ax = plt.subplot(1, 6, i+3)
            ax.clear()
            ax.imshow(w[0][1][i])

        ax = plt.subplot(1, 6, 6)
        ax.clear()
        ax.imshow(w[0][1][0] + w[0][1][1] + w[0][1][2])

    anim = animation.FuncAnimation(fig, update, interval=100, repeat=True,frames=len(wav))
    plt.show()

cdx = 15
s = ut.stackSlices(db[cases[cdx]])
m, midx = ut.genMask(db[cases[cdx]])

wav = []
for m in midx:
    s1 = s[..., m]
    wav.append(pywt.swt2(s1, 'haar', level=1))
    
plotWavlet(wav,s[...,midx])


