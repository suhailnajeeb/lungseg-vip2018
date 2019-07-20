import numpy as np
nSlice = 4
sliceStep =3

# x = np.zeros((5,5,30))
# y = np.zeros((5,5,30))

x = np.arange(0,100)
y = np.arange(0,100) * -1

# creating n-slice stacks
x_stacks = None
y_stacks = None
# stacking slices which have to be padded on top
for sidx in range(0,  x.shape[-1]):
    
    # center slice
    cen_slice = np.expand_dims(x[..., sidx],-1)
    cen_mask = np.expand_dims(y[..., sidx],-1)

    # getting index of top neighbours
    if sidx-(nSlice*sliceStep) >= 0:
        top_idx = list(range(sidx-(nSlice*sliceStep), sidx, sliceStep))
    else:
        top_idx = sorted(list(range(sidx-sliceStep, 0, -sliceStep)))

    # pad by repeating first slice
    if len(top_idx) < nSlice:
        nPad = nSlice - len(top_idx)
        padSlices = [0] * nPad
        padSlices.extend(top_idx)
        top_idx = padSlices

    top_slices = x[..., top_idx]
    top_masks = y[..., top_idx]
    
    # getting index of bottom neighbours
    if sidx+(nSlice*sliceStep) < x.shape[-1]:
        bot_idx = list(range(sidx+sliceStep, sidx+(nSlice*sliceStep)+1, sliceStep))
    else:
        bot_idx = list(range(sidx+sliceStep, x.shape[-1], sliceStep))

    # pad by repeating last slice
    if len(bot_idx) < nSlice:
        nPad = nSlice - len(bot_idx)
        bot_idx.extend([x.shape[-1]-1] * nPad)

    bot_slices = x[..., bot_idx]
    bot_masks = y[..., bot_idx]

    x0 = np.concatenate((top_slices,cen_slice,bot_slices),-1)
    y0 = np.concatenate((top_masks,cen_mask,bot_masks),-1) 

    if x_stacks is None:
        x_stacks = np.expand_dims(x0, -1)
        y_stacks = np.expand_dims(y0, -1)
    else:
        x_stacks = np.concatenate((x_stacks, np.expand_dims(x0, -1)),-1)
        y_stacks = np.concatenate((y_stacks, np.expand_dims(y0, -1)),-1)

for i in range(x_stacks.shape[-1]):
    print(x_stacks[...,i])
    print("==============")

print(x_stacks.shape)
