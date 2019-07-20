from trainUtils import segmentLung, genMask, stackSlices, plotScan
import pickle
import progressbar as pbar

with open('./dataset1_meta.p','rb') as df:
        cases = pickle.load(df)


bar = pbar.ProgressBar(max_value=len(cases))

for idx, name in enumerate(cases):
    
    print(name)

    case = cases[name]

    bar.update(idx+1)
    
    slices = stackSlices(case)
    mask,midx = genMask(case)

    seg = segmentLung(slices,threshold=-320)

    plotScan(seg,mask,midx)

    