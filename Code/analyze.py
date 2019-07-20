
from trainUtils import *
from skimage import measure
from progressbar import ProgressBar
from matplotlib import pyplot as plt
from skimage.measure import regionprops
from pprint import pprint
import pickle
import itertools
from correctedAnnots import *
import sys


functions = {
	'genMeta': 0,
	'plotCenter': 1,
	'HUFreq': 2,
	'countTumorSlices': 3,
	'plotSlices': 4,
	'minMax' : 5,
	'genRegionProps':6,
}

if len(sys.argv) < 2:
	print("Incorrect usage, need argument :")
	pprint(functions)
	sys.exit(1)

func = functions[sys.argv[1]]

if func == 0:
	# generating Meta Data
	dataFolder = '../Dataset3/VIP_CUP18_TrainingData'
	data = getMetaData3(dataFolder,savePath='./dataset3_train_metadata_allregions.p')	# use getMetaData2 for new dataset

else:
	# loading case meta data
	dataMetaPath = 'dataset3_%s_metadata.p' % sys.argv[2]
	with open(dataMetaPath, 'rb',) as df:
		dataMeta = pickle.load(df)

	# loading region props data
	if func in [1,5]:
		dataRegionPath = 'dataset3_%s_regionProps.p' % sys.argv[2]
		with open(dataRegionPath ,'rb') as df:
			dataRegion = pickle.load(df, encoding='latin1')

# plotting centroid of all masks over entire dataset
if func == 1:

	x,y,caseID = [], [], []
	bar = ProgressBar(max_value=len(dataRegion))
	for idx,case in enumerate(dataRegion):
		bar.update(idx + 1)

		if case in corrections:
			if corrections[case] is not None:
				yCorr, xCorr = corrections[case]
				print(yCorr, xCorr)
			else:
				print("Skipping %s"%case)
				continue
		else:
			yCorr, xCorr = 0, 0


		slices = dataRegion[case]

		for s in slices:
			props = slices[s][0]
			x.append( (props['centroid'][1] + xCorr) ) 
			x.append( (props['bbox'][0] + xCorr) )
			x.append( (props['bbox'][2] + xCorr) )
			y.append( (props['centroid'][0] + yCorr))
			y.append( (props['bbox'][1] + yCorr))
			y.append( (props['bbox'][3] + yCorr))
			
			caseID.extend([case]*3)

	# filtering
	xf,yf,cf = [],[],[]

	for x0,y0,c0 in zip(x,y,caseID):

		if True  :
			xf.append(x0)
			yf.append(y0)
			cf.append(c0)
		
	plt.figure()
	plt.scatter(xf,yf)

	# for i,c in enumerate(cf):
		# plt.annotate(c,(xf[i],yf[i]))

	print(np.unique(cf))
	print(len(xf))
	plt.ylim(0,512)
	plt.xlim(0,512)
	plt.grid()
	plt.show()

# Calculating Frequecny of HU values over entire dataset
if func == 2:

	bar = ProgressBar(max_value=len(dataMeta))
	freq = None

	for idx, x in enumerate(dataMeta):

		bar.update(idx+1)
		case = dataMeta[x]
		slices = stackSlices(case)
		slices = (slices - 1024) if np.min(slices) >= 0 else slices
		print(slices.shape)
		slices = slices.flatten()

		if freq is None:
			freq,bins = np.histogram(slices, range=(-2000,3000),bins =80, density = False)
		else:
			freq += np.histogram(slices,range=(-2000,3000),bins = 80, density = False)[0]
		
	plt.figure()
	plt.bar(bins[:-1],freq, width= (bins[-1]-bins[0])/len(bins)  ) 
	plt.grid()
	plt.show()

# counting total (n)umber of slices and (t)umor slices
if func == 3:
	
	bar = ProgressBar(max_value=len(dataMeta))

	nSlice, tSlice = 0, 0

	for idx,caseName in enumerate(dataMeta):

		bar.update(idx+1)

		case = dataMeta[caseName]

		nSlice += len(case['scans'])
		tSlice += len(case['roi'])

	print("Total Slices : %d" % nSlice)
	print("Tumor Slices : %d" % tSlice)

# plotting slices and masks
if func == 4:

	print(dataMeta.keys())
	q = -1
	while q != 0:

		if sys.version[0] != '3' :
			q = raw_input("case ID : ")
		else:
			q = input("case ID : ")
		q = int(q)

		if q == 0:
			break

		# visualizing mask
		caseName = 'LUNG1-%03d'%q
	# for caseName in dataMeta:
		print(caseName)
		img = stackSlices(dataMeta[caseName])

		# # plotting histogram
		# plt.figure()
		# plt.hist( img.flatten(), bins = 80, color ='c')
		# plt.grid()

		# getting tumor mask and slice index in which tumor exist
		mask, maskSlices = genMask(dataMeta[caseName])

		# plotting
		plotScan(img, mask, maskSlices, threshold=0)

		# plotScan3D(img,mask,sidx=maskSlices)

# finding minimiums and maximums of tumor region
if func == 5:
	area, intensity = [], []
	bar = ProgressBar(max_value=len(dataRegion))
	for idx, case in enumerate(dataRegion):
		bar.update(idx + 1)

		if case in corrections:
			continue

		slices = dataRegion[case]

		for s in slices:
			props = slices[s][0]
			
			# if props['area'] > 100:
			# 	continue

			area.append(props['area'])
			intensity.append(props['mean_intensity'])
	
	area = np.array(area)
	intensity = np.array(intensity)
	
	print("\n Stats : \n ")
	print("- Min Area : %f px2" % np.min(area) )
	print("- Max Area : %f px2" % np.max(area) )
	print("- Mean Area : %f px2" % np.mean(area) )
	print("- Q2 Area : %f px2" % np.percentile(area,50) )
	print("- Q1 Area : %f px2" % np.percentile(area, 25))

	print("\n")
	print("- Min Intensity : %f HU" % np.min(intensity) )
	print("- Max Intensity : %f HU" % np.max(intensity) )
	print("- Mean Intensity : %f HU" % np.mean(intensity) )
	print("- Median Intensity : %f HU" % np.percentile(intensity,50) )

	plt.subplot(1,2,1)
	sortedArea = np.sort(area)
	plt.scatter(np.arange(len(area)), sortedArea, marker='o')
	plt.grid()
	plt.title("Area")

	plt.subplot(1,2,2)
	sortedIntensity = np.sort(intensity)
	plt.scatter(np.arange(len(intensity)), sortedIntensity, marker='o')
	plt.grid()
	plt.title("Intensity")
	
	plt.show()

# finding region properties
if func == 6:
	
	dataProps = {}
	bar = ProgressBar(max_value=len(dataMeta))
	for i, case in enumerate(dataMeta):
		bar.update(i+1)
		dataProps[case] = {}
		s = stackSlices(dataMeta[case])
		s = (s-1024) if np.min(s) >=0 else s	
		m,midx = genMask(dataMeta[case])

		for idx in midx:
			dataProps[case][idx] = regionprops(m[...,idx],s[...,idx])

	# dumping to disk
	savePath = "_".join(dataMetaPath.split('_')[0:2]) + "_regionProps.p"
	with open(savePath, 'wb') as df:
		pickle.dump(dataProps, df, protocol=2)
