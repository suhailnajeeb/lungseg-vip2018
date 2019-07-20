from __future__ import print_function
import os
import sys
import gc
import time
import glob
import pickle
import pydicom as dicom
import pywt
import numpy as np
from sklearn.model_selection import KFold
from skimage.draw import polygon
from skimage import measure, morphology

import scipy.ndimage
from scipy.ndimage import filters

import cv2
from albumentations.augmentations.transforms import ElasticTransform

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Slider, Button

import plotly
from plotly import figure_factory as FF
import plotly.graph_objs as go

import progressbar as pbar

import h5py

if sys.version_info[0] < 3:
	from Queue import Queue
else:
	from queue import Queue

import threading

from correctedAnnots import *
import metrics

# threshold values (HU scale) for different types of materials
thresholdHU = {'air':       (-2000, -1000),
			   'lung':      (-600, -500),
			   'fat':       (-100, -50),
			   'water':     (-10, 10),
			   'blood':     (30, 70),
			   'muscle':    (10, 40),
			   'liver':     (40, 60),
			   'bone':      (700, 3000) }

def ensureDir(filePath):
	
	''' This function checks if the folder at filePath exists.
		If not, it creates it. '''

	directory = os.path.dirname(filePath)

	if not os.path.exists(directory):
		os.makedirs(directory)

def getMetaData(pathToData, savePath = './metaData.p'):

	''' This function looks through the different patitent folders
	and extracts the metadata, including the ROI annotations and
	saves them in a dictionary indexed by case name 
	
	The dictionary can be dumped to disk at the savePath and loaded
	later to retreive all the information without having to loop over
	all cases again '''

	# getting all case folders
	caseFolders = sorted(glob.glob(pathToData + "/*"))

	# dictionary to record data
	data = {}

	# getting scan data and annotations
	bar = pbar.ProgressBar(max_value=len(caseFolders))      # progressbar to keep track

	for idx,case in enumerate(caseFolders):
		bar.update(idx+1)

		# getting folders inside each case folder
		studyFolder = glob.glob(case + '/*')[0]
		subFolders = glob.glob(studyFolder + '/*')
		
		# dcm files in two folders; one for scans, another for annotation
		try:
			dcmFilesA = glob.glob(subFolders[0] + '/*.dcm')
			dcmFilesB = glob.glob(subFolders[1] + '/*.dcm')
		except Exception as err:
			print(case, err)
			continue

		# scanData is in folder with more than one dcm file
		scanData = dcmFilesA if len(dcmFilesA) > 1 else dcmFilesB
		annotData = dcmFilesB[0] if len(dcmFilesA) > 1 else dcmFilesA[0]
		
		# extracting the case name and studyID from path
		caseName = os.path.basename(case)
		studyID = os.path.basename(studyFolder)

		# creating entry in dict
		data[caseName] = {}
		data[caseName]['id'] =  studyID
	   
		# sorting slices according to depth
		scanData.sort( key = lambda x : float(dicom.read_file(x).ImagePositionPatient[2]) )

		# parsing metadata from the scan files
		dcm = dicom.read_file(scanData[0])

		data[caseName]['manufacturer'] = dcm.Manufacturer
		data[caseName]['model'] = dcm.ManufacturerModelName
		
		data[caseName]['bodyPart'] = dcm.BodyPartExamined
		data[caseName]['rows'] = int(dcm.Rows)
		data[caseName]['cols'] = int(dcm.Columns)
		
		data[caseName]['sliceThickness'] = float(dcm.SliceThickness)
		data[caseName]['pixelSpacing'] = np.array(dcm.PixelSpacing)
		data[caseName]['imagePosition'] = np.array(dcm.ImagePositionPatient)

		data[caseName]['scans'] = { float(dicom.read_file(scan).ImagePositionPatient[2]) : scan 
									for scan in scanData }       

		depthSorted = sorted([depth for depth in data[caseName]['scans']])

		''' data not present n all files '''
		# data[caseName]['weight'] = dcm.PatientWeight
		# data[caseName]['age'] = dcm.PatientAge
		# data[caseName]['sex'] = dcm.PatientSex
		
		
		# parsing annotation data
		data[caseName]['roi'] = []
		dcm = dicom.read_file(annotData)

		try:
			for seq in dcm.ROIContourSequence[0].ContourSequence:

				try:
					roi = {}
					roi['idx'] = int(seq.ContourNumber)
					roi['color'] = np.array(dcm.ROIContourSequence[0].ROIDisplayColor)
					roi['geom'] = seq.ContourGeometricType                                          # contour geometry
					roi['nodes'] = np.array(seq.ContourData).reshape(-1,3)                          # list of coordinates (x,y,z)
					roi['sidx'] = depthSorted.index( roi['nodes'][0, 2] )                           # slice index
					roi['sname'] = os.path.basename( data[caseName]['scans'][roi['nodes'][0, 2]] )  # slice file name
			
				except ValueError as err:
					print(caseName, err, "Skipping ROI")
					continue
					
				data[caseName]['roi'].append(roi)
		
		except Exception as err:
			print(caseName, err, "Skipping Case")
			continue
	
	# dumping to disk
	with open( savePath, 'wb') as df:
		pickle.dump(data, df, protocol=2)

	return data

def getMetaData2(pathToData, savePath = './metaData.p'):

	''' USED FOR DATASET 3: 
	This function looks through the different patitent folders
	and extracts the metadata, including the ROI annotations and
	saves them in a dictionary indexed by case name 
	
	The dictionary can be dumped to disk at the savePath and loaded
	later to retreive all the information without having to loop over
	all cases again '''

	# getting all case folders
	caseFolders = sorted(glob.glob(pathToData + "/*"))

	# dictionary to record data
	data = {}

	# getting scan data and annotations
	bar = pbar.ProgressBar(max_value=len(caseFolders))      # progressbar to keep track

	for idx,case in enumerate(caseFolders):
		bar.update(idx+1)

		# getting folders inside each case folder
		subFolders = glob.glob(case + '/*/*')
		
		# dcm files in two folders; one for scans, another for annotation
		try:
			dcmFilesA = glob.glob(subFolders[0] + '/*.dcm')
			if len(subFolders) > 1:
				dcmFilesB = glob.glob(subFolders[1] + '/*.dcm')
		except Exception as err:
			print(case, err)
			continue

		if len(subFolders) > 1:
			# scanData is in folder with more than one dcm file
			scanData = dcmFilesA if len(dcmFilesA) > 1 else dcmFilesB
			annotData = dcmFilesB[0] if len(dcmFilesA) > 1 else dcmFilesA[0]
		else:
			scanData = dcmFilesA

		# extracting the case name and studyID from path
		caseName = os.path.basename(case)

		# creating entry in dict
		data[caseName] = {}
		data[caseName]['id'] =  caseName
	   
		# sorting slices according to depth
		scanData.sort( key = lambda x : float(dicom.read_file(x).ImagePositionPatient[2]) )

		# parsing metadata from the scan files
		dcm = dicom.read_file(scanData[0])

		data[caseName]['manufacturer'] = dcm.Manufacturer
		data[caseName]['model'] = dcm.ManufacturerModelName
		
		data[caseName]['rows'] = int(dcm.Rows)
		data[caseName]['cols'] = int(dcm.Columns)
		
		data[caseName]['sliceThickness'] = float(dcm.SliceThickness)
		data[caseName]['pixelSpacing'] = np.array(dcm.PixelSpacing)
		data[caseName]['imagePosition'] = np.array(dcm.ImagePositionPatient)

		data[caseName]['scans'] = { float(dicom.read_file(scan).ImagePositionPatient[2]) : scan 
									for scan in scanData }       

		depthSorted = sorted([depth for depth in data[caseName]['scans']])

		''' data not present n all files '''
		# data[caseName]['weight'] = dcm.PatientWeight
		# data[caseName]['age'] = dcm.PatientAge
		# data[caseName]['sex'] = dcm.PatientSex
		
		if len(subFolders) > 1:
			# parsing annotation data
			data[caseName]['roi'] = []
			dcm = dicom.read_file(annotData)
			
			segIDs = []
			for segObservations in dcm.RTROIObservationsSequence:
				segIDs.append(segObservations.ROIObservationLabel)

			try:
				segID = segIDs.index('GTV-1')		# segment class to use as roi, 'GTV-1' = tumor
			except Exception as err:
				print(caseName, err)

			try:
				for seq in dcm.ROIContourSequence[segID].ContourSequence:

					try:
						roi = {}
						roi['idx'] = int(seq.ContourNumber)
						roi['color'] = np.array(dcm.ROIContourSequence[segID].ROIDisplayColor)
						roi['geom'] = seq.ContourGeometricType                                          # contour geometry
						roi['nodes'] = np.array(seq.ContourData).reshape(-1,3)                          # list of coordinates (x,y,z)
						roi['sidx'] = depthSorted.index( roi['nodes'][0, 2] )                           # slice index,  roi['nodes'][0, 2] = z_value
						roi['sname'] = os.path.basename( data[caseName]['scans'][roi['nodes'][0, 2]] )  # slice file name
				
					except ValueError as err:
						print(caseName, err, "Skipping ROI")
						continue
						
					data[caseName]['roi'].append(roi)
			
			except Exception as err:
				print(caseName, err, "Skipping Case")
				continue
	
	# dumping to disk
	with open( savePath, 'wb') as df:
		pickle.dump(data, df, protocol=2)

	return data

def getMetaData3(pathToData, savePath = './metaData.p'):

	''' USED FOR DATASET 3: 
	This function looks through the different patitent folders
	and extracts the metadata, including the ROI annotations AND
	annotations of other regions such as Lung, Paitent body etc. 
	and saves them in a dictionary indexed by case name 
	
	The dictionary can be dumped to disk at the savePath and loaded
	later to retreive all the information without having to loop over
	all cases again '''

	# getting all case folders
	caseFolders = sorted(glob.glob(pathToData + "/*"))

	# dictionary to record data
	data = {}

	# label mapping
	labelMap = { 'Patient' : 'Patient',
				 'patient' : 'Patient',
				 'lung_l' : 'Lungs',
				 'lung_r' : 'Lungs',
				 'Lung L' : 'Lungs',
				 'Lung R' : 'Lungs',
				 'Lung' : 'Lungs',
				 'GTV-1' : 'roi',
				 'gtv-1' : 'roi',
				 }

	# getting scan data and annotations
	bar = pbar.ProgressBar(max_value=len(caseFolders))      # progressbar to keep track

	for idx,case in enumerate(caseFolders):
		bar.update(idx+1)

		# getting folders inside each case folder
		subFolders = glob.glob(case + '/*/*')
		
		# dcm files in two folders; one for scans, another for annotation
		try:
			dcmFilesA = glob.glob(subFolders[0] + '/*.dcm')
			if len(subFolders) > 1:
				dcmFilesB = glob.glob(subFolders[1] + '/*.dcm')
		except Exception as err:
			print(case, err)
			continue

		if len(subFolders) > 1:
			# scanData is in folder with more than one dcm file
			scanData = dcmFilesA if len(dcmFilesA) > 1 else dcmFilesB
			annotData = dcmFilesB[0] if len(dcmFilesA) > 1 else dcmFilesA[0]
		else:
			scanData = dcmFilesA

		# extracting the case name and studyID from path
		caseName = os.path.basename(case)

		# creating entry in dict
		data[caseName] = {}
		data[caseName]['id'] =  caseName
	   
		# sorting slices according to depth
		scanData.sort( key = lambda x : float(dicom.read_file(x).ImagePositionPatient[2]) )

		# parsing metadata from the scan files
		dcm = dicom.read_file(scanData[0])

		data[caseName]['manufacturer'] = dcm.Manufacturer
		data[caseName]['model'] = dcm.ManufacturerModelName
		
		data[caseName]['rows'] = int(dcm.Rows)
		data[caseName]['cols'] = int(dcm.Columns)
		
		data[caseName]['sliceThickness'] = float(dcm.SliceThickness)
		data[caseName]['pixelSpacing'] = np.array(dcm.PixelSpacing)
		data[caseName]['imagePosition'] = np.array(dcm.ImagePositionPatient)

		data[caseName]['scans'] = { float(dicom.read_file(scan).ImagePositionPatient[2]) : scan 
									for scan in scanData }       

		depthSorted = sorted([depth for depth in data[caseName]['scans']])

		''' data not present n all files '''
		# data[caseName]['weight'] = dcm.PatientWeight
		# data[caseName]['age'] = dcm.PatientAge
		# data[caseName]['sex'] = dcm.PatientSex
		
		if len(subFolders) > 1:
			# parsing annotation data
			dcm = dicom.read_file(annotData)
			
			segIDs = []
			for segObservations in dcm.RTROIObservationsSequence:
				segIDs.append(segObservations.ROIObservationLabel)

			for regionLabel in segIDs:

				if regionLabel not in list(labelMap.keys()):
					continue
				
				if regionLabel == 'GTV-1':
					data[caseName]['roi'] = []
				else:
					if labelMap[regionLabel] not in data[caseName]:
						data[caseName][labelMap[regionLabel]] = []
				try:
					segID = segIDs.index(regionLabel)		# segment class to use as roi, 'GTV-1' = tumor
				except Exception as err:
					print("\n\n", caseName, "\n", regionLabel, " : ", err)

				try:
					for seq in dcm.ROIContourSequence[segID].ContourSequence:

						try:
							roi = {}
							roi['idx'] = int(seq.ContourNumber)
							roi['color'] = np.array(dcm.ROIContourSequence[segID].ROIDisplayColor)
							roi['geom'] = seq.ContourGeometricType                                          # contour geometry
							roi['nodes'] = np.array(seq.ContourData).reshape(-1,3)                          # list of coordinates (x,y,z)
							roi['sidx'] = depthSorted.index( roi['nodes'][0, 2] )                           # slice index,  roi['nodes'][0, 2] = z_value
							roi['sname'] = os.path.basename( data[caseName]['scans'][roi['nodes'][0, 2]] )  # slice file name
					
						except ValueError as err:
							print(caseName, err, "Skipping ROI")
							continue
						
						if regionLabel == 'GTV-1':
							data[caseName]['roi'].append(roi)
						else:
							data[caseName][labelMap[regionLabel]].append(roi)
							
				except Exception as err:
					print(caseName, err, "Skipping Case")
					continue
	
	# dumping to disk
	with open( savePath, 'wb') as df:
		pickle.dump(data, df, protocol=2)

	return data

def getRegionLabels(pathToData):
	''' USED FOR DATASET 3: 
	This function looks through the different patitent folders
	and extracts the names of all labelled regions '''

	regionLabels = []

	# getting all case folders
	caseFolders = sorted(glob.glob(pathToData + "/*"))

	# getting scan data and annotations
	# progressbar to keep track
	bar = pbar.ProgressBar(max_value=len(caseFolders))

	for idx, case in enumerate(caseFolders):
		bar.update(idx+1)

		# getting folders inside each case folder
		subFolders = glob.glob(case + '/*/*')

		# dcm files in two folders; one for scans, another for annotation
		try:
			dcmFilesA = glob.glob(subFolders[0] + '/*.dcm')
			if len(subFolders) > 1:
				dcmFilesB = glob.glob(subFolders[1] + '/*.dcm')
		except Exception as err:
			print(case, err)
			continue

		if len(subFolders) > 1:
			# scanData is in folder with more than one dcm file
			scanData = dcmFilesA if len(dcmFilesA) > 1 else dcmFilesB
			annotData = dcmFilesB[0] if len(dcmFilesA) > 1 else dcmFilesA[0]
		else:
			scanData = dcmFilesA

		if len(subFolders) > 1:
			# parsing annotation data
			dcm = dicom.read_file(annotData)

			segIDs = []
			for segObservations in dcm.RTROIObservationsSequence:
				segIDs.append(segObservations.ROIObservationLabel)

			regionLabels.extend(segIDs)

	return np.unique(regionLabels,return_counts=True)

def loadMetaData(pathToMetaData):
	''' Loads picklekd metadata '''
	 
	with open(pathToMetaData, 'rb') as df:
		try:
			db = pickle.load(df)
		except:
			db = pickle.load(df,encoding='latin1')
	return db

def genMask(caseMetaData, regionLabel='roi'):

	''' This function takes the metadata of a CT scan as input
	and returns the 3D mask for the ROI '''

	x0,y0,z0 = caseMetaData['imagePosition']
	px,py = caseMetaData['pixelSpacing']

	thickness = caseMetaData['sliceThickness']

	# loading slices depthwise (lowest -> highest)
	depthSorted = sorted( [depth for depth in caseMetaData['scans']] )
	slices = [dicom.read_file(caseMetaData['scans'][depth].replace('\\','/')) for depth in depthSorted]

	# stacking slices along depth axis (last axis)
	img = np.stack( [s.pixel_array for s in slices], axis=-1 )

	# initializing mask
	mask = np.zeros_like(img, dtype=np.uint8)

	# slice index for which mask present kept here 
	maskSlices = []
	
	# looping over contours
	for c in caseMetaData[regionLabel]:
		pts = c['nodes']

		# getting the x-y plane index, i.e slice index of the contour
		sidx = c['sidx']
		maskSlices.append(sidx)

		# adjusting contour points to global origin
		x = (pts[:,0] - x0) / px
		y = (pts[:,1] - y0) / py
		
		# generating a closed polygon mask using contour pts
		xx, yy = polygon(x,y)

		# setting points within mask to value of 1
		mask[yy, xx, sidx] = 1

	return mask, sorted(np.unique(maskSlices))

def stackSlices(caseMetaData):
	
	''' This function takes the case meta data and loads
	the dicom files for all the slices, converts to Hounsfield Unit,
	stacks the pixel data, depthwise and returns the result '''

	# loading slices depthwise (lowest -> highest)
	depthSorted = sorted([depth for depth in caseMetaData['scans']])
	slices = [dicom.read_file(caseMetaData['scans'][depth].replace("\\", "/" ))for depth in depthSorted]

	# stacking slices along depth axis (last axis)
	img = np.stack([s.pixel_array for s in slices], axis=-1)
	img = np.int16(img)

	# converting to HU units
	for sidx in range(len(slices)):

		intercept = slices[sidx].RescaleIntercept
		slope = slices[sidx].RescaleSlope

		if slope != 1:
			img[sidx] = slope * img[sidx].astype(np.float64)
			img[sidx] = img[sidx].astype(np.int16)
		
			img[sidx] += np.int16(intercept)

	return img

def changePlane(stack, plane):
	''' This function takes a 3D stack of xy planes and reorients it such that
		the desired 2D plane views (xy, yz or xz) are indexed by the last axis'''

	sliceAxes = {
		'xy': (0, 1),
		'yz': (0, 2),
		'xz': (1, 2),
	}

	return np.rot90(stack, axes=sliceAxes[plane])

def getMaskIndex(mask):
	''' gets the index along the last axis in which tumor
	region exists (non zero) '''

	midx = []
	for i in range(mask.shape[-1]):
		if np.sum(mask[...,i]) > 0:
			midx.append(i)
	
	return sorted(midx)

def plotHist(img):

	# plotting histogram
	plt.figure()
	plt.hist( img.flatten(), bins = 80, color ='c')
	plt.grid()
	plt.show()

def plotScan(img, mask=None, sidx=None, plane = 'xy', threshold = None, colorMap=plt.cm.bone, title=None, colors = ['b','g','r','c','m','y','k','w']):

	''' This function takes a stacked slices (img) and optinally
	stacked mask and slice indices, and displays them sequentially
	in a 2D map color map '''

	# callback for displaying slices and slider control
	class callback(object):

		def __init__(self, fig, ax, img, mask, sidx, threshold, totalFrames, colors, title):
			self.fig = fig
			self.ax = ax
			self.title = title
			self.img = img
			self.img0 = img.copy()
			self.mask = mask
			self.sidx = sidx
			self.threshold = threshold
			self.interval = 30
			self.colors = colors
			self.pause = False
			self.curFrameNum = 0
			self.totalFrames = totalFrames
			
			if isinstance(self.mask,(list,)):
				if len(self.mask) == 2 and plane == 'xy':
					self.dc = []
					self.msd = []
					self.h95 = []
					self.calcMetrics()
					self.props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

		def updateThreshold(self,value):
			self.threshold = value
			self.img = self.img0.copy()
			self.img[ self.img <= value ] = -1000

		def draw(self):

			i = self.curFrameNum
			
			self.ax.clear()
			if self.title is not None:
				self.ax.set_title("%s : Slice %03d" %(self.title,i))
			else:
				self.ax.set_title("Slice %03d" % i)
			

			if plane == 'xy':
				self.ax.imshow(self.img[...,i], cmap=colorMap)
				if self.mask is not None:
					if isinstance(self.mask, (list,)):
						for color_index, m in enumerate(self.mask):
							self.ax.contour(m[..., i], levels=[0], colors=self.colors[color_index % len(self.colors)])
					
						if len(self.mask) == 2:
							# results = "Dice : %1.4f\nMSD : %3.2f\nH95 : %3.2f" % (self.dc[i], self.msd[i], self.h95[i])
							# self.ax.text(0.05, 0.95, results, transform=self.ax.transAxes, fontsize=12,verticalalignment='top', bbox=self.props)
							pass
					else:
						self.ax.contour(self.mask[..., i], levels=[0], colors='r')
			
			elif plane == 'xz':
				self.ax.imshow(self.img[i, ...], cmap=colorMap)
				if self.mask is not None:
					if isinstance(self.mask, (list,)):
						for color_index, m in enumerate(self.mask):
							self.ax.contour(m[i, ...], levels=[0],colors=self.colors[color_index % len(self.colors)])
					else:
						self.ax.contour(self.mask[i, ...], levels=[0], colors='r')

			elif plane == 'yz':
				self.ax.imshow(self.img[:,i,:], cmap=colorMap)
				if self.mask is not None:
					if isinstance(self.mask, (list,)):
						for color_index, m in enumerate(self.mask):
							self.ax.contour(m[:,i,:], levels=[0],colors=self.colors[color_index % len(self.colors)])
					else:
						self.ax.contour(self.mask[:,i,:], levels=[0], colors='r')

		def update(self,frameNum):

			if self.pause:
				return

			if self.sidx is None:
				self.curFrameNum = frameNum
			else:
				self.curFrameNum = self.sidx[frameNum]

			self.draw()

		def togglePause(self, event):
			self.pause = not self.pause
		
		def nextSlice(self, event):
			if not self.pause:
				return
			
			self.curFrameNum += 1	
			self.curFrameNum = 0 if self.curFrameNum >= self.totalFrames else self.curFrameNum
			self.draw()
		
		def prevSlice(self, event):
			if not self.pause:
				return

			self.curFrameNum -= 1
			self.curFrameNum = self.totalFrames-1 if self.curFrameNum < 0 else self.curFrameNum
			self.draw()
				
		def calcMetrics(self):
			self.mask = [np.clip(m,0,1) for m in self.mask]
			
			for i in range(self.mask[0].shape[-1]):
				# self.dc.append(metrics.dice_coef(self.mask[0][...,i],self.mask[1][...,i]))
				# self.msd.append(0.5 * (metrics.mean_surfd(self.mask[0][...,i],self.mask[1][...,i]) + metrics.mean_surfd(self.mask[1][...,i],self.mask[0][...,i])))
				# self.h95.append(0.5 * (metrics.hausdorff_n(self.mask[0][...,i],self.mask[1][...,i],95) + metrics.hausdorff_n(self.mask[1][...,i],self.mask[0][...,i],95)))
				pass	


	fig = plt.figure()
	ax = plt.subplot(1,1,1)

	i = 0 if sidx is None else sidx[0]

	ax.set_title("Slice %03d" % i)

	if plane == 'xy':
		frameAxis = 2
		cs = ax.imshow(img[..., i], cmap=colorMap)
		if mask is not None:
			if isinstance(mask, (list,)):
				for color_index, m in enumerate(mask):
					ax.contour(m[..., i], levels=[0],colors=colors[color_index % len(colors)])
			else:
				ax.contour(mask[..., i], levels=[0], colors='r')
	
	elif plane == 'yz':
		frameAxis = 1
		cs = ax.imshow(img[:,i,:], cmap=colorMap)
		if mask is not None:
			if isinstance(mask, (list,)):
				for color_index, m in enumerate(mask):
					ax.contour(m[:,i,:], levels=[0],colors=colors[color_index % len(colors)])
			else:
				ax.contour(mask[:,i,:], levels=[0], colors='r')
	
	elif plane == 'xz':
		frameAxis = 0
		cs = ax.imshow(img[i, ...], cmap=colorMap)
		if mask is not None:
			if isinstance(mask, (list,)):
				for color_index, m in enumerate(mask):
					ax.contour(m[i, ...], levels=[0],colors=colors[color_index % len(colors)])
			else:
				ax.contour(mask[i, ...], levels=[0], colors='r')

	plt.axis('off')
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	
	# animation controls
	totalFrames = img.shape[frameAxis] if sidx is None else len(sidx)
	cb = callback(fig,ax,img,mask,sidx,threshold, totalFrames, colors, title)

	pauseBut = Button(plt.axes([0.1, 0.375, 0.1, 0.05]), 'Pause', color='grey')
	slicePlus = Button(plt.axes([0.16, 0.30, 0.04, 0.05]), '>>', color='grey')
	sliceMinus = Button(plt.axes([0.1, 0.30, 0.04, 0.05]), '<<', color='grey')
	pauseBut.on_clicked(cb.togglePause)
	slicePlus.on_clicked(cb.nextSlice)
	sliceMinus.on_clicked(cb.prevSlice)

	if threshold is not None:
		img[ img <= threshold ] = -1000
		thresh = Slider(plt.axes([0.35,0.05,0.3,0.05]), 'HU Threshold',
						valmin=-1500, valmax=3000, valinit=threshold,
						color='lightblue')

		thresh.on_changed(cb.updateThreshold)

	cb.anim = animation.FuncAnimation(fig, cb.update, interval=cb.interval, repeat=True,frames=totalFrames)

	plt.show()

def makeMesh(img, threshold = 300, stepSize=1):

	''' This function takes stacked slices and returns the vertices
	and faces of the mesh generated using the marching cube alg '''

	verts, faces, norm, val = measure.marching_cubes(img, threshold, step_size=stepSize, allow_degenerate=True) 
	
	return verts, faces

def plotScan3D(img, mask = None, sidx = None, threshold = 100, title = '3D Plot'):

	''' This fucntion takes stacked slices and optionally masks, index
	of slices and a threshold value, and creates an interactive
	3D plot of the data. 
	
	Threshold values are in Hounsfield units '''

	if mask is not None:
		roi = np.multiply(img,mask)
		bg = img - roi

		if sidx is not None:
			roi = roi[...,sidx]
			bg = bg[...,sidx]
	
		vertsBg, facesBg = makeMesh(bg,threshold=threshold)
		xBg, yBg, zBg = zip(*vertsBg)

		vertsRoi, facesRoi = makeMesh(roi, threshold=threshold)
		xRoi, yRoi, zRoi = zip(*vertsRoi)

		colormap1 = ['rgb(0,135,255)','rgb(0,191,255)']
		
		fig1 = FF.create_trisurf(x=xBg,y=yBg,z=zBg,plot_edges=False,
								colormap=colormap1, 
								simplices=facesBg)

		fig1['data'][0].update(opacity=0.75)

		colormap2 = ['rgb(255,90,0)', 'rgb(255,50,0)']

		fig2 = FF.create_trisurf(x=xRoi, y=yRoi, z=zRoi, plot_edges=False,
								 colormap=colormap2,
								 simplices=facesRoi,
								 backgroundcolor='rgb(64, 64, 64)',
								 title=title)

		fig2['data'][0].update(opacity=0.85)

		fig = [fig1.data[0], fig1.data[1], fig2.data[0],fig2.data[1]]

	else:

		if sidx is not None:
			img = img[...,sidx]

		verts, faces = makeMesh(img,threshold=threshold)
		x, y, z = zip(*verts)

		colormap = ['rgb(0,135,255)','rgb(255,191,0)']
		
		fig = FF.create_trisurf(x=x,y=y,z=z,plot_edges=False,
								colormap=colormap, 
								simplices=faces,
								backgroundcolor='rgb(64, 64, 64)',
								title=title)


	plotly.offline.plot(fig, filename='./' + title + '.html')

def getVoxelDims(caseMetaData):
	
	dz0 = caseMetaData['sliceThickness']
	dx0, dy0 = caseMetaData['pixelSpacing']

	return (dy0, dx0, dz0)

def resampleVoxels(img, oldDim, newDim):

	'''This function takes stacked slices (img) and its meta data
	along with new voxel dimensions (in mm). It uses the zoom function
	with nearest neighbour interploation to produce the output stack
	with desired voxel dimensions'''

	# scale factor along each axis x,y,z
	scaleFactor = np.divide(oldDim,newDim)

	# new image shape in pixels, rounded
	newShape = np.round(img.shape * scaleFactor)

	# scaleFactor for acheiving integer shape (i.e rounded values found above)
	scaleFactor = np.divide(newShape,img.shape)
	newDim = np.divide(oldDim,scaleFactor)

	img = scipy.ndimage.interpolation.zoom(img, scaleFactor, order=1)

	return img, newDim

def segmentLung(img, fillLung = False, threshold=400):

	# helper function that takes in region labels and returns
	# label of largest region
	def getLargestRegionLabel(labels, bg = -1):
		
		# getting seperate regions and their area (i.e count of unique label)
		regions, area = np.unique(labels, return_counts=True)
		
		# ignoring regions with label of background (by default = 0)
		area = area[ regions != bg ]
		regions = regions[ regions != bg ]
		
		# getting largest region if one exists (i.e not all bg)
		if len(regions) > 0:
			return regions[np.argmax(area)]
		else:
			return None
		
	# converting stack to binary and adding 1 to make values 1 and 2 ...
	# lung tissue (i.e greather than threshold) = 2 (i.e white), air pockets = 1 (i.e black)
	imgBin = np.array( img > threshold, dtype = np.int8) + 1

	# ... because measure.label automatically considers 0 values as bg
	# measure.label assigns a number to each connected region
	labels = measure.label(imgBin)

	# getting label for air out side lung from corners of image
	bgLabel = labels[0,0,0]

	# makes air outside lung white, so that this region blends with
	# lung lining => only inside of lung left black/white
	imgBin[labels == bgLabel] = 2

	# fills inside of lung using connected components
	if fillLung:
		
		for sidx, s in enumerate(imgBin):

			s = s-1
			labels2 = measure.label(s)
			maxRegionLabel = getLargestRegionLabel(labels2, bg = 0)

			if maxRegionLabel is not None:
				imgBin[sidx][labels2 != maxRegionLabel] = 1

	# returining binary image i.e between 0 and 1
	imgBin -= 1

	# inverting image => lung tissue including tumors = black
	imgBin = 1-imgBin
	
	# fill left over air pockets - fails when 
	# there is tumor growing into lung from lung lining
	# labels = measure.label(imgBin,background=0)
	# maxRegionLabel = getLargestRegionLabel(labels, bg = 0)
	# imgBin[labels != maxRegionLabel] = 0

	return imgBin

def elasticTransform_legacy(image, mask, sigma, alpha_affine, random_seed=None):
	"""Elastic deformation of images as described in [Simard2003]_ (with modifications).
	.. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
		 Convolutional Neural Networks applied to Visual Document Analysis", in
		 Proc. of the International Conference on Document Analysis and
		 Recognition, 2003.

	 Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
	"""
	
	random_state = np.random.RandomState(random_seed)
	
	if len(image.shape)<3:
		image = np.expand_dims(image,-1)
	if len(mask.shape)<3:
		mask = np.expand_dims(mask,-1)
	
	shape = image.shape
	shape_size = shape[:2]

	# Random affine
	center_square = np.float32(shape_size) // 2
	square_size = min(shape_size) // 3
	pts1 = np.float32([center_square + square_size, [center_square[0] + square_size,center_square[1] - square_size], center_square - square_size])
	pts2 = pts1 + random_state.uniform(-alpha_affine,alpha_affine, size=pts1.shape).astype(np.float32)
	
	M = cv2.getAffineTransform(pts1, pts2)
	
	image_w = np.zeros_like(image)
	for i in range(image.shape[-1]):
		image_w[...,i] = cv2.warpAffine(image[...,i], M, shape_size[::-1], borderMode=cv2.BORDER_CONSTANT, borderValue=int(np.min(image[...,i])))
	
	mask_w = np.zeros_like(mask)
	for i in range(mask.shape[-1]):
		mask_w[...,i] = cv2.warpAffine(mask[...,i] , M, shape_size[::-1], borderMode=cv2.BORDER_REPLICATE)

	blur_size = int(2*sigma) | 1
	dx = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), ksize=(blur_size, blur_size), sigmaX=sigma)
	dy = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), ksize=(blur_size, blur_size), sigmaX=sigma)
	
	if len(dx.shape) < 3:
		dx = np.expand_dims(dx,-1)
		dy = np.expand_dims(dy,-1)

	gx, gy = np.meshgrid(np.arange(shape_size[1]), np.arange(shape_size[0]))
	gx = np.expand_dims(gx,-1)
	gy = np.expand_dims(gy,-1)

	gx = np.repeat(gx,dx.shape[-1], -1)
	gy = np.repeat(gy,dy.shape[-1], -1)

	gx = (gx + dx).astype(np.float32)
	gy = (gy + dy).astype(np.float32)

	image_d = np.zeros_like(image_w)
	mask_d = np.zeros_like(mask_w)

	for i in range(image.shape[-1]):
		image_d[...,i] = cv2.remap(image_w[...,i], gx[...,i], gy[...,i], interpolation=cv2.INTER_LINEAR)
	
	radix = gx.shape[-1]
	for i in range(mask.shape[-1]):
		mask_d[...,i] = cv2.remap(mask_w[...,i], gx[...,i%radix], gy[...,i%radix], interpolation=cv2.INTER_LINEAR)
	
	return image_d, mask_d

def getTotalEntryCount(dbMetaData, onlyTumor, augments, nonTumorSlices, plane = 'xy'):
	''' This function calculates the total number of samples that will
	be generated after augmentation and other preprocesses. It is used
	to pre-allocate space for the hdf5 database '''

	totalSamples = 0
	baseSamples = 0

	for idx, name in enumerate(dbMetaData):

		case = dbMetaData[name]

		if name in corrections:
			if corrections[name] is None:
				continue

		if onlyTumor:
			if plane == 'xy':
				nSlices = len(np.unique([cntr['sidx'] for cntr in case['roi']]))
			elif plane == 'yz' or plane == 'xz':
				mask, _ = genMask(case)
				mask = changePlane(mask, plane)
				nSlices = len(getMaskIndex(mask))
		else:
			if plane == 'xy':
				if nonTumorSlices < 0:
					nSlices = len([depth for depth in case['scans']])
				else:
					nSlices = nonTumorSlices + len(np.unique([cntr['sidx'] for cntr in case['roi']]))
			elif plane == 'yz' or plane == 'xz':
				nSlices = 100

		nSlicesOriginal = nSlices
		
		if 'rot90' in augments or 'rot180' in augments or 'rot270' in augments:
			nAngles = len( [ r for r in augments if r in ['rot90','rot180', 'rot270'] ] )
			nSlices = nSlices + nSlicesOriginal * nAngles
		if 'horFlip' in augments:
			nSlices = nSlices + nSlicesOriginal
		if 'verFlip' in augments:
			nSlices = nSlices + nSlicesOriginal
		if 'elasticTransform' in augments:
			nSlices = (nSlices * 2) if 'compound' in augments else (nSlices + nSlicesOriginal)

		totalSamples += nSlices
		baseSamples += nSlicesOriginal

	return totalSamples,baseSamples

def show(x, timeout_ms):

	def close_event():
		plt.close()

	fig = plt.figure()
	timer = fig.canvas.new_timer(interval=timeout_ms)
	timer.add_callback(close_event)

	plt.imshow(x)

	timer.start()
	plt.show()

def createDatabase2D(dbMetaData, dbPath, maskRegions = ['roi'], wavelet=False, onlyTumor=False, nonTumorSlices=-1, augments=[], crop=None, resizeTo = None, rescale = False, clipHU=None, expandMasks=False, minArea=100, applyGaussianFilter=False):

	''' This function creates an HDF5 database file with the 2D tumor slices
		under 'slice' and their coressponding masks under 'mask'. If multiple
		regions in mask, each region mask is a seperate channel. The name of 
		the case file from which the slice came from is under 'case' '''

	X = None         # slices
	Y = None         # masks
	hasTumor = []	 # whether slice has tumor(1) or not(0) 
	caseNames = []   # name of case where slice came from
	
	if crop is None:
		cx1, cx2 = (0, 512 if resizeTo is None else resizeTo[0])
		cy1, cy2 = (0, 512 if resizeTo is None else resizeTo[1])
	else:
		cx1, cy1, cx2, cy2 = crop
	
	if resizeTo is None:
		w = cx2 - cx1
		h = cy2 - cy1
	else:
		w = resizeTo[0]
		h = resizeTo[1]

	nTotalEntries,nBaseEntries = getTotalEntryCount(dbMetaData, onlyTumor, augments, nonTumorSlices)

	try:
		print("\nCreating database with %d entries" % nTotalEntries)
		print("Each slice has shape %d x %d" % (h, w))
		ensureDir(dbPath)
		db = h5py.File(dbPath, mode='w')
		db.create_dataset("slice", (nTotalEntries, h, w,), np.float32)

		if len(maskRegions) == 1:		
			db.create_dataset("mask",  (nTotalEntries, h, w,), np.float32 if applyGaussianFilter else np.uint8)
		elif len(maskRegions) > 1:
			db.create_dataset("mask",  (nTotalEntries, h, w, len(maskRegions)), np.float32 if applyGaussianFilter else np.uint8)

		db.create_dataset("case",  (nTotalEntries,), np.dtype('|S16'))
		db.create_dataset("tumor", (nTotalEntries,), np.uint8)			# binary label
		# db.create_dataset("area", (nTotalEntries,1), np.float32)
		# db.create_dataset("bbox", (nTotalEntries,4), np.float32)		# (x1,y1,x2,y2)
		# db.create_dataset("centroid", (nTotalEntries,2), np.float32)
		# db.create_dataset("solidity", (nTotalEntries,1), np.float32) # Ratio of pixels in the region to pixels of the convex hull image.
		if wavelet:
			db.create_dataset("wavelet", (nTotalEntries, h, w,), np.float32)

	except Exception as err:
		print("Error creating database : ", err)
		db.close()
		return

	didx = 0				# database index
	maxBufferSize = 200		# flushing to disk every 200 entries
	bar = pbar.ProgressBar(max_value = len(dbMetaData))

	print("\nGenerating %d Entries from %d (%dx)" % (nTotalEntries,nBaseEntries, (nTotalEntries//nBaseEntries)))
	
	if len(augments) > 0:
		if 'rot90' in augments:
			print("Applying 90 degree rotation")
		if 'rot180' in augments:
			print("Applying 180 degree rotation")
		if 'rot270' in augments:
			print("Applying 270 degree rotation")
		if 'horFlip' in augments:
			print("Applying Horizontal Flip")
		if 'verFlip' in augments:
			print("Applying Vertical Flip")
		if 'elasticTransform' in augments:
			print("Applying Random Elastic Transformations")
		if 'compound' in augments:
			print("Compounding all augmentations")

	for idx, name in enumerate(dbMetaData):

		bar.update(idx + 1)
		case = dbMetaData[name]

		if name in corrections:
			if corrections[name] is not None:
				yCorr, xCorr = corrections[name]
			else:
				continue
		else:
			yCorr, xCorr = 0 ,0

		# getting slices
		slices = stackSlices(case)
		# adjusting HU scale for other manufacturer (CMS)
		if np.min(slices) < 0:
			slices = slices + 1024
		# creating binary label for each slice
		sliceLabels = np.uint8([0] * slices.shape[-1])
		# current dimension of voxels (mm)
		dz0 = case['sliceThickness']
		dx0, dy0 = case['pixelSpacing']
		# resampling voxels to be 0.97 x 0.97 x 3.00 mm3
		slices, _ = resampleVoxels(slices, oldDim=(dy0,dx0,dz0), newDim=(0.97,0.97,3))
		
		try:
			# getting masks
			fullMask = None
			roi_idx = None
			for region in maskRegions:
				masks, midx = genMask(case, regionLabel=region)

				# applying corrections
				if yCorr or xCorr:
					masks = np.roll(masks, yCorr, 0)
					masks = np.roll(masks, xCorr, 1)

				masks, _ = resampleVoxels(masks, oldDim=(dy0, dx0, dz0), newDim=(0.97,0.97,3))
				# setting binary label
				if region == 'roi':
					sliceLabels[midx] = 1
					roi_idx = midx.copy()

				# concatanating seperate regions as channels in image
				if len(maskRegions) > 1:
					if fullMask is None:
						fullMask = np.expand_dims(masks,-1) 
					else:
						fullMask = np.concatenate( (fullMask, np.expand_dims(masks,-1)), axis=-1)
			# brining channel axis before depth axis
			if len(maskRegions)>1:
				masks = np.moveaxis(fullMask,-1,-2)
			
			midx = roi_idx
		except Exception as err:
			print("\nError while getting masks: %s - %s" %(name,err)) 
			continue

		# clipping HU values
		if clipHU is not None:
			slices = np.clip(slices, clipHU[0], clipHU[1])
			
			# rescale values to between 0.0 and 1.0
			if rescale:
				slices += (0-clipHU[0])
				slices = np.float64(slices)
				slices = np.divide(slices, ((clipHU[1]-clipHU[0]) * 1.0))
		
		# cropping
		x = slices[cy1:cy2, cx1:cx2, :]
		y = masks[cy1:cy2, cx1:cx2, :]
		
		# TODO: use better interpolator so that values dont change too much
		# resizing
		if resizeTo is not None:
			xResized = None
			yResized = None

			for sidx in range(x.shape[-1]):
				xr = np.expand_dims(cv2.resize( x[...,sidx], resizeTo,interpolation=cv2.INTER_NEAREST),-1)
				yr = np.expand_dims(cv2.resize( y[...,sidx], resizeTo,interpolation=cv2.INTER_NEAREST),-1)

				if xResized is None:
					xResized = xr
					yResized = yr
				else:
					xResized = np.concatenate((xResized,xr),-1)
					yResized = np.concatenate((yResized, yr), -1)

			x = xResized
			y = yResized

		# taking only tumor slices
		if onlyTumor:
			x = x[..., midx]
			y = y[..., midx]
			sliceLabels = sliceLabels[midx]
		else:
			non_tumor_idx = [ii for ii in np.arange(slices.shape[-1]) if ii not in midx]
			non_tumor_idx = np.random.choice(non_tumor_idx, size=nonTumorSlices, replace=True if len(non_tumor_idx) < nonTumorSlices else False).tolist()
			midx.extend(non_tumor_idx)
			midx = sorted(midx)
			x = x[..., midx]
			y = y[..., midx]
			sliceLabels = sliceLabels[midx]

		sliceLabels = sliceLabels.tolist()

		# aplying dialation on masks
		if expandMasks:

			for i in range(masks.shape[-1]):
				m = y[..., i]
				s = x[..., i]

				if np.sum(m) == 0:
					continue

				props = measure.regionprops(m, s)[0]

				if props['area'] > minArea:
					continue

				passes = 0
				while props['area'] < minArea and passes < 10:
					passes += 1
					l = np.ceil(np.sqrt(props['area']))
					l = np.int16(l)
					l = 3 if l < 3 else l
					l = l if l % 2 else l + 1
					kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (l, l))
					m = cv2.dilate(m, kernel, iterations=1)
					props = measure.regionprops(m, s)[0]

				# plt.subplot(1, 2, 1)
				# plt.imshow(y[..., i].reshape(cy2 - cy1, cx2 - cx1))
				# plt.title("Original Mask")

				# plt.subplot(1, 2, 2)
				# plt.imshow(m.reshape(cy2 - cy1, cx2 - cx1))
				# plt.title("Expanded Mask after %d passes" % passes)
				# plt.show()

				# time.sleep(1)
				# plt.close()

				y[..., i] = m

		# smoothing out masks
		if applyGaussianFilter:
			for i in range(masks.shape[-1]):
				m = y[..., i]
				
				if np.sum(m) == 0:
					continue

				m = np.float32(m)
				m = filters.gaussian_filter(m, 1.5)

				# plt.subplot(1, 2, 1)
				# plt.imshow(y[..., i].reshape(cy2 - cy1, cx2 - cx1))
				# plt.title("Original Mask")

				# plt.subplot(1, 2, 2)
				# plt.imshow(m.reshape(cy2 - cy1, cx2 - cx1))
				# plt.title("Filtered Mask")
				# plt.show()

				y[..., i] = m
		
		# applying augmentations
		try:
			if len(augments) > 0:
				
				augSets = []
				
				if 'rot90' in augments or 'rot180' in augments or 'rot270' in augments:
					xRot = None
					yRot = None

					rotMatrices = []
					rows = (cy2-cy1) if resizeTo is None else resizeTo[1]
					cols = (cx2-cx1) if resizeTo is None else resizeTo[0]

					if 'rot90' in augments:
						rotMatrices.append(cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1))
					if 'rot180' in augments:
						rotMatrices.append(cv2.getRotationMatrix2D((cols / 2, rows / 2), 180, 1))
					if 'rot270' in augments:
						rotMatrices.append(cv2.getRotationMatrix2D((cols / 2, rows / 2), 270, 1))
					
					for rotMat in rotMatrices:
						for idx in range(x.shape[-1]):
							s = x[...,idx]
							m = y[...,idx]
							sr = cv2.warpAffine(s, rotMat, (cols, rows))
							sr = np.expand_dims(sr, -1)
							mr = cv2.warpAffine(m, rotMat, (cols, rows))
							mr = np.expand_dims(mr, -1)

							if xRot is None:
								xRot = sr
								yRot = mr
							else:
								xRot = np.concatenate((xRot,sr),-1)
								yRot = np.concatenate((yRot,mr),-1)

					augSets.append( [xRot,yRot])

				if 'horFlip' in augments:
					xHorFlipped = None
					yHorFlipped = None

					for idx in range(x.shape[-1]):
						s = x[..., idx]
						m = y[..., idx]
						shf = cv2.flip(s, 1)
						shf = np.expand_dims(shf,-1)
						mhf = cv2.flip(m, 1)
						mhf = np.expand_dims(mhf,-1)

						if xHorFlipped is None:
							xHorFlipped = shf
							yHorFlipped = mhf
						else:
							xHorFlipped = np.concatenate((xHorFlipped,shf),-1)
							yHorFlipped = np.concatenate((yHorFlipped,mhf),-1)

					augSets.append([xHorFlipped, yHorFlipped])

				if 'verFlip' in augments:
					xVerFlipped = None
					yVerFlipped = None

					for idx in range(x.shape[-1]):
						s = x[...,idx]
						m = y[...,idx]

						svf = cv2.flip(s, 0)
						svf = np.expand_dims(svf, -1)
						mvf = cv2.flip(m, 0)
						mvf = np.expand_dims(mvf, -1)

						if xVerFlipped is None:
							xVerFlipped = svf
							yVerFlipped = mvf
						else:
							xVerFlipped = np.concatenate((xVerFlipped, svf),-1)
							yVerFlipped = np.concatenate((yVerFlipped, mvf),-1)

					augSets.append([xVerFlipped, yVerFlipped])

				if 'elasticTransform' in augments:
					xElastic = None
					yElastic = None
					elasticTransformer = ElasticTransform(sigma=30, alpha_affine=40)

					if 'compound' in augments:
						sliceLabelsOriginal = sliceLabels[:]
						for n in range(len(augments) - 2):
							sliceLabels.extend(sliceLabelsOriginal)	# adding labels for previous augmentatioons

						for (xaug,yaug) in augSets:
							x = np.concatenate((x, xaug),-1)
							y = np.concatenate((y, yaug),-1)

					for idx in range(x.shape[-1]):
						s = x[..., idx]
						m = y[..., idx]
						
						s = np.expand_dims(s,-1)
						comp = np.concatenate((s,m),axis=-1)

						ecomp = elasticTransformer.apply(comp)
						xe,ye = ecomp[...,0],ecomp[...,1:]
						
						if len(xe.shape) < 3 :
							xe = np.expand_dims(xe,-1)
						if len(ye.shape) < 3 :
							ye = np.expand_dims(ye,-1)

						if len(maskRegions) > 1:
							ye = np.expand_dims(ye,-1)

						if xElastic is None:
							xElastic = xe
							yElastic = ye
						else:
							xElastic = np.concatenate((xElastic, xe),-1)
							yElastic = np.concatenate((yElastic, ye),-1)

					if 'compound' in augments:
						x = np.concatenate((x,xElastic),-1)
						y = np.concatenate((y,yElastic),-1)
						sliceLabelsOriginal = sliceLabels[:]
						sliceLabels.extend(sliceLabelsOriginal)
					else:
						augSets.append([xElastic,yElastic])

				if 'compound' not in augments:			
					sliceLabelsOriginal = sliceLabels[:]
					for n in range(len(augments)):
						sliceLabels.extend(sliceLabelsOriginal)

					for (xaug,yaug) in augSets:
						x = np.concatenate((x, xaug),-1)
						y = np.concatenate((y, yaug),-1)
		
		except Exception as err:
			print("\nError while applying augmentations: %s - %s" %(name,err))
			continue

		# appending to buffer
		try:
			if X is None:
				X = x[...]
				Y = y[...]
			else:
				X = np.concatenate((X, x[...]),-1)
				Y = np.concatenate((Y, y[...]),-1)
		except Exception as err:
			print("\nError while appending to buffer: %s - %s" %(name,err))
			continue

		caseNames.extend([name] * x.shape[-1])
		hasTumor.extend(sliceLabels)

		# flushing to disk every now and then
		if X.shape[-1] > maxBufferSize:
			flushSize = X.shape[-1]
			# print("Flushing %d entries" % flushSize)
			X = np.moveaxis(X,-1,0)
			Y = np.moveaxis(Y,-1,0)
			caseNames = np.array(caseNames,dtype='|S16')
			hasTumor = np.uint8(hasTumor)		

			# area,bbox,centroid,solidity = calculateRegionProps(X,Y)

			db["slice"][didx:didx + flushSize, ...] = np.float32(X[...])
			db["mask"][didx:didx + flushSize, ...] = Y[...]
			db["case"][didx:didx + flushSize, ...] = caseNames[...]
			db["tumor"][didx:didx + flushSize, ...] = hasTumor[...]
			# db["area"][didx:didx + flushSize, ...] = area[...]
			# db["bbox"][didx:didx + flushSize, ...] = bbox[...]
			# db["centroid"][didx:didx + flushSize, ...] = centroid[...]
			# db["solidity"][didx:didx + flushSize, ...] = solidity[...]
			
			if wavelet:
				wav = calculateWavelet(X)
				db["wavelet"][didx:didx + flushSize, ...] = wav[...]

			X = None
			Y = None
			hasTumor = []
			caseNames = []

			didx += flushSize

	# Flushing remainders
	if X is not None:
		if X.shape[-1] > 0:
			flushSize = X.shape[-1]
			# print("Flushing %d entries" % flushSize)
			X = np.moveaxis(X, -1, 0)
			Y = np.moveaxis(Y, -1, 0)
			caseNames = np.array(caseNames,dtype='|S16')
			hasTumor = np.uint8(hasTumor)

			# area,bbox,centroid,solidity = calculateRegionProps(X,Y)

			db["slice"][didx:didx + flushSize, ...] = np.float32(X[...])
			db["mask"][didx:didx + flushSize, ...] = Y[...]
			db["case"][didx:didx + flushSize, ...] = caseNames[...]
			db["tumor"][didx:didx + flushSize, ...] = hasTumor[...]
			# db["area"][didx:didx + flushSize, ...] = area[...]
			# db["bbox"][didx:didx + flushSize, ...] = bbox[...]
			# db["centroid"][didx:didx + flushSize, ...] = centroid[...]
			# db["solidity"][didx:didx + flushSize, ...] = solidity[...]

			if wavelet:
				wav = calculateWavelet(X)
				db["wavelet"][didx:didx + flushSize, ...] = wav[...]

	db.close()

def createDatabase3D(dbMetaData, dbPath, nSlices, sliceStep=1, maskType='2D', onlyTumor=False, nonTumorSlices=-1, augments=[], clipHU=None, crop=None, resizeTo=None,rescale=False, expandMasks=False, minArea=100, applyGaussianFilter=False, slicePlanes=['xy']):
	''' This function creates an HDF5 database file containing *n-slice stacks* of 
		2D slices under 'slice' and their coressponding masks under 'mask'. 
		The name of the case file from which the slice came from is under 'case'
		
		The number of neighbouring slices to take is specified by nSlices. If 
		nSlices = 2, the top 2 and bottom 2 of the desired slice is saved along 
		with it. '''

	X = None         # slices
	Y = None         # masks
	hasTumor = []	 # whether slice has tumor(1) or not(0)
	caseNames = []   # name of case where slice came from

	# TODO update 512 to whaterver the dim is after resampling voxels
	if crop is None:
		cx1, cx2 = (0, 512 if resizeTo is None else resizeTo[0])
		cy1, cy2 = (0, 512 if resizeTo is None else resizeTo[1])
	else:
		cx1, cy1, cx2, cy2 = crop

	h = (cy2 - cy1) if resizeTo is None else resizeTo[1]
	w = (cx2 - cx1) if resizeTo is None else resizeTo[0]

	nTotalEntries,nBaseEntries = getTotalEntryCount(dbMetaData, onlyTumor, augments, nonTumorSlices)

	try:
		print("\nCreating database with %d entries" % nTotalEntries)
		print("Each stack has shape %d x %d x %d" % (h, w, (2*nSlices+1)))
		ensureDir(dbPath)
		db = h5py.File(dbPath, mode='w')
		db.create_dataset("slice", (nTotalEntries, h, w,(nSlices * 2) + 1), np.float32)

		if maskType == '2D':
			db.create_dataset("mask",  (nTotalEntries, h, w, 1), np.float32 if applyGaussianFilter else np.uint8)
		else:
			db.create_dataset("mask",  (nTotalEntries, h, w,(nSlices * 2) + 1), np.float32 if applyGaussianFilter else np.uint8)
		
		db.create_dataset("case",  (nTotalEntries,), np.dtype('|S16'))
		db.create_dataset("tumor", (nTotalEntries,), np.uint8)
	except Exception as err:
		print("Error creating database : ", err)
		db.close()
		return

	didx = 0				# database index
	maxBufferSize = 200		# flushing to disk every 200 entries
	bar = pbar.ProgressBar(max_value=len(dbMetaData))

	print("\nGenerating %d Entries from %d (%dx)" %
		  (nTotalEntries, nBaseEntries, (nTotalEntries // nBaseEntries)))

	if len(augments) > 0:
		if 'rot90' in augments:
			print("Applying 90 degree rotation")
		if 'rot180' in augments:
			print("Applying 180 degree rotation")
		if 'rot270' in augments:
			print("Applying 270 degree rotation")
		if 'horFlip' in augments:
			print("Applying Horizontal Flip")
		if 'verFlip' in augments:
			print("Applying Vertical Flip")
		if 'elasticTransform' in augments:
			print("Applying Random Elastic Transformations")
		if 'compound' in augments:
			print("Compounding all augmentations")

	for idx, name in enumerate(dbMetaData):

		bar.update(idx + 1)

		case = dbMetaData[name]

		yCorr, xCorr = 0, 0
		if name in corrections:
			if corrections[name] is not None:
				yCorr, xCorr = corrections[name]
			else:
				continue

		slices = stackSlices(case)
		masks, midx = genMask(case)
		sliceLabels = np.uint8([0] * slices.shape[-1])
		sliceLabels[midx] = 1

		# adjusting HU scale for other manufacturer (SIEMENS)
		if np.min(slices) < 0:
			slices = slices + 1024

		# applying corrections
		if yCorr or xCorr:
			masks = np.roll(masks, yCorr, 0)
			masks = np.roll(masks, xCorr, 1)

		# resampling voxels to be 0.97 x 0.97 x 3.00 mm3 
		# current dimension of voxels (mm)
		dz0 = case['sliceThickness']
		dx0, dy0 = case['pixelSpacing']
		slices, _ = resampleVoxels(slices, oldDim=(dy0, dx0, dz0), newDim=(0.97,0.97,3))
		masks, _ = resampleVoxels(masks, oldDim=(dy0, dx0, dz0), newDim=(0.97,0.97,3))

		# clipping HU values
		if clipHU is not None:
			slices = np.clip(slices, clipHU[0], clipHU[1])

			# rescale values to between 0.0 and 1.0
			if rescale:
				slices += (0-clipHU[0])
				slices = np.float64(slices)
				slices = np.divide(slices , ((clipHU[1]-clipHU[0]) * 1.0))

		# cropping
		x = slices[cy1:cy2, cx1:cx2, :]
		y = masks[cy1:cy2, cx1:cx2, :]

		# TODO: use better interpolator so that values dont change too much
		# resizing
		if resizeTo is not None:
			xResized = None
			yResized = None

			for sidx in range(x.shape[-1]):
				xr = np.expand_dims(cv2.resize( x[...,sidx], resizeTo, interpolation=cv2.INTER_NEAREST),-1)
				yr = np.expand_dims(cv2.resize( y[...,sidx], resizeTo, interpolation=cv2.INTER_NEAREST),-1)

				if xResized is None:
					xResized = xr
					yResized = yr
				else:
					xResized = np.concatenate((xResized,xr),-1)
					yResized = np.concatenate((yResized, yr), -1)

			x = xResized
			y = yResized

		# aplying dialation on masks
		if expandMasks:

			for i in range(masks.shape[-1]):
				m = y[..., i]
				s = x[..., i]

				if np.sum(m) == 0:
					continue

				props = measure.regionprops(m, s)[0]

				if props['area'] > minArea:
					continue

				passes = 0
				while props['area'] < minArea and passes < 10:
					passes += 1
					l = np.ceil(np.sqrt(props['area']))
					l = np.int16(l)
					l = 3 if l < 3 else l
					l = l if l % 2 else l + 1
					kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (l, l))
					m = cv2.dilate(m, kernel, iterations=1)
					props = measure.regionprops(m, s)[0]

				# plt.subplot(1, 2, 1)
				# plt.imshow(y[..., i].reshape(cy2 - cy1, cx2 - cx1))
				# plt.title("Original Mask")

				# plt.subplot(1, 2, 2)
				# plt.imshow(m.reshape(cy2 - cy1, cx2 - cx1))
				# plt.title("Expanded Mask after %d passes" % passes)
				# plt.show()

				# time.sleep(1)
				# plt.close()

				y[..., i] = m

		# smoothing out masks
		if applyGaussianFilter:
			for i in range(masks.shape[-1]):
				m = y[..., i]

				if np.sum(m) == 0:
					continue

				m = np.float32(m)
				m = filters.gaussian_filter(m, 1.5)

				# plt.subplot(1, 2, 1)
				# plt.imshow(y[..., i].reshape(cy2 - cy1, cx2 - cx1))
				# plt.title("Original Mask")

				# plt.subplot(1, 2, 2)
				# plt.imshow(m.reshape(cy2 - cy1, cx2 - cx1))
				# plt.title("Filtered Mask")
				# plt.show()

				y[..., i] = m

		# creating n-slice stacks
		x_stacks = None
		y_stacks = None

		for sidx in range(0,  x.shape[-1]):

			# getting index of top neighbours
			if sidx-(nSlices*sliceStep) >= 0:
				top_idx = list(range(sidx-(nSlices*sliceStep), sidx, sliceStep))
			else:
				top_idx = sorted(list(range(sidx-sliceStep, 0, -sliceStep)))

			# pad by repeating first slice
			if len(top_idx) < nSlices:
				nPad = nSlices - len(top_idx)
				padSlices = [0] * nPad
				padSlices.extend(top_idx)
				top_idx = padSlices

			# getting index of bottom neighbours
			if sidx+(nSlices*sliceStep) < x.shape[-1]:
				bot_idx = list(range(sidx+sliceStep, sidx + (nSlices*sliceStep)+1, sliceStep))
			else:
				bot_idx = list(range(sidx+sliceStep, x.shape[-1], sliceStep))

			# pad by repeating last slice
			if len(bot_idx) < nSlices:
				nPad = nSlices - len(bot_idx)
				bot_idx.extend([x.shape[-1]-1] * nPad)

			# getting slices
			cen_slice = np.expand_dims(x[..., sidx], -1)
			top_slices = x[..., top_idx]
			bot_slices = x[..., bot_idx]
			x0 = np.concatenate((top_slices, cen_slice, bot_slices), -1)

			cen_mask = np.expand_dims(y[..., sidx], -1)

			if maskType != '2D':
				top_masks = y[..., top_idx]
				bot_masks = y[..., bot_idx]
				y0 = np.concatenate((top_masks, cen_mask, bot_masks), -1)
			else:
				y0 = cen_mask

			if x_stacks is None:
				x_stacks = np.expand_dims(x0, -1)
				y_stacks = np.expand_dims(y0, -1)
			else:
				x_stacks = np.concatenate((x_stacks, np.expand_dims(x0, -1)), -1)
				y_stacks = np.concatenate((y_stacks, np.expand_dims(y0, -1)), -1)
		
		x = x_stacks
		y = y_stacks

		# taking only tumor slices
		if onlyTumor:
			x = x[..., midx]
			y = y[..., midx]
			sliceLabels = sliceLabels[midx]

		sliceLabels = sliceLabels.tolist()

		# applying augmentations
		if len(augments) > 0:

			augSets = []

			if 'rot90' in augments or 'rot180' in augments or 'rot270' in augments:
				xRot = None
				yRot = None

				rotMatrices = []
				rows = (cy2 - cy1) if resizeTo is None else resizeTo[1]
				cols = (cx2 - cx1) if resizeTo is None else resizeTo[0]
				if 'rot90' in augments:
					rotMatrices.append(cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1))
				if 'rot180' in augments:
					rotMatrices.append(cv2.getRotationMatrix2D((cols / 2, rows / 2), 180, 1))
				if 'rot270' in augments:
					rotMatrices.append(cv2.getRotationMatrix2D((cols / 2, rows / 2), 270, 1))

				for rotMat in rotMatrices:
					for idx in range(x.shape[-1]):
						s = x[..., idx]
						m = y[..., idx]
						xr =  np.expand_dims(cv2.warpAffine(s, rotMat, (cols, rows)),-1)
						yr = np.expand_dims(cv2.warpAffine(m, rotMat, (cols, rows)), -1)

						if xRot is None:
							xRot = xr
							yRot = yr
						else:
							xRot = np.concatenate((xRot, xr), -1)
							yRot = np.concatenate((yRot, yr), -1)

				augSets.append([xRot, yRot])

			if 'horFlip' in augments:
				xHorFlipped = None
				yHorFlipped = None

				for idx in range(x.shape[-1]):
					s = x[..., idx]
					m = y[..., idx]

					xf = np.expand_dims(cv2.flip(s, 1), -1)
					yf = np.expand_dims(cv2.flip(m, 1), -1)

					if xHorFlipped is None:
						xHorFlipped = xf
						yHorFlipped = yf
					else:
						xHorFlipped = np.concatenate((xHorFlipped, xf), -1)
						yHorFlipped = np.concatenate((yHorFlipped, yf), -1)

				augSets.append([xHorFlipped, yHorFlipped])

			if 'verFlip' in augments:
				xVerFlipped = None
				yVerFlipped = None

				for idx in range(x.shape[-1]):
					s = x[..., idx]
					m = y[..., idx]
					
					xf = np.expand_dims(cv2.flip(s, 0), -1)
					yf = np.expand_dims(cv2.flip(m, 0), -1)

					if xVerFlipped is None:
						xVerFlipped = xf
						yVerFlipped = yf
					else:
						xVerFlipped = np.concatenate((xVerFlipped, xf), -1)
						yVerFlipped = np.concatenate((yVerFlipped, yf), -1)

				augSets.append([xVerFlipped, yVerFlipped])

			if 'elasticTransform' in augments:
				xElastic = None
				yElastic = None
				elasticTransformer = ElasticTransform(sigma=30, alpha_affine=40)

				if 'compound' in augments:
					sliceLabelsOriginal = sliceLabels[:]
					for _ in range(len(augments) - 2):
						# adding labels for previous augmentatioons
						sliceLabels.extend(sliceLabelsOriginal)

					for (xaug, yaug) in augSets:
						x = np.concatenate((x, xaug), -1)
						y = np.concatenate((y, yaug), -1)

				for idx in range(x.shape[-1]):
					s = x[..., idx]
					m = y[..., idx]

					if maskType == '2D':
						m = np.repeat(m, s.shape[-1], axis=-1)

					comp = np.concatenate((s,m),axis=-1)
					ecomp = elasticTransformer.apply(comp)
					xe,ye = ecomp[...,0:s.shape[-1]], ecomp[...,s.shape[-1]:]

					if maskType == '2D':
						mid_idx = (s.shape[-1] - 1)//2
						ye = ye[..., mid_idx]
						ye = np.expand_dims(ye,-1)

					xe = np.expand_dims(xe, -1)
					ye = np.expand_dims(ye,-1)
					
					if xElastic is None:
						xElastic = xe
						yElastic = ye
					else:
						xElastic = np.concatenate((xElastic, xe), -1)
						yElastic = np.concatenate((yElastic, ye), -1)

				if 'compound' in augments:
					x = np.concatenate((x, xElastic), -1)
					y = np.concatenate((y, yElastic), -1)
					sliceLabelsOriginal = sliceLabels[:]
					sliceLabels.extend(sliceLabelsOriginal)
				else:
					augSets.append([xElastic, yElastic])

			if 'compound' not in augments:
				sliceLabelsOriginal = sliceLabels[:]
				for n in range(len(augments)):
					sliceLabels.extend(sliceLabelsOriginal)

				for (xaug, yaug) in augSets:
					x = np.concatenate((x, xaug), -1)
					y = np.concatenate((y, yaug), -1)

		# appending to buffer
		if X is None:
			X = x[...]
			Y = y[...]
		else:
			X = np.concatenate((X, x[...]), -1)
			Y = np.concatenate((Y, y[...]), -1)

		caseNames.extend([name] * x.shape[-1])
		hasTumor.extend(sliceLabels)

		# flushing to disk every now and then
		if X.shape[-1] > maxBufferSize:
			flushSize = X.shape[-1]
			# print("Flushing %d entries" % flushSize)
			X = np.moveaxis(X, -1, 0)
			Y = np.moveaxis(Y, -1, 0)
			caseNames = np.array(caseNames,dtype='|S16')
			hasTumor = np.uint8(hasTumor)

			db["slice"][didx:didx + flushSize, ...] = np.float32(X[...])
			db["mask"][didx:didx + flushSize, ...] = Y[...]
			db["case"][didx:didx + flushSize, ...] = caseNames[...]
			db["tumor"][didx:didx + flushSize, ...] = hasTumor[...]

			X = None
			Y = None
			hasTumor = []
			caseNames = []

			didx += flushSize

	# Flushing remainders
	if X is not None:
		if X.shape[-1] > 0:
			flushSize = X.shape[-1]
			# print("Flushing %d entries" % flushSize)
			X = np.moveaxis(X, -1, 0)
			Y = np.moveaxis(Y, -1, 0)
			caseNames = np.array(caseNames, dtype='|S16')
			hasTumor = np.uint8(hasTumor)

			db["slice"][didx:didx + flushSize, ...] = np.float32(X[...])
			db["mask"][didx:didx + flushSize, ...] = Y[...]
			db["case"][didx:didx + flushSize, ...] = caseNames[...]
			db["tumor"][didx:didx + flushSize, ...] = hasTumor[...]

	db.close()

class DataQueuer (threading.Thread):
	
	def __init__(self, db, xName, yName, Q, sidx, extendDim,swapTimeChannel,maskType):
		threading.Thread.__init__(self)

		self.db = db
		self.xName = xName
		self.yName = yName
		self.Q = Q
		self.sidx = sidx
		self.extendDim = extendDim
		self.swapTimeChannel = swapTimeChannel
		self.maskType = maskType

	def run(self):

		try:
			while True:

				for i in self.sidx:

					i = sorted(i)
					X, Y = [], []
					
					# always shuffle within a batch
					shuffle_idx = np.arange(len(i))
					np.random.shuffle(shuffle_idx)
					
					for nx in self.xName: 
						x = self.db[nx][i, ...]
						
						if self.swapTimeChannel:
							# x = x[..., 2:-2]
							x = np.moveaxis(x, -1, 1)

						if self.extendDim:
							x = np.expand_dims(x, -1)

						x = np.float32(x)[shuffle_idx]

						X.append(x)
					
					for ny in self.yName:
						y = self.db[ny][i, ...]

						if self.swapTimeChannel and ny == 'mask' and self.maskType == '3D' and len(y.shape)>3 :
							# y = y[..., 4:5]
							y = np.moveaxis(y, -1, 1)

						if self.maskType == '2D' and ny == 'mask' and len(y.shape)>4:
							y = y[..., y.shape[-1]//2]

						if self.extendDim and ny == 'mask':
							if len(y.shape) == 3 or self.maskType == '3D': 
								y = np.expand_dims(y, -1)
		
						y = np.float32(y)[shuffle_idx]

						Y.append(y)

					# print("Queued")
					self.Q.put([X, Y])
		except:
			return

def dataGenerator(dbPath, xName, yName, batchSize, dataIndex=None, shuffle=True, extendDim=False, maxMemoryUsage = 2, numWorkers = 4, swapTimeChannel=False, maskType = '2D'):
	
	''' This generator queues and outputs batches from a hdf5 database.
	The name of the x entries and y entries must be given as well as the batchSize.
	Optionally, the index of the data to be used for creating batches may be passed as dataIndex.
	This is useful when splitting the database into folds for training.
	
	The max RAM usage for queueing batches may be speicied by maxMemoryUsage (GB).
	The number of threads dedicated to queueing batches is specified by numWorkers '''

	db = h5py.File(dbPath, "r")

	if dataIndex is None:
		totalSamples = db[xName[0]].shape[0]
		idx = np.arange(totalSamples)
	else:
		totalSamples = len(dataIndex)
		idx = dataIndex

	if dataIndex is None:
		# if db contains empty entries due to error during generation, skip these indices
		idx2 = []
		cases = db['case']
		for cdx, c in enumerate(cases):
			if len(c) != 0:
				if cdx in idx:
					idx2.append(cdx)
		
		idx = idx2
		totalSamples = len(idx)

	if shuffle:
		np.random.shuffle(idx)
	
	# creating list of indices of each batch
	sidx = [idx[i: ((i + batchSize) if (i+batchSize)<totalSamples else (totalSamples-1))] for i in range(0, len(idx), batchSize)]

	#  padding batch number to distribute evenly among workers
	if len(sidx) % numWorkers:
		padding = numWorkers - (len(sidx) % numWorkers)
		
		# repeating batches
		for i in range(padding):
			sidx.append(sidx[i])

	n = len(sidx) // numWorkers
	
	# chunking batches for each worker
	sidxChunks = [sidx[i:i + n] for i in range(0, len(sidx), n)]

	# creating a Queue for batches
	# sizeQueueEntry = (db[xName].shape[1] * db[xName].shape[2] * 4 * 2 * batchSize)/(1024.0 ** 3)	# GB
	# maxQueueSize = int(maxMemoryUsage / sizeQueueEntry)
	
	maxQueueSize = 500
	Q = Queue(maxsize = maxQueueSize)

	threads = [ DataQueuer(db,xName,yName,Q,sidxChunks[n],extendDim,swapTimeChannel,maskType) for n in range(numWorkers) ]
	for t in threads:
		t.setDaemon(True)
		t.start()

	try:
		while True:
			x,y = Q.get()
			Q.task_done()
			yield (x,y)

	except:

		# closing db - triggers expcetion in dataQueuer
		db.close()

		# getting elements from queue to remove block of .put() in threads
		for n in range(numWorkers):
			x, y = Q.get_nowait()
			Q.task_done()

		return

def getSampleCount(dbPath,xName):

	with h5py.File(dbPath, "r") as db:
		count = db[xName].shape[-1]

	return count

def calculateRegionProps(slices,masks):

	area = np.zeros((slices.shape[0], 1), dtype=np.float32)
	bbox = np.zeros((slices.shape[0], 4), dtype=np.float32)
	centroid = np.zeros((slices.shape[0], 2), dtype=np.float32)
	solidity = np.zeros((slices.shape[0], 1), dtype=np.float32)

	h = np.float32(slices.shape[1])
	w = np.float32(slices.shape[2])
	
	for idx in range(slices.shape[0]):
		try:
			props = measure.regionprops(masks[idx, ...],slices[idx, ...])
			area[idx] = props[0]['area']
			bbox[idx, ...] = np.array( [props[0]['bbox'][0]/h, props[0]['bbox'][1]/w, props[0]['bbox'][2]/h, props[0]['bbox'][3]/w] ) 
			centroid[idx, ...] = np.array( [props[0]['centroid'][0]/h, props[0]['centroid'][1]/w] ) 
			solidity[idx] =  props[0]['solidity']
		except Exception as err:
			print(err)
			continue

	return area,bbox,centroid,solidity

def calculateWavelet(slices):
	wav = np.zeros(slices.shape, dtype=np.float32)

	for idx in range(slices.shape[0]):
		try:
			wav[idx,...] = pywt.swt2(slices[idx,...], 'haar', level=1)[0][0]
		except Exception as err:
			print(err)
			continue
			
	return wav

def unshuffle(l, order):
	l_out = np.zeros(l.shape)
	for i, j in enumerate(order):
		l_out[j, ...] = l[i, ...]
	return l_out

def cleanMask(m):

	if np.max(m) > 1:
		m = np.clip(m,0,1)
	 
	cnts = cv2.findContours(m, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
	
	m1 = [cv2.contourArea(x) for x in cnts]
   
	if not m1:
	   return m

	n = np.max(m1)
	i = np.where(m1==n)
	i = int(i[0])
	cv2.drawContours(m,cnts,i,1,cv2.FILLED)
	
	if len(cnts)>1:
		for j in range(len(m1)):
			if j!=i:
				cv2.drawContours(m,cnts,j,0,cv2.FILLED)

	return m

def createDatabase2D_Coronal(dbMetaData, dbPath, wavelet=False, onlyTumor=False, augments=[], crop=None, resizeTo=None, rescale=False, clipHU=None, expandMasks=False, minArea=100, applyGaussianFilter=False):
	''' This function creates an HDF5 database file with the 2D tumor slices
		under 'slice' and their coressponding masks under 'mask'. The name of 
		the case file from which the slice came from is under 'case' '''

	X = None         # slices
	Y = None         # masks
	hasTumor = []	 # whether slice has tumor(1) or not(0)
	caseNames = []   # name of case where slice came from

	if crop is None:
		cy1, cy2 = (0, 120 if resizeTo is None else resizeTo[0])
		cx1, cx2 = (0, 512 if resizeTo is None else resizeTo[1])
	else:
		cx1, cy1, cx2, cy2 = crop

	if resizeTo is None:
		w = cx2 - cx1
		h = cy2 - cy1
	else:
		w = resizeTo[0]
		h = resizeTo[1]

	nTotalEntries, nBaseEntries = getTotalEntryCount(dbMetaData, onlyTumor, augments, plane='yz')

	try:
		print("\nCreating database with %d entries" % nTotalEntries)
		print("Each slice has shape %d x %d" % (h, w))
		ensureDir(dbPath)
		db = h5py.File(dbPath, mode='w')
		db.create_dataset("slice", (nTotalEntries, h, w,), np.float32)
		db.create_dataset("mask",  (nTotalEntries, h, w,),np.float32 if applyGaussianFilter else np.uint8)
		db.create_dataset("case",  (nTotalEntries,), np.dtype('|S16'))
		db.create_dataset("tumor", (nTotalEntries,), np.uint8)			# binary label
	except Exception as err:
		print("Error creating database : ", err)
		db.close()
		return

	didx = 0				# database index
	maxBufferSize = 200		# flushing to disk every 200 entries
	bar = pbar.ProgressBar(max_value=len(dbMetaData))

	print("\nGenerating %d Entries from %d (%dx)" %(nTotalEntries, nBaseEntries, (nTotalEntries//nBaseEntries)))

	if len(augments) > 0:
		if 'rot90' in augments:
			print("Applying 90 degree rotation")
		if 'rot180' in augments:
			print("Applying 180 degree rotation")
		if 'rot270' in augments:
			print("Applying 270 degree rotation")
		if 'horFlip' in augments:
			print("Applying Horizontal Flip")
		if 'verFlip' in augments:
			print("Applying Vertical Flip")
		if 'elasticTransform' in augments:
			print("Applying Random Elastic Transformations")
		if 'compound' in augments:
			print("Compounding all augmentations")

	for idx, name in enumerate(dbMetaData):

		bar.update(idx + 1)
		case = dbMetaData[name]

		if name in corrections:
			if corrections[name] is not None:
				yCorr, xCorr = corrections[name]
			else:
				continue
		else:
			yCorr, xCorr = 0, 0

		slices = stackSlices(case)
		masks, midx = genMask(case)

		slices = changePlane(slices, 'yz')
		masks = changePlane(masks, 'yz')
		midx = getMaskIndex(masks)

		sliceLabels = np.uint8([0] * slices.shape[-1])
		sliceLabels[midx] = 1

		# adjusting HU scale for other manufacturer (CMS)
		if np.min(slices) < 0:
			slices = slices + 1024

		# applying corrections
		if yCorr or xCorr:
			masks = np.roll(masks, yCorr, 0)
			masks = np.roll(masks, xCorr, 1)

		# resampling voxels to be 0.97 x 0.97 x 0.97 mm3
		# current dimension of voxels (mm)
		dz0 = case['sliceThickness']
		dx0, dy0 = case['pixelSpacing']

		slices, _ = resampleVoxels(slices, oldDim=(dz0, dy0, dx0), newDim=(0.97, 0.97, 0.97))
		masks, _ = resampleVoxels(masks, oldDim=(dz0, dy0, dx0), newDim=(0.97, 0.97, 0.97))

		# clipping HU values
		if clipHU is not None:
			slices = np.clip(slices, clipHU[0], clipHU[1])

			# rescale values to between 0.0 and 1.0
			if rescale:
				slices += (0-clipHU[0])
				slices = np.float64(slices)
				slices = np.divide(slices, ((clipHU[1]-clipHU[0]) * 1.0))

		# padding bottom
		if slices.shape[0] < h:
			nPad = h - slices.shape[0]
			padding = np.zeros((nPad, slices.shape[1],slices.shape[2]))
			slices = np.concatenate((slices,padding),axis=0) 
			masks = np.concatenate((masks,padding),axis=0)
			print("\nPadding %s by %d layers\n" % ( name,nPad))

		# cropping
		x = slices[cy1:cy2, cx1:cx2, :]
		y = masks[cy1:cy2, cx1:cx2, :]
		
		# print("\nBefore Crop", end = " ")
		# print(slices.shape, end="\t")
		# print("\nAfter Crop", end = " ")
		# print(x.shape, end = "\n")

		# TODO: use better interpolator so that values dont change too much
		# resizing
		if resizeTo is not None:
			xResized = None
			yResized = None

			for sidx in range(x.shape[-1]):
				xr = np.expand_dims(cv2.resize(x[..., sidx], resizeTo, interpolation=cv2.INTER_NEAREST), -1)
				yr = np.expand_dims(cv2.resize(y[..., sidx], resizeTo, interpolation=cv2.INTER_NEAREST), -1)

				if xResized is None:
					xResized = xr
					yResized = yr
				else:
					xResized = np.concatenate((xResized, xr), -1)
					yResized = np.concatenate((yResized, yr), -1)

			x = xResized
			y = yResized

		# taking only tumor slices
		if onlyTumor:
			x = x[..., midx]
			y = y[..., midx]
			sliceLabels = sliceLabels[midx]
			sliceLabels = sliceLabels.tolist()
		# taking random sample
		else:
			all_idx = np.arange(0, x.shape[-1])
			non_tumor_idx = [ii for ii in all_idx if ii not in midx]
			rand_sample = np.random.choice(non_tumor_idx, 100, replace=True if len(non_tumor_idx)<100 else False)
			rand_sample = sorted(rand_sample)

			x = x[..., rand_sample]
			y = y[..., rand_sample]
			sliceLabels = [0] * 100


		# aplying dialation on masks
		if expandMasks:

			for i in range(masks.shape[-1]):
				m = y[..., i]
				s = x[..., i]

				if np.sum(m) == 0:
					continue

				props = measure.regionprops(m, s)[0]

				if props['area'] > minArea:
					continue

				passes = 0
				while props['area'] < minArea and passes < 10:
					passes += 1
					l = np.ceil(np.sqrt(props['area']))
					l = np.int16(l)
					l = 3 if l < 3 else l
					l = l if l % 2 else l + 1
					kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (l, l))
					m = cv2.dilate(m, kernel, iterations=1)
					props = measure.regionprops(m, s)[0]

				# plt.subplot(1, 2, 1)
				# plt.imshow(y[..., i].reshape(cy2 - cy1, cx2 - cx1))
				# plt.title("Original Mask")

				# plt.subplot(1, 2, 2)
				# plt.imshow(m.reshape(cy2 - cy1, cx2 - cx1))
				# plt.title("Expanded Mask after %d passes" % passes)
				# plt.show()

				# time.sleep(1)
				# plt.close()

				y[..., i] = m

		# smoothing out masks
		if applyGaussianFilter:
			for i in range(masks.shape[-1]):
				m = y[..., i]

				if np.sum(m) == 0:
					continue

				m = np.float32(m)
				m = filters.gaussian_filter(m, 1.5)

				# plt.subplot(1, 2, 1)
				# plt.imshow(y[..., i].reshape(cy2 - cy1, cx2 - cx1))
				# plt.title("Original Mask")

				# plt.subplot(1, 2, 2)
				# plt.imshow(m.reshape(cy2 - cy1, cx2 - cx1))
				# plt.title("Filtered Mask")
				# plt.show()

				y[..., i] = m

		# applying augmentations
		try:
			if len(augments) > 0:

				augSets = []

				if 'rot90' in augments or 'rot180' in augments or 'rot270' in augments:
					xRot = None
					yRot = None

					rotMatrices = []
					rows = (cy2-cy1) if resizeTo is None else resizeTo[1]
					cols = (cx2-cx1) if resizeTo is None else resizeTo[0]

					if 'rot90' in augments:
						rotMatrices.append(cv2.getRotationMatrix2D((cols // 2, rows // 2), 90, 1))
					if 'rot180' in augments:
						rotMatrices.append(cv2.getRotationMatrix2D((cols // 2, rows // 2), 180, 1))
					if 'rot270' in augments:
						rotMatrices.append(cv2.getRotationMatrix2D((cols // 2, rows // 2), 270, 1))

					for rotMat in rotMatrices:
						for idx in range(x.shape[-1]):
							s = x[..., idx]
							m = y[..., idx]

							if xRot is None:
								xRot = cv2.warpAffine(s, rotMat, (cols, rows))
								yRot = cv2.warpAffine(m, rotMat, (cols, rows))
							else:
								xRot = np.dstack((xRot, cv2.warpAffine(s, rotMat, (cols, rows))))
								yRot = np.dstack((yRot, cv2.warpAffine(m, rotMat, (cols, rows))))

					augSets.append([xRot, yRot])

				if 'horFlip' in augments:
					xHorFlipped = None
					yHorFlipped = None

					for idx in range(x.shape[-1]):
						s = x[..., idx]
						m = y[..., idx]

						if xHorFlipped is None:
							xHorFlipped = cv2.flip(s, 1)
							yHorFlipped = cv2.flip(m, 1)
						else:
							xHorFlipped = np.dstack((xHorFlipped, (cv2.flip(s, 1))))
							yHorFlipped = np.dstack((yHorFlipped, (cv2.flip(m, 1))))

					augSets.append([xHorFlipped, yHorFlipped])

				if 'verFlip' in augments:
					xVerFlipped = None
					yVerFlipped = None

					for idx in range(x.shape[-1]):
						s = x[..., idx]
						m = y[..., idx]
						if xVerFlipped is None:
							xVerFlipped = cv2.flip(s, 1)
							yVerFlipped = cv2.flip(m, 1)
						else:
							xVerFlipped = np.dstack((xVerFlipped, (cv2.flip(s, 1))))
							yVerFlipped = np.dstack((yVerFlipped, (cv2.flip(m, 1))))

					augSets.append([xVerFlipped, yVerFlipped])

				if 'elasticTransform' in augments:
					xElastic = None
					yElastic = None

					if 'compound' in augments:
						sliceLabelsOriginal = sliceLabels[:]
						for n in range(len(augments) - 2):
							# adding labels for previous augmentatioons
							sliceLabels.extend(sliceLabelsOriginal)

						for (xaug, yaug) in augSets:
							x = np.dstack((x, xaug))
							y = np.dstack((y, yaug))

					for idx in range(x.shape[-1]):
						s = x[..., idx]
						m = y[..., idx]

						sigma = np.random.uniform(0.07, 0.15, 1)
						affineFactor = np.random.uniform(0.07, 0.15, 1)
						xe, ye = elasticTransform(
							s, m, sigma*s.shape[1], affineFactor*s.shape[1], 1234)

						if xElastic is None:
							xElastic = xe
							yElastic = ye
						else:
							xElastic = np.dstack((xElastic, xe))
							yElastic = np.dstack((yElastic, ye))

					if 'compound' in augments:
						x = np.dstack((x, xElastic))
						y = np.dstack((y, yElastic))
						sliceLabelsOriginal = sliceLabels[:]
						sliceLabels.extend(sliceLabelsOriginal)
					else:
						augSets.append([xElastic, yElastic])

				if 'compound' not in augments:
					sliceLabelsOriginal = sliceLabels[:]
					for n in range(len(augments)):
						sliceLabels.extend(sliceLabelsOriginal)

					for (xaug, yaug) in augSets:
						x = np.dstack((x, xaug))
						y = np.dstack((y, yaug))
		except Exception as err:
			print(name)
			print(err)
			continue

		# appending to buffer
		try:
			if X is None:
				X = x[...]
				Y = y[...]
			else:
				X = np.dstack((X, x[...]))
				Y = np.dstack((Y, y[...]))
		except Exception as err:
			print("\nError while appending to buffer")
			print(name)
			print(err)
			print("Tried to append", end = " ")
			print(x.shape, end = " ")
			print(" to :", end = " ")
			print(X.shape)
			continue

		caseNames.extend([name] * x.shape[-1])
		hasTumor.extend(sliceLabels)

		# flushing to disk every now and then
		if X.shape[-1] > maxBufferSize:
			flushSize = X.shape[-1]
			# print("Flushing %d entries" % flushSize)
			X = np.moveaxis(X, -1, 0)
			Y = np.moveaxis(Y, -1, 0)
			caseNames = np.array(caseNames, dtype='|S16')
			hasTumor = np.uint8(hasTumor)

			db["slice"][didx:didx + flushSize, ...] = np.float32(X[...])
			db["mask"][didx:didx + flushSize, ...] = Y[...]
			db["case"][didx:didx + flushSize, ...] = caseNames[...]
			db["tumor"][didx:didx + flushSize, ...] = hasTumor[...]

			if wavelet:
				wav = calculateWavelet(X)
				db["wavelet"][didx:didx + flushSize, ...] = wav[...]

			X = None
			Y = None
			hasTumor = []
			caseNames = []

			didx += flushSize

	# Flushing remainders
	if X is not None:
		if X.shape[-1] > 0:
			flushSize = X.shape[-1]
			# print("Flushing %d entries" % flushSize)
			X = np.moveaxis(X, -1, 0)
			Y = np.moveaxis(Y, -1, 0)
			caseNames = np.array(caseNames, dtype='|S16')
			hasTumor = np.uint8(hasTumor)

			db["slice"][didx:didx + flushSize, ...] = np.float32(X[...])
			db["mask"][didx:didx + flushSize, ...] = Y[...]
			db["case"][didx:didx + flushSize, ...] = caseNames[...]
			db["tumor"][didx:didx + flushSize, ...] = hasTumor[...]
		
			if wavelet:
				wav = calculateWavelet(X)
				db["wavelet"][didx:didx + flushSize, ...] = wav[...]

	db.close()

def getCVIndices(dbPath, n_splits=3, augPerSlice = 12, include_aug_in_test=False, ignore_compound=True, random_state=1337):

	db = h5py.File(dbPath,'r')

	all_cases = db['case'][...]
	cases = np.unique(all_cases)
	cases = np.array([c for c in cases if len(c) > 0])
	all_idx = np.arange(len(all_cases))

	sliceCount = { c : len(all_cases[all_cases==c])//augPerSlice for c in cases }

	kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
	kf.get_n_splits(cases)

	folds = []
	for train_case_idx, test_case_idx in kf.split(cases):
		trainCases = cases[train_case_idx]
		testCases = cases[test_case_idx]

		train_idx = []
		for c in trainCases:
			if ignore_compound:
				train_idx.extend( all_idx[all_cases==c][0:7*sliceCount[c]] )
			else:
				train_idx.extend( all_idx[all_cases==c] )
				
		test_idx = []
		for c in testCases:
			if include_aug_in_test:
				if ignore_compound:
					test_idx.extend( all_idx[all_cases==c][0:7*sliceCount[c]] )
				else:
					test_idx.extend( all_idx[all_cases==c] )
			else:
				test_idx.extend(all_idx[all_cases == c][0:sliceCount[c]])	# first N slices are unaugmented
	
		folds.append( [sorted(train_idx),sorted(test_idx)] )
	
	return folds

def createNStacks(x, nSlices, sliceStep=1):
	''' This function creates mini stacks of n-slices from the full
	3D CT scan stack '''

	x_stacks = None
	for sidx in range(0,  x.shape[-1]):
		# getting index of top neighbours
		if sidx-(nSlices*sliceStep) >= 0:
			top_idx = list(range(sidx-(nSlices*sliceStep), sidx, sliceStep))
		else:
			top_idx = sorted(list(range(sidx-sliceStep, 0, -sliceStep)))

		# pad by repeating first slice
		if len(top_idx) < nSlices:
			nPad = nSlices - len(top_idx)
			padSlices = [0] * nPad
			padSlices.extend(top_idx)
			top_idx = padSlices

		# getting index of bottom neighbours
		if sidx+(nSlices*sliceStep) < x.shape[-1]:
			bot_idx = list(range(sidx+sliceStep, sidx +
								(nSlices*sliceStep)+1, sliceStep))
		else:
			bot_idx = list(range(sidx+sliceStep, x.shape[-1], sliceStep))

		# pad by repeating last slice
		if len(bot_idx) < nSlices:
			nPad = nSlices - len(bot_idx)
			bot_idx.extend([x.shape[-1]-1] * nPad)

		# getting slices
		cen_slice = np.expand_dims(x[..., sidx], -1)
		top_slices = x[..., top_idx]
		bot_slices = x[..., bot_idx]
		x0 = np.concatenate((top_slices, cen_slice, bot_slices), -1)

		if x_stacks is None:
			x_stacks = np.expand_dims(x0, -1)
		else:
			x_stacks = np.concatenate((x_stacks, np.expand_dims(x0, -1)), -1)

	return x_stacks
