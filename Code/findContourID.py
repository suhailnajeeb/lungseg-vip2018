import pydicom as dicom
import glob
import os
import progressbar as pbar
import numpy as np

def getMetaData2(pathToData):

	# getting all case folders
	caseFolders = sorted(glob.glob(pathToData + "/*"))

	# getting scan data and annotations
	bar = pbar.ProgressBar(max_value=len(caseFolders))      # progressbar to keep track
	tumorIDs = []
	
	for idx,case in enumerate(caseFolders):
		bar.update(idx+1)

		# getting folders inside each case folder
		subFolders = glob.glob(case + '/*/*')
		
		# extracting the case name and studyID from path
		caseName = os.path.basename(case)

		# dcm files in two folders; one for scans, another for annotation
		try:
			dcmFilesA = glob.glob(subFolders[0] + '/*.dcm')
			if len(subFolders) > 1:
				dcmFilesB = glob.glob(subFolders[1] + '/*.dcm')
		except Exception as err:
			print(caseName, err)
			continue

		if len(subFolders) > 1:
			# scanData is in folder with more than one dcm file
			scanData = dcmFilesA if len(dcmFilesA) > 1 else dcmFilesB
			annotData = dcmFilesB[0] if len(dcmFilesA) > 1 else dcmFilesA[0]
		else:
			print(case)
			print("No Annotations\n")
	
		if len(subFolders) > 1:
			# parsing annotation data
			dcm = dicom.read_file(annotData)
			
			segIDs = []
			for segObservations in dcm.RTROIObservationsSequence:
				segIDs.append(segObservations.ROIObservationLabel)
			
			segID = segIDs.index('GTV-1')		# segment class to use as roi, 1 = tumor
			tumorIDs.append(segID)


	return tumorIDs


t = getMetaData2('../Dataset3/VIP_CUP18_TrainingData')
print(np.unique(t))

