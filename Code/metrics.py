import numpy as np
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import generate_binary_structure,  binary_erosion, distance_transform_edt
from scipy.ndimage.measurements import label, find_objects
from scipy.stats import pearsonr


smooth = 1.

def dice_coef(y_true, y_pred, threshold = 0.5):

	''' This function calculates the dice coefficient 
	between the tensors y_true (ground truth) and y_pred (model output) '''

	y_true_f = y_true.flatten()
	y_pred_f = y_pred.flatten()
	y_pred_f = np.round(y_pred_f + (0.5-threshold))
	y_pred_f = np.clip(y_pred_f, 0, 1)

	intersection = np.sum(y_true_f * y_pred_f)
	
	return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def surfd(y_true, y_pred, sampling=1, connectivity=1):
	''' returns the mean surface distance between masks '''

	input_1 = np.atleast_1d(y_true.astype(np.bool))
	input_2 = np.atleast_1d(y_pred.astype(np.bool))

	conn = generate_binary_structure(input_1.ndim, connectivity)

	S = input_1 ^ binary_erosion(input_1, conn)
	Sprime = input_2 ^ binary_erosion(input_2, conn)

	dta = distance_transform_edt(~S, sampling)
	dtb = distance_transform_edt(~Sprime, sampling)
	sds = np.concatenate([np.ravel(dta[Sprime != 0]), np.ravel(dtb[S != 0])])

	return sds

def mean_surfd(input1, input2, sampling=1, connectivity=1):
	try:
		msd = np.mean(surfd(input1, input2, sampling=sampling, connectivity=connectivity))
	except:
		msd = 0

	return msd

def hausdorff_n(input1, input2, n, sampling=1, connectivity=1):

	if n < 0 or n > 100:
		return -1

	try:
		hn = np.percentile(surfd(input1, input2, sampling=sampling, connectivity=connectivity), n)
	except:
		hn = 0

	return hn

def tversky_index(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1.0):
	''' This function returns the negative of the tversky_index.
	alpha = weight of '0' class i.e bg, beta = weight of '1' class i.e tumor '''

	y_true_f = y_true.flatten()
	y_pred_f = y_pred.flatten()

	truepos = np.sum(y_true_f * y_pred_f)
	fp_and_fn = (alpha * np.sum(y_pred_f * (1 - y_true_f)) + beta * np.sum((1 - y_pred_f) * y_true_f))

	return (truepos + smooth) / (truepos + smooth + fp_and_fn)
