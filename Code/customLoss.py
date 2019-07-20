import numpy as np
from keras import backend as K


# def dice_coef(y_true, y_pred, threshold=0.2, smooth=1.0):

# 	''' This function calculates the and returns the of dice coefficient
# 	 between the tensors y_true (ground truth) and y_pred (model output)
# 	 by applying a thershold on the predicted mask '''

# 	y_true_f = K.flatten(y_true)
# 	y_pred_f = K.flatten(y_pred)
# 	y_pred_f = K.round(y_pred_f + (0.5-threshold))
# 	y_pred_f = K.clip(y_pred_f,0,1)

# 	intersection = K.sum(y_true_f * y_pred_f)

# 	return -(2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef(y_true, y_pred, smooth=1):
	
	''' This function calculates the dice coeficient for each entry in batch,
	then averages over the batch '''

	intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
	union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])

	return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def dice_coef_3d(y_true, y_pred, smooth=1):
	''' This function calculates the dice coeficient for each slice in stack for each entry in batch,
	then averages over the batch. axis = 1 -> time / slice axis '''

	intersection = K.sum(y_true * y_pred, axis=[2, 3, 4])
	union = K.sum(y_true, axis=[2, 3, 4]) + K.sum(y_pred, axis=[2, 3, 4])
	dice_slices = K.mean((2. * intersection + smooth) / (union + smooth), axis = 1)		# mean over stack

	return K.mean(dice_slices, axis=0)	# mean over batch


def dice_coef_singleChannel(y_true, y_pred, ch):

	''' This function calculates the dice coefficent over a single channel
	and then averages over the batch '''

	y_true_ch = y_true[..., ch:ch+1]
	y_pred_ch = y_pred[..., ch:ch+1]
	
	return dice_coef(y_true_ch,y_pred_ch)

def dice_coef_multi(y_true, y_pred, channelWeights = [1.0,0.1,0.1]):

	''' This function computes the dice coefficient over each channel, then returns
	the weighted average of the dice coefficents '''

	avg_dice = None

	for ch,w in enumerate(channelWeights):
		if avg_dice is None:
			avg_dice = w * dice_coef_singleChannel(y_true,y_pred,ch)
		else:
			avg_dice += w * dice_coef_singleChannel(y_true,y_pred,ch)
	'''
	w = channelWeights
	avg_dice = w[0] * dice_coef_singleChannel(y_true, y_pred, 0)
	avg_dice = K.sum(avg_dice,w[1] * dice_coef_singleChannel(y_true, y_pred, 1))
	avg_dice = K.sum(avg_dice,w[2] * dice_coef_singleChannel(y_true, y_pred, 2))
	'''

	return avg_dice

def dice_coef_loss(y_true, y_pred):

	''' This function calculates the and returns the negative of dice coefficient
	 between the tensors y_true (ground truth) and y_pred (model output) '''

	return -dice_coef(y_true,y_pred)

def log_dice_coef_loss(y_true, y_pred):
	''' This function calculates the and returns the negative of log of the 
	dice coefficient between the tensors y_true (ground truth) and y_pred (model output) '''

	return - K.log(dice_coef(y_true,y_pred) + 1e-8)


def log_dice_coef_3d_loss(y_true, y_pred):
	''' This function calculates the and returns the negative of log of the 
	dice coefficient between the 3d tensors y_true (ground truth) and y_pred (model output) '''

	return - K.log(dice_coef_3d(y_true, y_pred) + 1e-8)

def log_dice_coef_loss_multi(y_true, y_pred):
	''' This function calculates the and returns the negative of log of the weighted
	dice coefficient between the channels in the tensors y_true (ground truth) and 
	y_pred (model output) '''

	return -K.log(dice_coef_multi(y_true,y_pred) + 1e-8)

def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1.0):
	''' This function returns the negative of the tversky_index.
	alpha = weight of '0' class i.e bg, beta = weight of '1' class i.e tumor '''

	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)

	truepos = K.sum(y_true_f * y_pred_f)
	fp_and_fn = (alpha * K.sum(y_pred_f * (1 - y_true_f)) + beta * K.sum((1 - y_pred_f) * y_true_f))

	return -(truepos + smooth) / (truepos + smooth + fp_and_fn)

def jaccard_coef_logloss(y_true, y_pred, smooth=1.0):
	''' This function returns the negative logarithm of the 
	jaccard coefficient. '''

	y_true_f = K.batch_flatten(y_true)
	y_pred_f = K.batch_flatten(y_pred)

	truepos = K.sum(y_true_f * y_pred_f)
	falsepos = K.sum(y_pred_f) - truepos
	falseneg = K.sum(y_true_f) - truepos

	jaccard = (truepos + smooth) / (smooth + truepos + falseneg + falsepos)

	return -K.log(jaccard + smooth)

def mape (y_true, y_pred):

	''' This function returns the mean absolute percentage error
	without scaling to 100, i.e between 0 and 1 '''
	
	diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
											K.epsilon(),
											None))
	return K.mean(diff, axis=-1)

def mean_length_error(y_true, y_pred):
	y_true_f = K.sum(K.round(K.flatten(y_true)))
	y_pred_f = K.sum(K.round(K.flatten(y_pred)))
	delta = (y_pred_f - y_true_f)
	return K.mean(K.tanh(delta))

def dice_coef_channel(ch):

	''' This is a closure that returns a function which calculates 
	the dice coefficient over a single channel of the output. It can be 
	used with the Keras API when compiling model by calling the closure
	with the channel index of choice '''

	def dice_ch(y_true,y_pred):
		y_true_ch = y_true[..., ch:ch+1]
		y_pred_ch = y_pred[..., ch:ch+1]
		return dice_coef(y_true_ch, y_pred_ch)

	dice_ch.__name__ = 'dice_ch_%d' % ch
	return dice_ch

def weighted_crossentropy(weights=[1,1]):

	def bin_xentropy(y_true, y_pred):
		y_pred = K.clip(y_pred, K.epsilon(), 1)
		logloss = -(weights[1] * y_true * K.log(y_pred + K.epsilon()) + weights[0] * (1-y_true) * K.log(1 - y_pred + K.epsilon()))
		return K.mean(logloss, axis=0)

	bin_xentropy.__name__ = 'weighted_bin_xentropy'
	return bin_xentropy

def weighted_binary_accuracy(weight=0.25, threshold=0.5):

	def bin_acc(y_true,y_pred):
		y_pred = K.round(y_pred + (0.5-threshold))
		y_pred = K.clip(y_pred, K.epsilon(), 1)

		acc = weight * y_true * (y_true-y_pred) + (1-y_true) * (y_pred-y_true)
		return K.mean(acc, axis=0)

	return bin_acc
