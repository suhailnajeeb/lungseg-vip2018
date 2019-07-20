from keras.models import Model
import customInitializer

# core layers
from keras.layers import Lambda, Input, Dense, Dropout, Flatten, Activation
from keras.layers.merge import add, concatenate, average, multiply
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU, LeakyReLU, ReLU
from keras.regularizers import l2

# layers for 2d convnet
from keras.layers import Reshape, Conv2D, LocallyConnected2D, Conv2DTranspose, MaxPooling2D, UpSampling2D,  SpatialDropout2D, Cropping2D, ZeroPadding2D, GlobalMaxPooling2D

# layers of 3d convnet
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D,  AveragePooling2D, AveragePooling3D,  SpatialDropout3D, Cropping3D, ZeroPadding3D

# recurrent layers
from keras.layers import TimeDistributed, ConvLSTM2D

# other utils
from keras import backend as K
from keras.utils import plot_model
from keras.models import load_model
	
import customLoss as cl

import sys

modelArch = {}												
addModel = lambda f:modelArch.setdefault(f.__name__,f)


# build and return model
def makeModel(architecture,verbose=True):

	model = modelArch[architecture]()

	if verbose:
		print(model.summary(line_length=150))
	
	return model


# setting up models with different weights
def setupModels(architecture, weightsPath, verbose = False):

	models = []

	for w in weightsPath:
		model = makeModel(architecture, verbose=verbose)
		
		adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-06, decay=0.00001)
		model.compile(loss='mae', optimizer=adam , metrics=['mae'])
		
		model._make_predict_function()
		model.load_weights(w)
		models.append(model)

	return models

@addModel
def maskNet001():

	in1 = Input((512,512,1))

	#encoder
	stack1E = Conv2D(64, (3, 3), strides=2, activation='relu', padding='same',data_format='channels_last')(in1)
	stack1E = BatchNormalization()(stack1E)
	
	stack2E = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same',data_format='channels_last')(stack1E)
	stack2E = BatchNormalization()(stack2E)
	stack2E = MaxPooling2D(pool_size=(2, 2))(stack2E)

	stack3E = Conv2D(128, (3, 3), strides=2, activation='relu', padding='same',data_format='channels_last')(stack2E)
	stack3E = BatchNormalization()(stack3E)

	stack4E = Conv2D(128, (3, 3), strides=1, activation='relu', padding='same',data_format='channels_last')(stack3E)
	stack4E = BatchNormalization()(stack4E)
	stack4E = MaxPooling2D(pool_size=(2, 2))(stack4E)

	stack5E = Conv2D(256, (3, 3), activation='relu', padding='same',data_format='channels_last')(stack4E)
	stack5E = BatchNormalization()(stack5E)
	stack5E = MaxPooling2D(pool_size=(2, 2))(stack5E)

	#decoder
	stack5D = Dropout(0.5)(stack5E)
	stack5D = Conv2D(256, (3, 3), activation='relu', padding='same',data_format='channels_last')(stack5D)
	# stack5D = add([stack5D,stack5E])

	stack4D = UpSampling2D((2, 2))(stack5D)
	stack4D = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(stack4D)
	stack4D = add([stack4D,stack4E])
	stack4D = BatchNormalization()(stack4D)

	stack3D = UpSampling2D((2, 2))(stack4D)
	stack3D = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(stack3D)
	stack3D = add([stack3D,stack3E])
	stack3D = BatchNormalization()(stack3D)
	
	stack2D = UpSampling2D((2, 2))(stack3D)
	stack2D = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(stack2D)
	stack2D = BatchNormalization()(stack2D)

	stack1D = UpSampling2D((2, 2))(stack2D)
	stack1D = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(stack1D)
	stack1D = BatchNormalization()(stack1D)

	stack0D = UpSampling2D((2, 2))(stack1D)
	stack0D = Conv2D(1, (3, 3), activation='sigmoid', padding='same',data_format='channels_last')(stack0D)

	return Model(inputs=in1, outputs=stack0D)

@addModel
def maskNet002():

	in1 = Input((350,350,1))

	#encoder
	stack1E = Conv2D(32, (3, 3), strides=1, dilation_rate=2, activation='relu', padding='same',data_format='channels_last')(in1)
	stack1E = BatchNormalization()(stack1E)
	
	stack2E = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same',data_format='channels_last')(stack1E)
	stack2E = BatchNormalization()(stack2E)
	stack2E = MaxPooling2D(pool_size=(2, 2))(stack2E)

	stack3E = Conv2D(64, (3, 3), strides=2, activation='relu', padding='same',data_format='channels_last')(stack2E)
	stack3E = BatchNormalization()(stack3E)

	stack4E = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same',data_format='channels_last')(stack3E)
	stack4E = BatchNormalization()(stack4E)
	stack4E = MaxPooling2D(pool_size=(2, 2))(stack4E)

	stack5E = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(stack4E)
	stack5E = BatchNormalization()(stack5E)
	stack5E = MaxPooling2D(pool_size=(2, 2))(stack5E)

	#decoder
	# stack5D = Dropout(0.5)(stack5E)
	stack5D = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(stack5E)
	stack5D = BatchNormalization()(stack5D)
	stack5D = concatenate([stack5D,stack5E])
	stack5D = SpatialDropout2D(0.25)(stack5D)

	stack4D = UpSampling2D((2, 2))(stack5D)
	stack4D = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(stack4D)
	stack4D = BatchNormalization()(stack4D)
	stack4D = concatenate([stack4D,stack4E])
	stack4D = SpatialDropout2D(0.25)(stack4D)
	
	stack3D = UpSampling2D((2, 2))(stack4D)
	stack3D = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(stack3D)
	stack3D = BatchNormalization()(stack3D)
	stack3D = concatenate([stack3D,stack3E])
	stack3D = SpatialDropout2D(0.25)(stack3D)

	stack2D = UpSampling2D((2, 2))(stack3D)
	stack2D = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(stack2D)
	stack2D = BatchNormalization()(stack2D)

	stack1D = UpSampling2D((2, 2))(stack2D)
	stack1D = Conv2D(1, (3, 3), activation='sigmoid', padding='same',data_format='channels_last')(stack1D)
	stack1D = Cropping2D(cropping=1)(stack1D)
	
	return Model(inputs=in1, outputs=stack1D)

@addModel
def maskNet002a():

	''' Change Log :  
	added conv after final size achieved. 
	increased no. of filters in stack 4,5
	added 1-2 dialation rate pairs in encoder
	increased spatial dropout to 0.5 in stack5D
	'''
	in1 = Input((224, 224, 1))

	#encoder
	stack1E = Conv2D(32, (3, 3), strides=1, dilation_rate=1, activation='relu',  padding='same',data_format='channels_last')(in1)
	stack1E = BatchNormalization()(stack1E)
	
	stack2E = Conv2D(64, (3, 3), strides=1, dilation_rate=2, activation='relu',  padding='same', data_format='channels_last')(stack1E)
	stack2E = BatchNormalization()(stack2E)
	stack2E = MaxPooling2D(pool_size=(2, 2))(stack2E)

	stack3E = Conv2D(64, (3, 3), strides=2, dilation_rate=1,activation='relu',  padding='same', data_format='channels_last')(stack2E)
	stack3E = BatchNormalization()(stack3E)

	stack4E = Conv2D(128, (3, 3), strides=1, dilation_rate=2, activation='relu',  padding='same', data_format='channels_last')(stack3E)
	stack4E = BatchNormalization()(stack4E)
	stack4E = MaxPooling2D(pool_size=(2, 2))(stack4E)

	stack5E = Conv2D(256, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack4E)
	stack5E = BatchNormalization()(stack5E)
	stack5E = MaxPooling2D(pool_size=(2, 2))(stack5E)

	#decoder
	stack5D = Conv2D(256, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack5E)
	stack5D = BatchNormalization()(stack5D)
	stack5D = concatenate([stack5D,stack5E])
	stack5D = SpatialDropout2D(0.50)(stack5D)

	stack4D = UpSampling2D((2, 2))(stack5D)
	stack4D = Conv2D(128, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack4D)
	stack4D = BatchNormalization()(stack4D)
	stack4D = concatenate([stack4D,stack4E])
	stack4D = SpatialDropout2D(0.25)(stack4D)
	
	stack3D = UpSampling2D((2, 2))(stack4D)
	stack3D = Conv2D(64, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack3D)
	stack3D = BatchNormalization()(stack3D)
	stack3D = concatenate([stack3D,stack3E])
	stack3D = SpatialDropout2D(0.25)(stack3D)

	stack2D = UpSampling2D((2, 2))(stack3D)
	stack2D = Conv2D(64, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack2D)
	stack2D = BatchNormalization()(stack2D)

	stack1D = UpSampling2D((2, 2))(stack2D)
	stack1D = Conv2D(64, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack1D)
	stack1D = BatchNormalization()(stack1D)

	stack1D = Conv2D(32, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack1D)
	stack1D = BatchNormalization()(stack1D)

	stack1D = Conv2D(1, (1, 1), activation='sigmoid', padding='same',data_format='channels_last')(stack1D)
	# stack1D = Cropping2D(cropping=1)(stack1D)
	
	return Model(inputs=in1, outputs=stack1D)

@addModel
def maskNet002b():

	''' Change Log :  
	based on maskNet002b; replaced upsample2d by conv2dtranspose
	move concatenate to after conv2dtranspose
	decreased spatial droupout to 0.125 for last dropout layer
	'''
	in1 = Input((352, 352, 1))

	#encoder
	stack1E = Conv2D(32, (3, 3), strides=1, dilation_rate=1, activation='relu',  padding='same',data_format='channels_last')(in1)
	stack1E = BatchNormalization()(stack1E)
	
	stack2E = Conv2D(64, (3, 3), strides=1, dilation_rate=2, activation='relu',  padding='same', data_format='channels_last')(stack1E)
	stack2E = BatchNormalization()(stack2E)
	stack2E = MaxPooling2D(pool_size=(2, 2))(stack2E)

	stack3E = Conv2D(64, (3, 3), strides=2, dilation_rate=1,activation='relu',  padding='same', data_format='channels_last')(stack2E)
	stack3E = BatchNormalization()(stack3E)

	stack4E = Conv2D(128, (3, 3), strides=1, dilation_rate=2, activation='relu',  padding='same', data_format='channels_last')(stack3E)
	stack4E = BatchNormalization()(stack4E)
	stack4E = MaxPooling2D(pool_size=(2, 2))(stack4E)

	stack5E = Conv2D(256, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack4E)
	stack5E = BatchNormalization()(stack5E)
	stack5E = MaxPooling2D(pool_size=(2, 2))(stack5E)

	#decoder
	stack5D = Conv2D(256, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack5E)
	stack5D = BatchNormalization()(stack5D)
	stack5D = concatenate([stack5D,stack5E])
	stack5D = SpatialDropout2D(0.50)(stack5D)

	stack4D = Conv2DTranspose(128, (2, 2), strides=2, activation='relu',  padding='same',data_format='channels_last')(stack5D)
	stack4D = concatenate([stack4D,stack4E])
	stack4D = Conv2D(128, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack4D)
	stack4D = BatchNormalization()(stack4D)
	stack4D = SpatialDropout2D(0.25)(stack4D)
	
	stack3D = Conv2DTranspose(64, (2, 2), strides=2, activation='relu',  padding='same',data_format='channels_last')(stack4D)
	stack3D = concatenate([stack3D,stack3E])
	stack3D = Conv2D(64, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack3D)
	stack3D = BatchNormalization()(stack3D)
	stack3D = SpatialDropout2D(0.125)(stack3D)

	stack2D = Conv2DTranspose(64, (2, 2), strides=2, activation='relu',  padding='same',data_format='channels_last')(stack3D)
	stack2D = Conv2D(64, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack2D)
	stack2D = BatchNormalization()(stack2D)

	stack1D = Conv2DTranspose(64, (2, 2), strides=2, activation='relu',  padding='same',data_format='channels_last')(stack2D)
	stack1D = Conv2D(64, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack1D)
	stack1D = BatchNormalization()(stack1D)

	stack1D = Conv2D(32, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack1D)
	stack1D = BatchNormalization()(stack1D)

	stack1D = Conv2D(1, (1, 1), activation='sigmoid', padding='same',data_format='channels_last')(stack1D)
	# stack1D = Cropping2D(cropping=1)(stack1D)
	
	return Model(inputs=in1, outputs=stack1D)

@addModel
def maskNet002c():

	''' Change Log :  
	based on maskNet002c;
	changed encoder to aggregrate features from conv by different dialation rates
	changed order of layers, dropout after concat
	'''

	def aggBlock(stackIn, nFilters, dialation_rates=[1,2,4]):
		s = []
		for d in dialation_rates:
			s0 = Conv2D(nFilters, (3, 3), strides=1, dilation_rate=d, activation='relu',  padding='same',data_format='channels_last')(stackIn)
			s0 = BatchNormalization()(s0)
			s.append(s0)

		s1 = concatenate(s)
		return s1

	in1 = Input((224, 224, 1))

	#encoder
	stack1E = aggBlock(in1, 32, dialation_rates=[1,2,4])
	stack1E = Conv2D(64, (3, 3), strides=1, dilation_rate=1, activation='relu',  padding='same', data_format='channels_last')(stack1E)
	stack1E = BatchNormalization()(stack1E)
	
	stack2E = MaxPooling2D(pool_size=(2, 2))(stack1E)
	stack2E = aggBlock(stack2E, 32, dialation_rates=[1,2,4,8])
	stack2E = Conv2D(128, (3, 3), strides=1, dilation_rate=1,activation='relu',  padding='same', data_format='channels_last')(stack2E)
	stack2E = BatchNormalization()(stack2E)
	stack2E = Dropout(0.25)(stack2E)

	stack3E = MaxPooling2D(pool_size=(2, 2))(stack2E)
	stack3E = aggBlock(stack3E, 64, dialation_rates=[1,2,4,8])
	stack3E = Conv2D(128, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack3E)
	stack3E = BatchNormalization()(stack3E)
	
	stack4E = MaxPooling2D(pool_size=(2, 2))(stack3E)
	stack4E = aggBlock(stack4E, 64, dialation_rates=[1,2,4,8])
	stack4E = Conv2D(256, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack4E)
	stack4E = BatchNormalization()(stack4E)

	stack5E = MaxPooling2D(pool_size=(2, 2))(stack4E)
	stack5E = aggBlock(stack5E, 64, dialation_rates=[1,2,4,8])
	stack5E = Conv2D(256, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack5E)
	stack5E = BatchNormalization()(stack5E)
	stack5E = Dropout(0.25)(stack5E)

	#decoder
	stack5D = Conv2D(256, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack5E)
	stack5D = BatchNormalization()(stack5D)
	stack5D = SpatialDropout2D(0.25)(stack5D)

	stack4D = Conv2DTranspose(128, (2, 2), strides=2, activation='relu',  padding='same',data_format='channels_last')(stack5D)
	stack4D = concatenate([stack4D,stack4E])
	stack4D = SpatialDropout2D(0.25)(stack4D)
	stack4D = Conv2D(128, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack4D)
	stack4D = BatchNormalization()(stack4D)
	
	stack3D = Conv2DTranspose(64, (2, 2), strides=2, activation='relu',  padding='same',data_format='channels_last')(stack4D)
	stack3D = concatenate([stack3D,stack3E])
	stack3D = SpatialDropout2D(0.125)(stack3D)
	stack3D = Conv2D(64, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack3D)
	stack3D = BatchNormalization()(stack3D)

	stack2D = Conv2DTranspose(64, (2, 2), strides=2, activation='relu',  padding='same',data_format='channels_last')(stack3D)
	stack2D = Conv2D(64, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack2D)
	stack2D = BatchNormalization()(stack2D)

	stack1D = Conv2DTranspose(64, (2, 2), strides=2, activation='relu',  padding='same',data_format='channels_last')(stack2D)
	stack1D = Conv2D(64, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack1D)
	stack1D = BatchNormalization()(stack1D)

	stack1D = Conv2D(32, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack1D)
	stack1D = BatchNormalization()(stack1D)

	stack1D = Conv2D(1, (1, 1), activation='sigmoid', padding='same',data_format='channels_last')(stack1D)
	
	return Model(inputs=in1, outputs=stack1D)

@addModel
def maskNet002d():

	''' Change Log :  
	based on maskNet002c;
	changed encoder to aggregrate features from conv by different dialation rates 1,2,4
	changed order of layers, dropout after concat
	increased no. of filters, last conv 3x3 instead of 1x1
	'''

	def aggBlock(stackIn, nFilters, dialation_rates=[1,2,4]):
		s = []
		for d in dialation_rates:
			s0 = Conv2D(nFilters, (3, 3), strides=1, dilation_rate=d, activation='relu',  padding='same',data_format='channels_last')(stackIn)
			s0 = BatchNormalization()(s0)
			s.append(s0)

		s1 = concatenate(s)
		return s1

	in1 = Input((224, 224, 1))

	#encoder
	stack1E = aggBlock(in1, 64, dialation_rates=[1,2,4])
	stack1E = Conv2D(64, (3, 3), strides=1, dilation_rate=1, activation='relu',  padding='same', data_format='channels_last')(stack1E)
	stack1E = BatchNormalization()(stack1E)
	
	stack2E = MaxPooling2D(pool_size=(2, 2))(stack1E)
	stack2E = aggBlock(stack2E, 128, dialation_rates=[1,2,4])
	stack2E = Conv2D(128, (3, 3), strides=1, dilation_rate=1,activation='relu',  padding='same', data_format='channels_last')(stack2E)
	stack2E = BatchNormalization()(stack2E)
	stack2E = Dropout(0.25)(stack2E)

	stack3E = MaxPooling2D(pool_size=(2, 2))(stack2E)
	stack3E = aggBlock(stack3E, 128, dialation_rates=[1,2,4])
	stack3E = Conv2D(128, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack3E)
	stack3E = BatchNormalization()(stack3E)
	
	stack4E = MaxPooling2D(pool_size=(2, 2))(stack3E)
	stack4E = aggBlock(stack4E, 256, dialation_rates=[1,2,4])
	stack4E = Conv2D(256, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack4E)
	stack4E = BatchNormalization()(stack4E)
	stack4E = Dropout(0.25)(stack4E)

	stack5E = MaxPooling2D(pool_size=(2, 2))(stack4E)
	stack5E = aggBlock(stack5E, 256, dialation_rates=[1,2,4])
	stack5E = Conv2D(256, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack5E)
	stack5E = BatchNormalization()(stack5E)
	stack5E = Dropout(0.25)(stack5E)

	#decoder
	stack5D = Conv2D(256, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack5E)
	stack5D = BatchNormalization()(stack5D)
	stack5D = SpatialDropout2D(0.25)(stack5D)

	stack4D = Conv2DTranspose(128, (2, 2), strides=2, activation='relu',  padding='same',data_format='channels_last')(stack5D)
	stack4D = concatenate([stack4D,stack4E])
	stack4D = SpatialDropout2D(0.25)(stack4D)
	stack4D = Conv2D(128, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack4D)
	stack4D = BatchNormalization()(stack4D)
	
	stack3D = Conv2DTranspose(64, (2, 2), strides=2, activation='relu',  padding='same',data_format='channels_last')(stack4D)
	stack3D = concatenate([stack3D,stack3E])
	stack3D = SpatialDropout2D(0.125)(stack3D)
	stack3D = Conv2D(64, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack3D)
	stack3D = BatchNormalization()(stack3D)

	stack2D = Conv2DTranspose(64, (2, 2), strides=2, activation='relu',  padding='same',data_format='channels_last')(stack3D)
	stack2D = Conv2D(64, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack2D)
	stack2D = BatchNormalization()(stack2D)

	stack1D = Conv2DTranspose(64, (2, 2), strides=2, activation='relu',  padding='same',data_format='channels_last')(stack2D)
	stack1D = Conv2D(64, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack1D)
	stack1D = BatchNormalization()(stack1D)

	stack1D = Conv2D(32, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack1D)
	stack1D = BatchNormalization()(stack1D)
	stack1D = Conv2D(16, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack1D)
	stack1D = BatchNormalization()(stack1D)
	stack1D = Conv2D(1, (3, 3), activation='sigmoid', padding='same',data_format='channels_last')(stack1D)
	
	return Model(inputs=in1, outputs=stack1D)

@addModel
def maskNet002e():

	''' Change Log :  
	based on maskNet002d;
	moved batch norm from each branch of aggregator to after concatenation
	first conv after first aggregator has stride 2 instead of 1
	changed encoder to aggregrate features from conv by different dialation rates 1,2
	increase no. of filters, dropout in last encoder stack to 512, 0.5
	added bottle nect to top of decoder
	changed conv2dtranspose kernel to 3x3, stride 2 
	removed batch norm from final 2 convs. last conv 1x1 kernel
	'''

	def aggBlock(stackIn, nFilters, dialation_rates=[1,2]):
		s = []
		for d in dialation_rates:
			s0 = Conv2D(nFilters, (3, 3), strides=1, dilation_rate=d, activation='relu',  padding='same',data_format='channels_last')(stackIn)
			s.append(s0)

		s1 = concatenate(s)
		s1 = BatchNormalization()(s1)
		return s1

	in1 = Input((224, 224, 1))

	#encoder
	stack1E = aggBlock(in1, 64, dialation_rates=[1,2,4])
	stack1E = Conv2D(128, (3, 3), strides=2, dilation_rate=1, activation='relu',  padding='same', data_format='channels_last')(stack1E)
	stack1E = BatchNormalization()(stack1E)

	stack2E = aggBlock(stack1E, 128, dialation_rates=[1,2])
	stack2E = Conv2D(128, (3, 3), strides=1, dilation_rate=1,activation='relu',  padding='same', data_format='channels_last')(stack2E)
	stack2E = BatchNormalization()(stack2E)

	stack3E = MaxPooling2D(pool_size=(2, 2))(stack2E)
	stack3E = aggBlock(stack3E, 128, dialation_rates=[1,2])
	stack3E = Conv2D(128, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack3E)
	stack3E = BatchNormalization()(stack3E)
	
	stack4E = MaxPooling2D(pool_size=(2, 2))(stack3E)
	stack4E = aggBlock(stack4E, 256, dialation_rates=[1,2])
	stack4E = Conv2D(256, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack4E)
	stack4E = BatchNormalization()(stack4E)
	stack4E = Dropout(0.25)(stack4E)

	stack5E = MaxPooling2D(pool_size=(2, 2))(stack4E)
	stack5E = aggBlock(stack5E, 256, dialation_rates=[1,2])
	stack5E = Conv2D(256, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack5E)
	stack5E = BatchNormalization()(stack5E)
	stack5E = Dropout(0.5)(stack5E)

	#decoder
	stack5D = Conv2D(64, (1, 1), activation='relu',  padding='same',data_format='channels_last')(stack5E)
	stack5D = BatchNormalization()(stack5D)
	stack5D = Conv2D(256, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack5D)
	stack5D = BatchNormalization()(stack5D)

	stack4D = Conv2DTranspose(128, (3, 3), strides=2, activation='relu',  padding='same',data_format='channels_last')(stack5D)
	stack4D = concatenate([stack4D,stack4E])
	stack4D = SpatialDropout2D(0.25)(stack4D)
	stack4D = Conv2D(128, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack4D)
	stack4D = BatchNormalization()(stack4D)
	
	stack3D = Conv2DTranspose(64, (3, 3), strides=2, activation='relu',  padding='same',data_format='channels_last')(stack4D)
	stack3D = concatenate([stack3D,stack3E])
	stack4D = SpatialDropout2D(0.125)(stack4D)
	stack3D = Conv2D(64, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack3D)
	stack3D = BatchNormalization()(stack3D)

	stack2D = Conv2DTranspose(64, (3, 3), strides=2, activation='relu',  padding='same',data_format='channels_last')(stack3D)
	stack2D = Conv2D(64, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack2D)
	stack2D = BatchNormalization()(stack2D)

	stack1D = Conv2DTranspose(64, (3, 3), strides=2, activation='relu',  padding='same',data_format='channels_last')(stack2D)
	stack1D = Conv2D(64, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack1D)
	stack1D = BatchNormalization()(stack1D)

	stack1D = Conv2D(16, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack1D)
	stack1D = Conv2D(1, (1, 1), activation='sigmoid', padding='same', data_format='channels_last')(stack1D)
	
	return Model(inputs=in1, outputs=stack1D)

@addModel
def maskNet003():

	def encoder(inputSlice):
		#encoder
		stack1E = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same',data_format='channels_last')(inputSlice)
		stack1E = BatchNormalization()(stack1E)
		
		stack2E = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same',data_format='channels_last')(stack1E)
		stack2E = BatchNormalization()(stack2E)
		stack2E = MaxPooling2D(pool_size=(2, 2))(stack2E)

		stack3E = Conv2D(64, (3, 3), strides=2, activation='relu', padding='same',data_format='channels_last')(stack2E)
		stack3E = BatchNormalization()(stack3E)

		stack4E = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same',data_format='channels_last')(stack3E)
		stack4E = BatchNormalization()(stack4E)
		stack4E = MaxPooling2D(pool_size=(2, 2))(stack4E)

		stack5E = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(stack4E)
		stack5E = BatchNormalization()(stack5E)
		stack5E = MaxPooling2D(pool_size=(2, 2))(stack5E)

		return [stack5E,stack4E,stack3E]

	def halfDecoder(encStates):
		stack5E, stack4E, stack3E = encStates

		# stack5D = Dropout(0.5)(stack5E)
		stack5D = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(stack5E)
		stack5D = BatchNormalization()(stack5D)
		stack5D = SpatialDropout2D(0.25)(stack5D)

		stack4D = UpSampling2D((2, 2))(stack5D)
		stack4D = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(stack4D)
		stack4D = BatchNormalization()(stack4D)
		stack4D = concatenate([stack4D,stack4E])
		stack4D = SpatialDropout2D(0.25)(stack4D)
		
		stack3D = UpSampling2D((2, 2))(stack4D)
		stack3D = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(stack3D)
		stack3D = BatchNormalization()(stack3D)
		stack3D = concatenate([stack3D,stack3E])
		stack3D = SpatialDropout2D(0.25)(stack3D)
		stack3D = UpSampling2D((2, 2))(stack3D)

		return stack3D    
	
	sliceShape = (350, 350, 3)
	
	inputSlices = Input(sliceShape)

	# splitting top, center and bottom slices
	inTop = Lambda(lambda x : x[:,:,:,0])(inputSlices)
	inCen = Lambda(lambda x : x[:,:,:,1])(inputSlices)
	inBot = Lambda(lambda x : x[:,:,:,2])(inputSlices)
 
	inTop = Reshape( (sliceShape[0],sliceShape[1],1))(inTop) 
	inCen = Reshape( (sliceShape[0],sliceShape[1],1))(inCen)
	inBot = Reshape( (sliceShape[0],sliceShape[1],1))(inBot)

	# encoding
	encTop = encoder(inTop)
	encCen = encoder(inCen)
	encBot = encoder(inBot)

	# half decoding
	decTop = halfDecoder(encTop)
	decCen = halfDecoder(encCen)
	decBot = halfDecoder(encBot)

	# combining
	combinedStacks = concatenate([decTop, decCen, decBot])
	combinedStacks = SpatialDropout2D(0.33)(combinedStacks)

	stack1 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(combinedStacks)
	stack1 = BatchNormalization()(stack1)
	stack1 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last')(combinedStacks)
	stack1 = BatchNormalization()(stack1)

	stack2 = UpSampling2D((2, 2))(stack1)

	stack2 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(stack2)
	stack2 = BatchNormalization()(stack2) 
	stack2 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(stack2)
	stack2 = BatchNormalization()(stack2)

	out = Conv2D(1, (1, 1), activation='sigmoid', padding='same',data_format='channels_last')(stack2)
	out = Cropping2D((1,1))(out)

	return Model(inputs=inputSlices, outputs=out)

@addModel
def maskNet004():

	''' Change Log :  
	Based on masknet002a, but changed to ouput
	region props besides mask
	'''
	in1 = Input((350, 350, 1))

	#encoder
	stack1E = Conv2D(32, (3, 3), strides=1, dilation_rate=1, activation='relu',  padding='same',data_format='channels_last')(in1)
	stack1E = BatchNormalization()(stack1E)
	
	stack2E = Conv2D(64, (3, 3), strides=1, dilation_rate=2, activation='relu',  padding='same', data_format='channels_last')(stack1E)
	stack2E = BatchNormalization()(stack2E)
	stack2E = MaxPooling2D(pool_size=(2, 2))(stack2E)

	stack3E = Conv2D(64, (3, 3), strides=2, dilation_rate=1,activation='relu',  padding='same', data_format='channels_last')(stack2E)
	stack3E = BatchNormalization()(stack3E)

	stack4E = Conv2D(128, (3, 3), strides=1, dilation_rate=2, activation='relu',  padding='same', data_format='channels_last')(stack3E)
	stack4E = BatchNormalization()(stack4E)
	stack4E = MaxPooling2D(pool_size=(2, 2))(stack4E)

	stack5E = Conv2D(256, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack4E)
	stack5E = BatchNormalization()(stack5E)
	stack5E = MaxPooling2D(pool_size=(2, 2))(stack5E)

	stack6E = Conv2D(128, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack5E)
	stack6E = BatchNormalization()(stack6E)
	stack6E = MaxPooling2D(pool_size=(2, 2))(stack6E)

	stack7E = Conv2D(64, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack6E)
	stack7E = BatchNormalization()(stack7E)
	stack7E = MaxPooling2D(pool_size=(2, 2))(stack7E)

	stack8E = Conv2D(256, (1, 1), activation='relu',  padding='same',data_format='channels_last')(stack7E)
	stack8E = BatchNormalization()(stack8E)
	stack8E = MaxPooling2D(pool_size=(2, 2))(stack8E)
	
	stackProps = Flatten(name='Properties_Branch')(stack8E)
	stackProps = Dense(512, activation='relu')(stackProps)
	stackProps = BatchNormalization()(stackProps)
	stackProps = Dropout(0.5)(stackProps)

	stackProps = Dense(256, activation='relu')(stackProps)
	stackProps = BatchNormalization()(stackProps)
	stackProps = Dropout(0.25)(stackProps)

	stackProps = Dense(128, activation='relu')(stackProps)
	stackProps = BatchNormalization()(stackProps)

	outCentroid = Dense(2, activation='sigmoid',name='centroid')(stackProps)

	#decoder
	stack5D = Conv2D(256, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack5E)
	stack5D = BatchNormalization()(stack5D)
	stack5D = concatenate([stack5D,stack5E])
	stack5D = SpatialDropout2D(0.50)(stack5D)

	stack4D = UpSampling2D((2, 2))(stack5D)
	stack4D = Conv2D(128, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack4D)
	stack4D = BatchNormalization()(stack4D)
	stack4D = concatenate([stack4D,stack4E])
	stack4D = SpatialDropout2D(0.25)(stack4D)

	stack3D = UpSampling2D((2, 2))(stack4D)
	stack3D = Conv2D(64, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack3D)
	stack3D = BatchNormalization()(stack3D)
	stack3D = concatenate([stack3D,stack3E])
	stack3D = SpatialDropout2D(0.25)(stack3D)

	stack2D = UpSampling2D((2, 2))(stack3D)
	stack2D = Conv2D(64, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack2D)
	stack2D = BatchNormalization()(stack2D)

	stack1D = UpSampling2D((2, 2))(stack2D)
	stack1D = Conv2D(64, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack1D)
	stack1D = BatchNormalization()(stack1D)

	stack1D = Conv2D(32, (3, 3), activation='relu',  padding='same',data_format='channels_last')(stack1D)
	stack1D = BatchNormalization()(stack1D)

	stack1D = Conv2D(1, (1, 1), activation='sigmoid', padding='same',data_format='channels_last')(stack1D)
	outMask = Cropping2D(cropping=1,name='mask')(stack1D)
	
	return Model(inputs=in1, outputs=outMask)

@addModel
def ruNet001():

	def Conv2D_BN(stackIn, nFilters, kernel_size,strides=1, dilation_rate=1, activation='relu', padding='same'):
		''' Performs 2D convolution followed by Batch Normalization '''
		stackOut = Conv2D(nFilters, kernel_size, strides=strides, dilation_rate=dilation_rate, activation=activation,padding=padding,data_format='channels_last')(stackIn)
		stackOut = BatchNormalization()(stackOut)
		return stackOut

	def convBlock(stackIn, nFilters, filterSize, pool=False, upsample=False, kernelInitializer='he_normal'):

		conv1 = Conv2D(nFilters, filterSize, strides=1,padding='same', kernel_initializer=kernelInitializer, data_format='channels_last')
		stack1 = conv1(stackIn)
		stack2 = BatchNormalization()(stack1)
		stack3 = LeakyReLU()(stack2)

		conv2 = Conv2D(nFilters, filterSize, strides=1, padding='same', kernel_initializer=kernelInitializer, data_format='channels_last')
		stack4 = conv2(stack3)
		stack5 = add([stack1, stack4])
		stack6 = BatchNormalization()(stack5)
		stack7 = LeakyReLU()(stack6)

		conv3 = Conv2D(nFilters, filterSize, strides=1, padding='same',weights=conv2.get_weights(), data_format='channels_last')
		stack8 = conv3(stack7)
		stack9 = add([stack1, stack8])
		stack10 = BatchNormalization()(stack9)
		stack11 = LeakyReLU()(stack10)

		if pool:
			stack11 = MaxPooling2D((2, 2), padding='same')(stack11)
		elif upsample:
			stack11 = Conv2DTranspose(nFilters, (2,2), strides=2, padding='same',kernel_initializer=kernelInitializer)(stack11)
			stack11 = BatchNormalization()(stack11)
			stack11 = LeakyReLU()(stack11)

		return stack11

	def encoder(stackIn):

		s1_1 = Conv2D_BN(stackIn, 64, kernel_size=(3,3), strides=1, dilation_rate=1, activation='relu',padding='same')
		s1_2 = Conv2D_BN(stackIn, 64, kernel_size=(3,3), strides=1, dilation_rate=2, activation='relu',padding='same')
		s1_4 = Conv2D_BN(stackIn, 64, kernel_size=(3,3), strides=1, dilation_rate=4, activation='relu',padding='same')
		s1 = concatenate([s1_1, s1_2, s1_4])
		s1 = Conv2D_BN(s1, 64, kernel_size=(3,3), strides=1, activation='relu',padding='same')
		s1 = MaxPooling2D((2,2))(s1)

		s2 = convBlock(s1,64, 3, pool=False)
		s3 = convBlock(s2,128, 3, pool=True)
		s4 = convBlock(s3,64, 3, pool=False)
		s5 = convBlock(s4,128, 3, pool=True)
		s6 = convBlock(s5,64, 3, pool=False)
		s7 = convBlock(s6,128, 3, pool=True)

		return [s1,s2,s3,s4,s5,s6,s7]

	def decoder(enc_states):

		es1,es2,es3,es4,es5,es6,es7 = enc_states

		s1 = Dropout(0.5)(es7)

		s2 = convBlock(s1, 128, 3, upsample=True)	
		s2 = concatenate([s2,es6])
		s2 = convBlock(s2, 64, 3, upsample=False)
		
		s3 = convBlock(s2, 128, 3, upsample=True)
		s3 = concatenate([s3,es4])
		s3 = convBlock(s3, 64, 3, upsample=False)

		s4 = convBlock(s3, 128, 3, upsample=True)
		s4 = concatenate([s4,es2])
		s4 = convBlock(s4, 64, 3, upsample=False)

		s5 = convBlock(s4, 128, 3, upsample=True)
		s5 = convBlock(s5, 64, 3, upsample=False)

		s6 = Conv2D(1, (1,1), activation='sigmoid', padding='same', kernel_initializer='he_normal', data_format='channels_last')(s5)

		return s6

	in1 = Input((224, 224, 1))
	en = encoder(in1)
	de = decoder(en)
	
	return Model(inputs=[in1], outputs=de)

@addModel
def volNet001():

	def encoder(stackIn):

		s1 = Conv3D(64, (3,3,3), strides=(1,1,1), padding='same', data_format='channels_last')(stackIn)
		s1a = ReLU()(s1)
		s1b = BatchNormalization()(s1a)

		s2 = Conv3D(64, (3,3,3), strides=(1,1,1), padding='same', data_format='channels_last')(s1b)
		s2 = add([s1,s2])
		s2a = ReLU()(s2)
		s2b = BatchNormalization()(s2a)

		s3 = MaxPooling3D((2,2,1))(s2b)
		s3 = Conv3D(128, (3,3,3), strides=(1,1,1), padding='same', data_format='channels_last')(s3)
		s3a = ReLU()(s3)
		s3b = BatchNormalization()(s3a)

		s4 = Conv3D(128, (3,3,3), strides=(1,1,1), padding='same', data_format='channels_last')(s3b)
		s4 = add([s3,s4])
		s4a = ReLU()(s4)
		s4b = BatchNormalization()(s4a)

		s5 = MaxPooling3D((2,2,1))(s4b)
		s5 = Conv3D(128, (3,3,3), strides=(1,1,1), padding='same', data_format='channels_last')(s5)
		s5a = ReLU()(s5)
		s5b = BatchNormalization()(s5a)

		s6 = Conv3D(128, (3,3,3), strides=(1,1,1), padding='same', data_format='channels_last')(s5b)
		s6 = add([s5,s6])
		s6a = ReLU()(s6)
		s6b = BatchNormalization()(s6a)
		
		s7 = MaxPooling3D((2,2,1))(s6b)
		s7 = Conv3D(128, (3,3,3), strides=(1,1,1), padding='same', data_format='channels_last')(s7)
		s7a = ReLU()(s7)
		s7b = BatchNormalization()(s7a)

		s8 = Conv3D(128, (3,3,3), strides=(1,1,1), padding='same', data_format='channels_last')(s7b)
		s8 = add([s7,s8])
		s8a = ReLU()(s8)
		s8b = BatchNormalization()(s8a)

		return s2b,s4b,s6b,s8b

	def decoder(enc):
		#decoder
		s2, s4, s6, s8 = enc

		stack5D = Conv3D(128, (3,3,3), strides=(1,1,1), activation='relu',  padding='same',data_format='channels_last')(s8)
		stack5D = BatchNormalization()(stack5D)
		stack5D = concatenate([stack5D, s8])
		stack5D = SpatialDropout3D(0.25)(stack5D)

		stack4D = Conv3D(128, (3,3,3), strides=(1,1,1), activation='relu',  padding='same',data_format='channels_last')(stack5D)
		stack4D = BatchNormalization()(stack4D)
		stack4D = UpSampling3D((2,2,1))(stack4D)
		stack4D = Conv3D(128, (3,3,3), strides=(1,1,1), activation='relu',  padding='same',data_format='channels_last')(stack4D)
		stack4D = BatchNormalization()(stack4D)
		stack4D = concatenate([stack4D, s6])
		stack4D = SpatialDropout3D(0.25)(stack4D)

		stack3D = Conv3D(64, (3,3,3), strides=(1,1,1), activation='relu',  padding='same',data_format='channels_last')(stack4D)
		stack3D = BatchNormalization()(stack3D)
		stack3D = Conv3D(64, (3,3,3), strides=(1,1,1), activation='relu',  padding='same',data_format='channels_last')(stack3D)
		stack3D = BatchNormalization()(stack3D)
		stack3D = UpSampling3D((2,2,1))(stack3D)
		stack3D = Conv3D(64, (3,3,3), strides=(1,1,1), activation='relu',  padding='same',data_format='channels_last')(stack3D)
		stack3D = BatchNormalization()(stack3D)
		stack3D = concatenate([stack3D, s4])
		stack3D = SpatialDropout3D(0.125)(stack3D)

		stack2D = Conv3D(64, (3,3,3), strides=(1,1,1), activation='relu',  padding='same',data_format='channels_last')(stack3D)
		stack2D = BatchNormalization()(stack2D)
		stack2D = Conv3D(64, (3,3,3), strides=(1,1,1), activation='relu',  padding='same',data_format='channels_last')(stack2D)
		stack2D = BatchNormalization()(stack2D)
		stack2D = UpSampling3D((2,2,1))(stack2D)
		stack2D = Conv3D(64, (3,3,3), strides=(1,1,1), activation='relu',  padding='same',data_format='channels_last')(stack2D)
		stack2D = BatchNormalization()(stack2D)
		stack2D = concatenate([stack2D, s2])
		stack2D = SpatialDropout3D(0.125)(stack2D)

		stack1D = Conv3D(64, (3,3,3), strides=(1,1,1), activation='relu',  padding='same',data_format='channels_last')(stack2D)
		stack1D = BatchNormalization()(stack1D)
		stack1D = Conv3D(64, (3,3,3), strides=(1,1,1), activation='relu',  padding='same',data_format='channels_last')(stack1D)
		stack1D = BatchNormalization()(stack1D)
		stack1D = Conv3D(32, (3,3,3), strides=(1,1,1), activation='relu',  padding='same',data_format='channels_last')(stack1D)
		stack1D = BatchNormalization()(stack1D)
		stack1D = Conv3D(1, (3,3,3), strides=(1,1,1), activation='sigmoid',  padding='same',data_format='channels_last')(stack1D)

		return stack1D

	in1 = Input((176,176,9,1))
	enc = encoder(in1)
	dec = decoder(enc)

	return Model(inputs=in1, outputs=dec)

@addModel
def volNet003():

	def encoder(stackIn):

		s1 = Conv3D(32, (3,3,3), strides=(1,1,1), padding='same', data_format='channels_last')(stackIn)
		s1a = ReLU()(s1)
		s1b = BatchNormalization()(s1a)

		s2 = Conv3D(64, (3,3,3), strides=(1,1,1), padding='same', data_format='channels_last')(s1b)
		s2a = ReLU()(s2)
		s2b = BatchNormalization()(s2a)

		s3 = MaxPooling3D((1,2,2))(s2b)
		s3 = Conv3D(64, (3,3,3), strides=(1,1,1), padding='same', data_format='channels_last')(s3)
		s3a = ReLU()(s3)
		s3b = BatchNormalization()(s3a)

		s4 = Conv3D(64, (3,3,3), strides=(1,1,1), padding='same', data_format='channels_last')(s3b)
		s4a = ReLU()(s4)
		s4b = BatchNormalization()(s4a)

		s5 = MaxPooling3D((1,2,2))(s4b)
		s5 = Conv3D(64, (3,3,3), strides=(1,1,1), padding='same', data_format='channels_last')(s5)
		s5a = ReLU()(s5)
		s5b = BatchNormalization()(s5a)

		s6 = Conv3D(64, (3,3,3), strides=(1,1,1), padding='same', data_format='channels_last')(s5b)
		s6a = ReLU()(s6)
		s6b = BatchNormalization()(s6a)
		
		s7 = MaxPooling3D((1,2,2))(s6b)
		s7 = Conv3D(128, (3,3,3), strides=(1,1,1), padding='same', data_format='channels_last')(s7)
		s7a = ReLU()(s7)
		s7b = BatchNormalization()(s7a)

		return s2b,s4b,s6b,s7b

	def decoder(enc):
		#decoder
		s2, s4, s6, s7 = enc

		stack5D = Conv3D(128, (3,3,3), strides=(1,1,1), activation='relu',  padding='same',data_format='channels_last')(s7)
		stack5D = BatchNormalization()(stack5D)
		stack5D = concatenate([stack5D, s7])
		stack5D = SpatialDropout3D(0.25)(stack5D)

		stack4D = Conv3D(64, (3,3,3), strides=(1,1,1), activation='relu',  padding='same',data_format='channels_last')(stack5D)
		stack4D = BatchNormalization()(stack4D)
		stack4D = UpSampling3D((1,2,2))(stack4D)
		stack4D = Conv3D(64, (3,3,3), strides=(1,1,1), activation='relu',  padding='same',data_format='channels_last')(stack4D)
		stack4D = BatchNormalization()(stack4D)
		stack4D = concatenate([stack4D, s6])
		stack4D = SpatialDropout3D(0.25)(stack4D)

		stack3D = Conv3D(64, (3,3,3), strides=(1,1,1), activation='relu',  padding='same',data_format='channels_last')(stack4D)
		stack3D = BatchNormalization()(stack3D)
		stack3D = Conv3D(64, (3,3,3), strides=(1,1,1), activation='relu',  padding='same',data_format='channels_last')(stack3D)
		stack3D = BatchNormalization()(stack3D)
		stack3D = UpSampling3D((1,2,2))(stack3D)
		stack3D = Conv3D(64, (3,3,3), strides=(1,1,1), activation='relu',  padding='same',data_format='channels_last')(stack3D)
		stack3D = BatchNormalization()(stack3D)
		stack3D = concatenate([stack3D, s4])
		stack3D = SpatialDropout3D(0.125)(stack3D)

		stack2D = Conv3D(64, (3,3,3), strides=(1,1,1), activation='relu',  padding='same',data_format='channels_last')(stack3D)
		stack2D = BatchNormalization()(stack2D)
		stack2D = Conv3D(64, (3,3,3), strides=(1,1,1), activation='relu',  padding='same',data_format='channels_last')(stack2D)
		stack2D = BatchNormalization()(stack2D)
		stack2D = UpSampling3D((1,2,2))(stack2D)
		stack2D = Conv3D(64, (3,3,3), strides=(1,1,1), activation='relu',  padding='same',data_format='channels_last')(stack2D)
		stack2D = BatchNormalization()(stack2D)
		stack2D = concatenate([stack2D, s2])
		stack2D = SpatialDropout3D(0.125)(stack2D)

		stack1D = Conv3D(64, (3,3,3), strides=(1,1,1), activation='relu',  padding='same',data_format='channels_last')(stack2D)
		stack1D = BatchNormalization()(stack1D)
		stack1D = Conv3D(32, (3,3,3), strides=(1,1,1), activation='relu',  padding='same',data_format='channels_last')(stack1D)
		stack1D = BatchNormalization()(stack1D)
		stack1D = Conv3D(1, (3,3,3), strides=(1,1,1), activation='sigmoid',  padding='same',data_format='channels_last')(stack1D)

		return stack1D

	in1 = Input((9,224,224,1))
	enc = encoder(in1)
	dec = decoder(enc)

	return Model(inputs=in1, outputs=dec)

@addModel
def LungNet001():

	def bn_block(x):
		return add([x, BatchNormalization()(x)])

	def conv_block(x, nb_filter, filter_size, atrous_rate=(1, 1)):
		x = Conv2D(nb_filter, filter_size, dilation_rate=atrous_rate, kernel_initializer='he_normal', padding='same')(x)
		x = bn_block(x)
		x = ReLU()(x)
		return x


	atrous_rates = [(1,1), (1,1), (2,2), (3,3), (5,5), (8,8), (13,13), (21,21), (34,34), (55,55)] 
		
	i = Input((224, 224, 1))
	
	t = bn_block(i)
	feat_list = [t]
	for layer in range(0, len(atrous_rates)):
		t = conv_block(t, 32, (3,3), atrous_rate=atrous_rates[layer])
		feat_list.append(t)
		
	t = concatenate(feat_list)
	t = Dropout(0.5)(t)

	t = conv_block(t, 128, (1,1))
	t = conv_block(t, 32, (1,1))

	t = Conv2D(3, (1, 1), activation='sigmoid', kernel_initializer='he_normal', name='mask')(t)

	return Model(inputs=[i], outputs=[t])

@addModel
def LungNet002():

	''' LungNet001 but number of filters = 64 '''

	def bn_block(x):
		return add([x, BatchNormalization()(x)])

	def conv_block(x, nb_filter, filter_size, atrous_rate=(1, 1)):
		x = Conv2D(nb_filter, filter_size, dilation_rate=atrous_rate, kernel_initializer='he_normal', padding='same')(x)
		x = bn_block(x)
		x = ReLU()(x)
		return x


	atrous_rates = [(1,1), (1,1), (2,2), (3,3), (5,5), (8,8), (13,13), (21,21), (34,34), (55,55)] 
		
	i = Input((224, 224, 1))
	
	t = bn_block(i)
	feat_list = [t]
	for layer in range(0, len(atrous_rates)):
		t = conv_block(t, 64, (3,3), atrous_rate=atrous_rates[layer])
		feat_list.append(t)
		
	t = concatenate(feat_list)
	t = Dropout(0.5)(t)

	t = conv_block(t, 128, (1,1))
	t = conv_block(t, 32, (1,1))

	t = Conv2D(3, (1, 1), activation='sigmoid', kernel_initializer='he_normal',name='mask')(t)

	return Model(inputs=[i], outputs=[t])

@addModel
def LungNet004():

	''' LungNet001 but no feature concatenation '''

	def bn_block(x):
		return add([x, BatchNormalization()(x)])

	def conv_block(x, nb_filter, filter_size, atrous_rate=(1, 1)):
		x = Conv2D(nb_filter, filter_size, dilation_rate=atrous_rate, kernel_initializer='he_normal', padding='same')(x)
		x = bn_block(x)
		x = ReLU()(x)
		return x


	atrous_rates = [(1,1), (1,1), (2,2), (3,3), (5,5), (8,8), (13,13), (21,21), (34,34), (55,55)] 
		
	i = Input((224, 224, 1))
	
	t = bn_block(i)
	feat_list = [t]
	for layer in range(0, len(atrous_rates)):
		t = conv_block(t, 32, (3,3), atrous_rate=atrous_rates[layer])
		feat_list.append(t)
		
	# t = concatenate(feat_list)
	t = Dropout(0.5)(t)

	t = conv_block(t, 128, (1,1))
	t = conv_block(t, 32, (1,1))

	t = Conv2D(3, (1, 1), activation='sigmoid', kernel_initializer='he_normal',name='mask')(t)

	return Model(inputs=[i], outputs=[t])

@addModel
def LungNet003():

	''' LungNet001 but variable number of filters '''

	def bn_block(x):
		return add([x, BatchNormalization()(x)])

	def conv_block(x, nb_filter, filter_size, atrous_rate=(1, 1)):
		x = Conv2D(nb_filter, filter_size, dilation_rate=atrous_rate, kernel_initializer='he_normal', padding='same')(x)
		x = bn_block(x)
		x = ReLU()(x)
		return x


	atrous_rates = [(1,1), (1,1), (2,2), (3,3), (5,5), (8,8), (13,13), (21,21), (34,34), (55,55)] 
	filter_number = [ 32, 32, 64, 64, 64, 64, 64, 32, 32, 32]
	i = Input((224, 224, 1))
	
	t = bn_block(i)
	feat_list = [t]
	for layer in range(0, len(atrous_rates)):
		t = conv_block(t, filter_number[layer], (3,3), atrous_rate=atrous_rates[layer])
		feat_list.append(t)
		
	t = concatenate(feat_list)
	t = Dropout(0.5)(t)

	t = conv_block(t, 128, (1,1))
	t = conv_block(t, 32, (1,1))

	t = Conv2D(3, (1, 1), activation='sigmoid', kernel_initializer='he_normal',name='mask')(t)

	return Model(inputs=[i], outputs=[t])

@addModel
def LungNet001a():
	
	'''Change Log :
	added bottleneck after concatentation
	increase no. of filters
	decreased dropout to 0.4
	'''

	def bn_block(x):
		return add([x, BatchNormalization()(x)])

	def conv_block(x, nb_filter, filter_size, atrous_rate=(1, 1)):
		x = Conv2D(nb_filter, filter_size, dilation_rate=atrous_rate, kernel_initializer='he_normal', padding='same')(x)
		x = bn_block(x)
		x = ReLU()(x)
		return x


	atrous_rates = [(1,1), (1,1), (2,2), (3,3), (5,5), (8,8), (13,13), (21,21), (34,34), (55,55)] 
		
	i = Input((224, 224, 1))
	
	t = bn_block(i)
	feat_list = [t]
	for layer in range(0, len(atrous_rates)):
		t = conv_block(t, 32, (3,3), atrous_rate=atrous_rates[layer])
		feat_list.append(t)
		
	t = concatenate(feat_list)
	t = Dropout(0.4)(t)

	t = conv_block(t, 128, (1, 1))
	t = conv_block(t, 64, (1, 1))
	t = conv_block(t, 128, (1, 1))
	t = conv_block(t, 32, (1,1))

	t = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='he_normal')(t)

	return Model(inputs=[i], outputs=[t])

@addModel
def LungNet001b():
	
	''' Change Log:
	Based on Lung001. Added increasing no. of filters
	'''

	def bn_block(x):
		return add([x, BatchNormalization()(x)])

	def conv_block(x, nb_filter, filter_size, atrous_rate=(1, 1)):
		x = Conv2D(nb_filter, filter_size, dilation_rate=atrous_rate, kernel_initializer='he_normal', padding='same')(x)
		x = bn_block(x)
		x = ReLU()(x)
		return x


	atrous_rates = [(1,1), (1,1), (2,2), (3,3), (5,5), (8,8), (13,13), (21,21), (34,34), (55,55)] 
	nFilters = [32,32,64,64,64,96,96,128,128,128]

	i = Input((224, 224, 1))
	
	t = bn_block(i)
	feat_list = [t]
	for layer in range(0, len(atrous_rates)):
		t = conv_block(t, nFilters[layer], (3,3), atrous_rate=atrous_rates[layer])
		feat_list.append(t)
		
	t = concatenate(feat_list)
	t = Dropout(0.5)(t)

	t = conv_block(t, 128, (1,1))
	t = conv_block(t, 32, (1,1))

	t = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='he_normal')(t)

	return Model(inputs=[i], outputs=[t])

@addModel
def LungNet001c():

	'''Change Log :
	based on LungNet001
	modified to output 3 mask channels 
	'''

	def bn_block(x):
		return add([x, BatchNormalization()(x)])

	def conv_block(x, nb_filter, filter_size, atrous_rate=(1, 1)):
		x = Conv2D(nb_filter, filter_size, dilation_rate=atrous_rate, kernel_initializer='he_normal', padding='same')(x)
		x = bn_block(x)
		x = ReLU()(x)
		return x


	atrous_rates = [(1,1), (1,1), (2,2), (3,3), (5,5), (8,8), (13,13), (21,21), (34,34), (55,55)] 
		
	i = Input((224, 224, 1))
	
	t = bn_block(i)
	feat_list = [t]
	for layer in range(0, len(atrous_rates)):
		t = conv_block(t, 32, (3,3), atrous_rate=atrous_rates[layer])
		feat_list.append(t)
		
	t = concatenate(feat_list)
	t = Dropout(0.5)(t)

	t = conv_block(t, 128, (1,1))
	t = conv_block(t, 32, (1,1))

	t = Conv2D(3, (1, 1), activation='sigmoid', kernel_initializer='he_normal')(t)

	return Model(inputs=[i], outputs=[t])

@addModel
def LungNet001d():

	'''Change Log :
	based on LungNet001c
	modified to output binary classification of slice as well (tumor v no tumor) 
	'''
	pass_idx = 0

	def bn_block(x, name=None):
		return add([x, BatchNormalization(name="%s_BN" % name if name is not None else None)(x)], name="%s_AddBN" % name if name is not None else None)

	def conv_block(x, nb_filter, filter_size, atrous_rate=(1, 1), name=None):
		x = Conv2D(nb_filter, filter_size, dilation_rate=atrous_rate, kernel_initializer='he_normal', padding='same', name= "%s_Conv" % name if name is not None else None)(x)
		x = bn_block(x, name = name)
		x = ReLU(name="%s_ReLU" % name if name is not None else None)(x)
		return x
	
	def encoder(inLayer, nFilters, atrousRates, pass_idx=0):

		t = bn_block(inLayer,name="BN_00_Pass_%02d" % pass_idx)
		feat_list = [t]
		
		for layer in range(0, len(atrousRates)):
			t = conv_block(t, nFilters, (3,3), atrous_rate=atrousRates[layer], name="ConvBlock_%02d_Pass_%02d" % (layer,pass_idx))
			feat_list.append(t)
		
		t = concatenate(feat_list, name="AtrousConcat_Pass_%02d" % pass_idx)

		return Model(inputs=[i], outputs=[t])
	
	i = Input((224, 224, 1),name="SliceInput")

	atrous_rates = [(1,1), (1,1), (2,2), (3,3), (5,5), (8,8), (13,13), (21,21), (34,34), (55,55)] 
	pass_idx+=1
	encPass01 = encoder(i, 32, atrous_rates, pass_idx)
	t = Dropout(0.5, name="AtrousDropout_Pass_%02d" % pass_idx)(encPass01.output)
	t = conv_block(t, 128, (1,1),name="Bottleneck_01_Pass_%02d" % pass_idx)
	tm = conv_block(t, 32, (1,1),name="Bottleneck_02_Pass_%02d" % pass_idx)
	masks = Conv2D(3, (1, 1), activation='sigmoid', kernel_initializer='he_normal', name='mask_Pass_%02d' % pass_idx)(tm)

	roiMask = Lambda(lambda x : x[:,:,:,0:1])(masks)
	lungMask = Lambda(lambda x : x[:,:,:,1:2])(masks)
	patientMask = Lambda(lambda x : x[:,:,:,2:3])(masks)
	roiInput = multiply([i,roiMask,lungMask,patientMask])

	atrous_rates = [(1,1), (1,1), (2,2), (3,3), (5,5), (8,8), (13,13), (21,21), (34,34), (55,55)] 
	pass_idx+=1
	encPass02 = encoder(roiInput, 32, atrous_rates, pass_idx)
	encPass02.set_weights(encPass01.get_weights())
	t = Dropout(0.5, name="AtrousDropout_Pass_%02d" % pass_idx)(encPass02.output)
	t = conv_block(t, 128, (1,1), name="Bottleneck_01_Pass_%02d" % pass_idx)
	tm = conv_block(t, 32, (1,1), name="Bottleneck_02_Pass_%02d" % pass_idx)
	masks = Conv2D(3, (1, 1), activation='sigmoid', kernel_initializer='he_normal', name='mask')(tm)

	# tb = MaxPooling2D((2, 2),name="binClass_Pooling_01")(encPass02.output)
	# tb = conv_block(tb, 64, (3,3), name="binClass_ConvAfterPool_01a")
	# tb = conv_block(tb, 64, (3,3), name="binClass_ConvAfterPool_01b")
	# tb = MaxPooling2D((2, 2), name="binClass_Pooling_02")(tb)
	# tb = conv_block(tb, 64, (3,3), name="binClass_ConvAfterPool_02a")
	# tb = conv_block(tb, 64, (3,3), name="binClass_ConvAfterPool_02b")
	# tb = MaxPooling2D((2, 2), name="binClass_Pooling_03")(tb)
	# tb = conv_block(tb, 64, (3,3), name="binClass_ConvAfterPool_03a")
	# tb = conv_block(tb, 64, (3,3), name="binClass_ConvAfterPool_03b")
	# tb = MaxPooling2D((2, 2), name="binClass_Pooling_04")(tb)
	# tb = conv_block(tb, 64, (3,3), name="binClass_ConvAfterPool_04a")
	# tb = conv_block(tb, 64, (3,3), name="binClass_ConvAfterPool_04b")
	# tb = MaxPooling2D((2, 2), name="binClass_Pooling_05")(tb)
	# tb = conv_block(tb, 128, (3, 3), name="binClass_ConvAfterPool_05a")
	# tb = conv_block(tb, 128, (3, 3), name="binClass_ConvAfterPool_05b")
	# tb = MaxPooling2D((2, 2), name="binClass_Pooling_06")(tb)

	# tb = Flatten(name="binClass_Flatten")(tb)
	# tb = Dense(64, activation ='relu',name="binClass_Dense_01")(tb)
	# tb = Dense(64, activation ='relu',name="binClass_Dense_02")(tb)
	# tb = Dense(64, activation ='relu',name="binClass_Dense_03")(tb)

	# binClass = Dense(1, activation = 'sigmoid', name='binClass')(tb)

	return Model(inputs=[i], outputs=[masks])

@addModel
def RecursiveLungNet001():

	def bn_block(x):
		return add([x, BatchNormalization()(x)])

	def conv_block(x, nb_filter, filter_size, atrous_rate=(1, 1)):
		x = Conv2D(nb_filter, filter_size, dilation_rate=atrous_rate, kernel_initializer='he_normal', padding='same')(x)
		x = bn_block(x)
		x = ReLU()(x)
		return x

	def recursive_conv_block(k, x, nb_filter, filter_size, atrous_rate=(1, 1)):
		conv1 = Conv2D(nb_filter, filter_size, dilation_rate=atrous_rate, kernel_initializer='he_normal', padding='same')
		x = conv1(x)
		x = bn_block(x)
		x = ReLU()(x)
		
		for itr in range(k):
			conv2 = Conv2D(nb_filter, filter_size, dilation_rate=atrous_rate, kernel_initializer='he_normal', padding='same', weights=conv1.get_weights())
			x = conv2(x)
			x = bn_block(x)
			x = ReLU()(x)
		
		return x

	atrous_rates = [(1,1), (1,1), (2,2), (3,3), (5,5), (8,8), (13,13), (21,21), (34,34), (55,55)] 
	n_recursions = [0,		2, 		2, 		2,		2,		1,		1,		1,		0,		0 ]

	i = Input((224, 224, 1))
	
	t = bn_block(i)
	feat_list = [t]
	for layer in range(0, len(atrous_rates)):
		t = recursive_conv_block(n_recursions[layer],t, 32, (3,3), atrous_rate=atrous_rates[layer])
		feat_list.append(t)
		
	t = concatenate(feat_list)
	t = Dropout(0.5)(t)

	t = conv_block(t, 128, (1,1))
	t = conv_block(t, 32, (1,1))

	t = Conv2D(3, (1, 1), activation='sigmoid', kernel_initializer='he_normal', name='mask')(t)

	return Model(inputs=[i], outputs=[t])

@addModel
def RecursiveLungNet002():

	def bn_block(x):
		return add([x, BatchNormalization()(x)])

	def conv_block(x, nb_filter, filter_size, atrous_rate=(1, 1)):
		x = Conv2D(nb_filter, filter_size, dilation_rate=atrous_rate, kernel_initializer='he_normal', padding='same')(x)
		x = bn_block(x)
		x = ReLU()(x)
		return x

	def recursive_conv_block(k, x, nb_filter, filter_size, atrous_rate=(1, 1)):
		conv1 = Conv2D(nb_filter, filter_size, dilation_rate=atrous_rate, kernel_initializer='he_normal', padding='same')
		x1 = conv1(x)
		x = bn_block(x1)
		x = ReLU()(x)
		
		for itr in range(k):
			conv2 = Conv2D(nb_filter, filter_size, dilation_rate=atrous_rate, kernel_initializer='he_normal', padding='same', weights=conv1.get_weights())
			x = conv2(x)
			if k%2 == 1:
				x = add([x1,x])
			x = bn_block(x)
			x = ReLU()(x)

		return xS

	atrous_rates = [(1,1), (1,1), (2,2), (3,3), (5,5), (8,8), (13,13), (21,21), (34,34), (55,55)] 
	n_recursions = [0,		3, 		3, 		3,		3,		3,		3,		3,		3,		3 ]

	i = Input((224, 224, 1))
	
	t = bn_block(i)
	feat_list = [t]
	for layer in range(0, len(atrous_rates)):
		t = recursive_conv_block(n_recursions[layer],t, 32, (3,3), atrous_rate=atrous_rates[layer])
		feat_list.append(t)
		
	t = concatenate(feat_list)
	t = Dropout(0.5)(t)

	t = conv_block(t, 128, (1,1))
	t = conv_block(t, 32, (1,1))

	t = Conv2D(3, (1, 1), activation='sigmoid', kernel_initializer='he_normal', name='mask')(t)

	return Model(inputs=[i], outputs=[t])

@addModel
def RecurrentLungNet001():

	def bn_block(x):
		return add([x, BatchNormalization()(x)])

	def conv_block(x, nb_filter, filter_size, atrous_rate=(1, 1)):
		x = Conv2D(nb_filter, filter_size, dilation_rate=atrous_rate, kernel_initializer='he_normal', padding='same')(x)
		x = bn_block(x)
		x = ReLU()(x)
		return x
	
	def lstm_conv_block(x, nb_filter, filter_size, atrous_rate=(1, 1), activation='tanh', return_seq=True, name=None):
		x = ConvLSTM2D(nb_filter, filter_size, dilation_rate=atrous_rate, kernel_initializer='he_normal', padding='same', return_sequences=return_seq, activation=activation,name=name)(x)
		return x
	
	def feat_extractor(slice_shape):
		# defining cnn feature extractor, applied to each slice
		atrous_rates = [(1,1), (1,1), (2,2), (3,3), (5,5), (8,8), (13,13), (21,21), (34,34), (55,55)] 
		
		i = Input((slice_shape))
		
		t = bn_block(i)
		feat_list = [t]
		for layer in range(0, len(atrous_rates)):
			t = conv_block(t, 32, (3,3), atrous_rate=atrous_rates[layer])
			feat_list.append(t)
			
		t = concatenate(feat_list)

		cnn = Model(inputs=[i], outputs=[t])
	
		return cnn


	slice_shape = (224, 224, 1)
	n_slices = 9

	i = Input((n_slices,) + slice_shape)

	cnn = feat_extractor(slice_shape)
	t = TimeDistributed(cnn, input_shape=((n_slices,) + slice_shape), name="CNN_FeatExtractor")(i)
	t = Dropout(0.5)(t)
	t = lstm_conv_block(t, nb_filter=128, filter_size=(1,1), activation='tanh', return_seq=True, name="Conv_LSTM_128")
	t = lstm_conv_block(t, nb_filter=32, filter_size=(1,1), activation='tanh', return_seq=True, name="Conv_LSTM_32")
	t = lstm_conv_block(t, nb_filter=1, filter_size=(1,1), activation='sigmoid', return_seq=True, name="mask")

	return Model(inputs=[i], outputs=[t])

@addModel
def RecurrentLungNet002():

	def bn_block(x):
		return add([x, BatchNormalization()(x)])

	def conv_block(x, nb_filter, filter_size, atrous_rate=(1, 1)):
		x = Conv2D(nb_filter, filter_size, dilation_rate=atrous_rate, kernel_initializer='he_normal', padding='same')(x)
		x = bn_block(x)
		x = ReLU()(x)
		return x
	
	def time_distributed_conv_block(x, nb_filter, filter_size, atrous_rate=(1, 1),name=None):
		x = TimeDistributed(Conv2D(nb_filter, filter_size, dilation_rate=atrous_rate, kernel_initializer='he_normal', padding='same'), name=name+'_conv')(x)
		xb = TimeDistributed(BatchNormalization(), name=name+'_bn')(x)
		x = add([x,xb], name=name+'_add')
		x = TimeDistributed(ReLU(), name=name+'_relu')(x)
		return x

	def lstm_conv_block(x, nb_filter, filter_size, atrous_rate=(1, 1), activation='tanh', return_seq=True, name=None):
		x = ConvLSTM2D(nb_filter, filter_size, dilation_rate=atrous_rate, kernel_initializer='he_normal', padding='same', return_sequences=return_seq, activation=activation,name=name)(x)
		return x
	
	def feat_extractor(slice_shape):
		# defining cnn feature extractor, applied to each slice
		atrous_rates = [(1,1), (1,1), (2,2), (3,3), (5,5), (8,8), (13,13), (21,21), (34,34), (55,55)] 
		
		i = Input((slice_shape))
		
		t = bn_block(i)
		feat_list = [t]
		for layer in range(0, len(atrous_rates)):
			t = conv_block(t, 32, (3,3), atrous_rate=atrous_rates[layer])
			feat_list.append(t)
			
		t = concatenate(feat_list)

		cnn = Model(inputs=[i], outputs=[t])
	
		return cnn


	slice_shape = (224, 224, 1)
	n_slices = 9

	i = Input((n_slices,) + slice_shape)

	cnn = feat_extractor(slice_shape)
	t = TimeDistributed(cnn, input_shape=((n_slices,) + slice_shape), name="CNN_FeatExtractor")(i)
	t = Dropout(0.25)(t)
	t = lstm_conv_block(t, nb_filter=32, filter_size=(3,3), activation='tanh', return_seq=False, name="ConvLSTM")
	# t = conv_block(t, nb_filter=32, filter_size=(1,1))
	t = Conv2D(1, (1,1), activation='sigmoid', kernel_initializer='he_normal', padding='same',name='mask')(t)

	return Model(inputs=[i], outputs=[t])

@addModel
def LungNet3D001():	
	if LSTM:
		def bn_block(x):
			return add([x, BatchNormalization()(x)])
			# return BatchNormalization()(x)

	def conv_block(x, nb_filter, filter_size, atrous_rate=(1, 1)):
		x = Conv2D(nb_filter, filter_size, dilation_rate=atrous_rate, kernel_initializer='he_normal', padding='same')(x)
		x = bn_block(x)
		x = ReLU()(x)
		return x

	def conv_block_3D(x, nb_filter, filter_size, atrous_rate=(1,1,1), recursivePass=False):
		conv_3d_1 = Conv3D(nb_filter, filter_size, dilation_rate=atrous_rate, kernel_initializer='he_normal', padding='same')
		x = conv_3d_1(x)
		x = bn_block(x)
		x = ReLU()(x)

		if recursivePass:
			conv_3d_2 = Conv3D(nb_filter, filter_size, dilation_rate=atrous_rate, padding='same',weights=conv_3d_1.get_weights())
			x = conv_3d_2(x)
			x = bn_block(x)
			x = ReLU()(x)

		return x


	slice_shape = (128, 128, 1)
	n_slices = 9

	i = Input((n_slices,) + slice_shape)

	atrous_rates = [(1,1,1), (1,1,1), (1,2,2), (1,3,3), (1,5,5), (1,8,8), (1,13,13), (1,21,21), (1,34,34)] 
	
	t = bn_block(i)
	feat_list = [t]
	for layer in range(0, len(atrous_rates)):
		t = conv_block_3D(t, 32, (3,3,3), atrous_rate=atrous_rates[layer], recursivePass=False)
		feat_list.append(t)
		
	t = concatenate(feat_list)

	t = SpatialDropout3D(0.10)(t)
	t = Conv3D(128, (1,1,1), activation='relu', kernel_initializer='he_normal', padding='valid')(t)
	t = Conv3D(32, (1,1,1), activation='relu', kernel_initializer='he_normal', padding='valid')(t)
	t = Conv3D(1, (1,1,1), activation='sigmoid', kernel_initializer='he_normal', padding='same',name='mask')(t)

	return Model(inputs=[i], outputs=[t])

@addModel
def LungNet3D002(fusionType='CONV'):

	def bn_block(x):
		return add([x, BatchNormalization()(x)])

	def lstm_conv_block(x, nb_filter, filter_size, atrous_rate=(1, 1), activation='tanh', return_seq=True, name=None):
		x = ConvLSTM2D(nb_filter, filter_size, dilation_rate=atrous_rate, kernel_initializer='he_normal', padding='same', return_sequences=return_seq, activation=activation,name=name)(x)
		return x

	def conv_block_3D(x, nb_filter, filter_size, atrous_rate=(1,1,1), recursivePass=False):
		conv_3d_1 = Conv3D(nb_filter, filter_size, dilation_rate=atrous_rate, kernel_initializer='he_normal', padding='same')
		x = conv_3d_1(x)
		x = bn_block(x)
		x = ReLU()(x)
		return x

	LungNet2D = load_model('./trained_models/LungNet001b_224x224.hdf5',custom_objects={'log_dice_coef_loss': cl.log_dice_coef_loss,
																				'dice_coef_loss': cl.dice_coef_loss,
																				'dice_coef': cl.dice_coef,
																				'log_dice_coef_loss_multi': cl.log_dice_coef_loss_multi,
																				'dice_coef_channel': cl.dice_coef_channel,
																				'dice_ch_0': cl.dice_coef_channel(0),
																				'dice_ch_1': cl.dice_coef_channel(1),
																				'dice_ch_2': cl.dice_coef_channel(2)})

	cnn2D = Model(inputs=[LungNet2D.input], outputs=[LungNet2D.layers[-6].output])
	print("Trainable layers in 2D model : ")
	for idx, layer in enumerate(cnn2D.layers):
		if idx < len(cnn2D.layers)-4:
			layer.trainable = False
		else:
			print(layer.name)

	slice_shape = (224, 224, 1)
	n_slices = 9

	i = Input((n_slices,) + slice_shape)
	t = TimeDistributed(cnn2D, input_shape=((n_slices,) + slice_shape), name="CNN_FeatExtractor")(i)
	t = Dropout(0.25)(t)

	if fusionType == 'LSTM':
		t = lstm_conv_block(t, nb_filter=64, filter_size=(3,3), activation='tanh', return_seq=True, name="Conv_LSTM_64")
		t = conv_block_3D(t, 32, (3,3,3))
		t = Conv3D(1, (1,1,1), activation='sigmoid', kernel_initializer='he_normal', padding='same',name='mask')(t)
	elif fusionType == 'CONV':
		t = conv_block_3D(t, 64, (3,3,3))
		t = conv_block_3D(t, 32, (3,3,3))
		t = Conv3D(1, (1,1,1), activation='sigmoid', kernel_initializer='he_normal', padding='same',name='mask')(t)

	return Model(inputs=[i], outputs=[t])

@addModel
def LungNetBinary001():

	def bn_block(x, name=None):
		return add([x, BatchNormalization(name=None if name is None else name + "_bn")(x)], name=None if name is None else name + "_add")

	def conv_block(x, nb_filter, filter_size, atrous_rate=(1, 1), name=None):
		x = Conv2D(nb_filter, filter_size, dilation_rate=atrous_rate, kernel_initializer='he_normal', padding='same', name=None if name is None else name + "_conv2d")(x)
		x = bn_block(x, name=name)
		x = ReLU(name=None if name is None else name + "_relu")(x)
		return x

	LungNet2D = load_model('./trained_models/LungNet001b_224x224.hdf5',custom_objects={'log_dice_coef_loss': cl.log_dice_coef_loss,
																				'dice_coef_loss': cl.dice_coef_loss,
																				'dice_coef': cl.dice_coef,
																				'log_dice_coef_loss_multi': cl.log_dice_coef_loss_multi,
																				'dice_coef_channel': cl.dice_coef_channel,
																				'dice_ch_0': cl.dice_coef_channel(0),
																				'dice_ch_1': cl.dice_coef_channel(1),
																				'dice_ch_2': cl.dice_coef_channel(2)})

	print("Trainable layers in 2D model : ")
	for idx, layer in enumerate(LungNet2D.layers):
		if idx < len(LungNet2D.layers)-7:
			layer.trainable = False
		else:
			print(layer.name)

	slice_shape = (224, 224, 1)

	t = LungNet2D.layers[-6].output
	t = Dropout(0.25, name="FeatDropout")(t)

	t = conv_block(t, 64, (3,3), name="BinConv1")
	t = MaxPooling2D((2,2))(t)
	t = conv_block(t, 64, (3,3), name="BinConv2")
	t = MaxPooling2D((2,2))(t)
	t = conv_block(t, 64, (3,3), name="BinConv3")
	t = MaxPooling2D((2,2))(t)
	t = conv_block(t, 64, (3,3), name="BinConv4")
	t = GlobalMaxPooling2D()(t)


	t = Dense(1024, activation='relu', kernel_initializer='he_normal')(t)
	t = Dense(64, activation='relu', kernel_initializer='he_normal')(t)
	t = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(t)

	return Model(inputs=[LungNet2D.input], outputs=[t])

if __name__ == '__main__':
	m = makeModel(sys.argv[1],True)
	plot_model(m, to_file='./model.png', show_shapes=True, rankdir='TB')
