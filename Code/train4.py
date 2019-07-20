from modelLib import *
from trainUtils import *
import customLoss
import os
import gc
import pickle
from optparse import OptionParser

from keras.models import load_model
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold

# suprresing tensorflow messages
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.set_random_seed(1306)
np.random.seed(1306)


def main():

	parser = OptionParser()

	parser.add_option("-l", "--load_model", dest="loadModel", action='store_true', default=False)
	parser.add_option("-c", "--continue", dest="continueTraining",action='store_true', default=False)

	parser.add_option("--vb", dest="verbose", action='store_true', default=False)
	parser.add_option("--nw", "--numWorkers", dest="numWorkers", default=4)
	parser.add_option("-e", "--epochs", dest="epochs", default=80)
	parser.add_option("--bs", "--batchSize", dest="batchSize", default=20)
	parser.add_option("--ed", "--extendDim", dest="extendDim", action='store_true', default=False)
	parser.add_option("--lr", "--learning_rate",dest="learningRate", default=0.001)
	parser.add_option("-p", "--patience", dest="patience", default=10)

	(options, args) = parser.parse_args()

	# database to use
	trainDbPath = 'P:/dataset3_2d_cropped_x-74-426_y-74-426_clipped-0-1800_resized-224-224_aug-R90-R180-R270-Hf-Vf-Et-Compound_scaled_allregions_train.hdf5'
	valDbPath = 'P:/dataset3_2d_cropped_x-74-426_y-74-426_clipped-0-1800_resized-224-224_scaled_allregions_val.hdf5'

	modelDir = '../models'               # where models are saved
	modelArch = 'LungNet004'            # model architecture to use from modelLib.py
	modelName = 'LungNet004_001'        # name to save model with

	epochs = int(options.epochs)
	batchSize = int(options.batchSize)
	numWorkers = int(options.numWorkers)

	modelFolder = os.path.join(modelDir, modelName)
	weightsFolder = os.path.join(modelFolder, "weights")
	ensureDir(weightsFolder)

	notes = "LungNet with double filters (64). Train data augmented with compounding on, 12x"

	with open(os.path.join(modelFolder, "trainingData.txt"), "w") as df:
		df.write("Train Dataset\t%s\n" % trainDbPath)
		df.write("Val Dataset\t%s\n" % valDbPath)
		df.write("Architecture\t%s\n" % modelArch)
		df.write("Batch Size\t%s\n" % batchSize)
		df.write("Notes\t%s\n" % notes)

	db = h5py.File(trainDbPath, 'r')
	trainCases = db['case'][...]
	db.close()

	db = h5py.File(valDbPath, 'r')
	valCases = db['case'][...]
	db.close()

	bestModelPath = os.path.join(weightsFolder, "best.hdf5")
	ensureDir(bestModelPath)

	# creating model
	model = makeModel(modelArch, verbose=options.verbose)
	model.save(os.path.join(modelFolder, modelName + '.h5'))

	adam = Adam(lr=float(options.learningRate), beta_1=0.9,beta_2=0.999, epsilon=1e-06, decay=0.00001)

	# loading model
	if options.loadModel:
		print("\n\nLoading Model Weights:\t %s" % bestModelPath)
		model.load_weights(bestModelPath)
		log = np.genfromtxt(os.path.join(modelFolder,  modelName + '_trainingLog.csv'), delimiter=',', dtype=str)[1:, 0]
		epochStart = len(log)
	else:
		epochStart = 0

	losses = {'mask':customLoss.log_dice_coef_loss_multi, 
           	#   'binClass': customLoss.weighted_crossentropy([1.1,1]),
 			 }

	lossWeights = {'mask' : 1.0,
				#    'binClass' : 0.5,
		     	  }

	metrics = {'mask': [customLoss.dice_coef_channel(0), customLoss.dice_coef_channel(1), customLoss.dice_coef_channel(2)],
            #    'binClass': 'binary_accuracy',
			  }

	model.compile(loss=losses, loss_weights=lossWeights, optimizer=adam, metrics=metrics)

	trainGen = dataGenerator(trainDbPath,['slice'],['mask'],batchSize, extendDim=options.extendDim,numWorkers=numWorkers)
	testGen = dataGenerator(valDbPath,['slice'],['mask'],batchSize, extendDim=options.extendDim,numWorkers=numWorkers)

	# callbacks
	check1 = ModelCheckpoint(os.path.join(weightsFolder, modelName + "_{epoch:02d}-loss-{val_loss:.3f}.hdf5"), monitor='val_loss', mode='auto')
	check2 = ModelCheckpoint(bestModelPath, monitor='val_loss', save_best_only=True, mode='auto')
	check3 = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=int(options.patience), verbose=0, mode='auto')
	check4 = CSVLogger(os.path.join(modelFolder, modelName +'_trainingLog.csv'), separator=',', append=True)
	check5 = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.001, cooldown=2, min_lr=1e-10)

	print("\nInitiating Training:\n")
	trained_model = model.fit_generator(trainGen, steps_per_epoch=(len(trainCases) // batchSize), epochs=epochs, initial_epoch=epochStart,
										validation_data= testGen, validation_steps=(len(valCases) // batchSize), callbacks=[check1,check2,check3,check4,check5], 
										verbose=1)
	trainGen.close()
	testGen.close()
	gc.collect()

	# closing hdf5 db in case it was left open due to unexpected shutdown
	for obj in gc.get_objects(): 
		try:
			if isinstance(obj, h5py.File):
				try:
					obj.close()
				except:
					pass # Was already closed
		except:
			pass

if __name__ == "__main__":
	main()
