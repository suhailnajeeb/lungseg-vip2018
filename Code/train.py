from modelLib import *
from trainUtils import *
import customLoss
import os
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
configTF = tf.ConfigProto()
configTF.gpu_options.allow_growth = True
sess = tf.Session(config=configTF)

tf.set_random_seed(1306)
np.random.seed(1306)

def main():

	parser = OptionParser()

	parser.add_option("-l","--load_model", dest="loadModel", default = False)
	parser.add_option("-c","--continue", dest="continueTraining", action = 'store_true', default = False)

	parser.add_option("--vb", dest="verbose", action = 'store_true', default = False)

	parser.add_option("-e","--epochs", dest="epochs", default = 100)
	parser.add_option("--bs","--batchSize", dest="batchSize", default = 5)
	parser.add_option("--lr","--learning_rate", dest="learningRate", default = 0.001)
	parser.add_option("-p", "--patience", dest="patience", default=70)

	(options, args) = parser.parse_args()

	# database to use
	dbPath = '../dbHdf5/dataset1_2d_onlyTumor_cropped_x-75-425_y-75-425.hdf5'

	modelDir = '../models'              # where models are saved
	modelArch = 'maskNet002'            # model architecture to use from modelLib.py
	modelName = 'maskNet002_007'        # name to save model with

	epochs = int(options.epochs)
	batchSize = int(options.batchSize)

	modelFolder = os.path.join(modelDir,modelName)
	weightsFolder = os.path.join(modelFolder, "weights")
	ensureDir(weightsFolder)

	notes = "Model trained on augmented data (hor flip, ver flip and elastic). Using Dice Coeff Loss"

	with open( os.path.join(modelFolder,"trainingData.txt"),"w" ) as df:
		df.write("Dataset\t%s\n" % dbPath)
		df.write("Architecture\t%s\n" % modelArch)
		df.write("Batch Size\t%s\n" % batchSize)
		df.write("Notes\t%s\n" % notes)

	db = h5py.File(dbPath,'r')
	X = db['slice'][...]
	X = np.float32(X)
	X = np.expand_dims(X, -1)
	
	Y = db['mask'][...]
	Y = np.expand_dims(Y, -1)
	Y = np.float32(Y)

	cases = db['case'][...]
	
	db.close()

	group_kfold = GroupKFold(n_splits=4)
	group_kfold.get_n_splits(X, Y, cases)

	kdx = 0
	for train_index, test_index in group_kfold.split(X, Y, cases):
		kdx += 1
		X_train = X[train_index]
		Y_train = Y[train_index]
		X_test = X[test_index]
		Y_test = Y[test_index]

		with open(os.path.join(modelFolder, "trainingData.txt"), "a") as df:
			df.write("\nTraining Cases for CV-%d (%d)\t" % (kdx, len(train_index)))
			df.write("\t".join(np.unique(cases[train_index])))
			df.write("\n")
			df.write("Test Cases for CV-%d (%d)\t" % (kdx, len(test_index)))
			df.write("\t".join(np.unique(cases[test_index])))
			df.write("\n")

		bestModelPath = os.path.join(weightsFolder, "best_fold_%02d.hdf5" % kdx)	
		ensureDir(bestModelPath)

		# creating model
		model = makeModel(modelArch, verbose=options.verbose)
		model.save( os.path.join(modelFolder, modelName + '.h5') )

		adam = Adam(lr=float(options.learningRate), beta_1=0.9, beta_2=0.999, epsilon=1e-06, decay=0.00001)

		model.compile(loss = [customLoss.dice_coef_loss], optimizer = adam)

		# loading model
		if options.loadModel:
			print("\n\nLoading Model Weights:\t %s" % modelName)
			model = load_model(bestModelPath)
			log = np.genfromtxt(os.path.join(modelFolder,  modelName + '_trainingLog.csv'),delimiter=',',dtype=str)[1:,0]
			epochStart = len(log)
		else:
			epochStart = 0

		print("\nCross Validation Fold : %02d \n" % kdx)
		# totalSamples = getSampleCount(dbPath,'slice')

		# trainGen = dataGenerator(dbPath,'slice','mask',batchSize,extendDim=True)
		# nTrainSamples = totalSamples

		
		# callbacks
		check1 = ModelCheckpoint( os.path.join(weightsFolder, modelName + "_fold_%02d" % kdx + "_{epoch:02d}-loss-{val_loss:.3f}.hdf5"), monitor='val_loss', save_best_only=True, mode='auto')
		check2 = ModelCheckpoint( bestModelPath, monitor='val_loss', save_best_only=True, mode='auto')
		check3 = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=int(options.patience), verbose=0, mode='auto')
		check4 = CSVLogger(os.path.join(modelFolder, modelName + '_trainingLog.csv'), separator=',', append=True)
		check5 = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=int(options.patience), verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=1e-10)

		print("\nInitiating Training:\n")
		# trained_model = model.fit_generator(trainGen, steps_per_epoch=(nTrainSamples // batchSize), epochs=epochs, initial_epoch=epochStart, 
		# 									callbacks=[check1,check2,check3,check4,check5], verbose=1)

		model.fit(X_train,Y_train, validation_data=(X_test,Y_test), batch_size=batchSize, epochs=epochs, initial_epoch=epochStart, 
				callbacks=[check1,check2,check3,check4,check5], verbose=1)

		del X_test
		del X_train
		del Y_test
		del Y_train
			
if __name__ == "__main__":
	main()
