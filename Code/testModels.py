from modelLib import *
from trainUtils import *
from keras.models import load_model
import customLoss as cl
import metrics as m
import keras
import keras.backend as K

# suprresing tensorflow messages
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
configTF = tf.ConfigProto()
configTF.gpu_options.allow_growth = True
sess = tf.Session(config=configTF)

tf.set_random_seed(1306)
np.random.seed(1306)


# database to use
dbPath = 'P:/dataset3_2d_onlyTumor_cropped_x-74-426_y-74-426_resized_224-224_clipped-0-1800_wavelet_scaled_val.hdf5'

db = h5py.File(dbPath, 'r')
X = db['slice'][...]
X = np.float32(X)
X = np.expand_dims(X, -1)

Y = db['mask'][...]
Y = np.expand_dims(Y, -1)
Y = np.float32(Y)

cases = db['case'][...]

tidx = list(range(0, X.shape[0]))   # all
np.random.shuffle(tidx)

X = X[tidx, ...]
Y = Y[tidx, ...]
cases = cases[tidx, ...]

modelDir = '../models'              # where models are saved

# models to use ensemble
modelName = ['maskNet002e_001']

# segmentation threshold
segThreshold = 0.2

YP = None
for mn in modelName:
	modelFolder = os.path.join(modelDir,mn)
	weightsFolder = os.path.join(modelFolder, "weights")
	bestModelPath = os.path.join(weightsFolder, "best.hdf5")

	model = load_model(bestModelPath,custom_objects={'log_dice_coef_loss': cl.log_dice_coef_loss, 'dice_coef_loss': cl.dice_coef_loss, 'dice_coef':cl.dice_coef})

	if YP is None:
		YP = model.predict(X, batch_size=50, verbose=1)
	else:
		YP += model.predict(X, batch_size=50, verbose=1)

	K.clear_session()

YP = np.divide(YP, len(modelName))

# YP = unshuffle(YP,tidx)

dc = []
dc3D = []
for i in range(YP.shape[0]):
	yp = YP[i,...]
	ygt = Y[i,...]

	if len(yp.shape) > 3:
		dcSlice=[]
		for j in range(yp.shape[2]):
			cenp = yp[:,:,j,:]
			gtp = ygt[:,:,j,:]
			dcSlice.append(m.dice_coef(gtp,cenp,threshold=segThreshold))

		dc.append(dcSlice)

	dc3D.append(m.dice_coef(ygt, yp,threshold=segThreshold))

if len(yp.shape) > 3:
	dc = np.array(dc)
	print("Mean DC : \n")
	print(np.mean(dc,axis=0))

dc3D = np.array(dc3D)
print("\nMean 3D DC : \n")
print(np.mean(dc3D))

'''
db = h5py.File( os.path.join('../predictedMasks','pred_' + dbName), mode='w')
db.create_dataset("slice", X.shape, X.dtype)		
db.create_dataset("maskGT",  Y.shape, Y.dtype)
db.create_dataset("maskPred",  YP.shape, YP.dtype)
db.create_dataset("case",  cases.shape, cases.dtype)
db['slice'][...] = X[...]
db['maskGT'][...] = Y[...]
db['maskPred'][...] = YP[...]
db['case'][...] = cases[...]
db.close()

k = 0
i = 2
while k != 'q':
	x = np.expand_dims(X[i,:],0)
	yp = np.array(YP[i,...])

	yp = np.reshape(yp, (yp.shape[0], yp.shape[0]))
	yp[yp < 0.5] = 0
	yp[yp>=0.5] = 1
	
	ygt = np.reshape(Y[i, :], (yp.shape[0], yp.shape[0]))

	dc = K.eval(m.dice_coef(ygt,yp))
	print("\nDice Coefficient : %f" % dc)

	plt.subplot(121)
	plt.imshow(yp)
	plt.title("Predicted Mask")

	plt.subplot(122)
	plt.imshow(ygt)
	plt.title("Ground Truth")

	plt.show()
	k = raw_input(" d = next, a = previous, q = quit ")

	if k == 'd':
		i += 1
		i %= len(Y)

	elif k == 'a':
		i-=1
		i = 0 if i<0 else i

'''
