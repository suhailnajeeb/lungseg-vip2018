from trainUtils import *
from skimage import measure
from progressbar import ProgressBar
from matplotlib import pyplot as plt
import pickle
import itertools


def plotScanWithCorrection(img, mask=None, sidx=None, threshold=None, colorMap=plt.cm.bone):
	''' This function takes a stacked slices (img) and optinally
	stacked mask and slice indices, and displays them sequentially
	in a 2D map color map '''

	# callback for displaying slices and slider control
	class callback(object):

		def __init__(self, fig, ax, img, mask, sidx, threshold):
			self.fig = fig
			self.ax = ax
			self.img = img
			self.img0 = img.copy()
			self.mask = mask
			self.sidx = sidx
			self.threshold = threshold
			self.interval = 1000
			self.dx = 0
			self.dy = 0
			self.stepSize = 1
			self.cb = None
		
		def increaseFPS(self, event):
			self.interval += 100
			self.interval = 5000 if self.interval > 5000 else self.interval
			cb.anim = animation.FuncAnimation(self.fig, self.nextSlice, interval=self.interval, repeat=True,
                                     frames=img.shape[-1] if sidx is None else len(sidx))
			print("Frame Interval : %d" % self.interval)

		def decreaseFPS(self,event):
			self.interval -= 100
			self.interval = 30 if self.interval < 100 else self.interval
			cb.anim = animation.FuncAnimation(self.fig, self.nextSlice, interval=self.interval, repeat=True,
										frames=img.shape[-1] if sidx is None else len(sidx))
			print("Frame Interval : %d" % self.interval)

		def increaseStep(self,event):
			self.stepSize +=1
			self.stepSize = 200 if self.stepSize > 200 else self.stepSize
			print("Step Size : %d" % self.stepSize)
		
		def decreaseStep(self,event):
			self.stepSize -= 1
			self.stepSize = 1 if self.stepSize < 1 else self.stepSize
			print("Step Size : %d" % self.stepSize)

		def moveLeft(self,event):
			self.mask = np.roll(self.mask, -self.stepSize, 1)
			self.dx -= self.stepSize
			self.dx %= self.img.shape[1]
			print("DX = %d" % self.dx)

		def moveRight(self,event):
			self.mask = np.roll(self.mask, self.stepSize, 1)
			self.dx += self.stepSize
			self.dx %= self.img.shape[1]
			print("DX = %d" % self.dx)

		def moveUp(self,event):
			self.mask = np.roll(self.mask, -self.stepSize, 0)
			self.dy -= self.stepSize
			self.dy %= self.img.shape[0]
			print("DY = %d" % self.dy)

		def moveDown(self,event):
			self.mask = np.roll(self.mask, self.stepSize, 0)
			self.dy += self.stepSize
			self.dy %= self.img.shape[0]
			print("DY = %d" % self.dy)

		def updateThreshold(self, value):
			self.threshold = value
			self.img = self.img0.copy()
			self.img[self.img <= value] = -1000

		def nextSlice(self, frameNum):
			self.ax.clear()

			if self.sidx is None:
				i = frameNum
			else:
				i = self.sidx[frameNum]

			self.ax.set_title("Slice %03d" % i)

			self.ax.imshow(self.img[..., i], cmap=colorMap)

			if mask is not None:
				self.ax.contour(self.mask[..., i], levels=[0], colors='r')

	fig = plt.figure()
	ax = plt.subplot(1, 1, 1)

	cb = callback(fig, ax, img, mask, sidx, threshold)

	if sidx is None:
		i = 0
	else:
		i = sidx[0]

	# if threshold is not None:
	# 	img[img <= threshold] = -1000
	# 	thresh = Slider(plt.axes([0.1, 0.05, 0.3, 0.05]), 'HU Threshold',
	# 			  valmin=-1500, valmax=3000, valinit=threshold,
	# 			  color='lightblue')

	# 	thresh.on_changed(cb.updateThreshold)

	incFPS = Button(plt.axes([0.1, 0.35, 0.08, 0.05]), 'FPS +', color='grey')
	decFPS = Button(plt.axes([0.1, 0.30, 0.08, 0.05]), 'FPS -', color='grey')

	incStep = Button(plt.axes([0.1, 0.2, 0.08, 0.05]), 'Step +', color='grey')
	decStep = Button(plt.axes([0.1, 0.15, 0.08, 0.05]), 'Step -', color='grey')
	
	xLeft = Button(plt.axes([0.1, 0.05, 0.075, 0.05]), 'Left', color='grey')
	xRight = Button(plt.axes([0.2, 0.05, 0.075, 0.05]), 'Right', color='grey')
	yUp = Button(plt.axes([0.3, 0.05, 0.075, 0.05]), 'Up', color='grey')
	yDown = Button(plt.axes([0.4, 0.05, 0.08, 0.05]), 'Down',color='grey')
	
	xLeft.on_clicked(cb.moveLeft)
	xRight.on_clicked(cb.moveRight)
	yUp.on_clicked(cb.moveUp)
	yDown.on_clicked(cb.moveDown)
	
	incStep.on_clicked(cb.increaseStep)
	decStep.on_clicked(cb.decreaseStep)
	incFPS.on_clicked(cb.increaseFPS)
	decFPS.on_clicked(cb.decreaseFPS)

	ax.set_title("Slice %03d" % i)
	cs = ax.imshow(img[..., i], cmap=colorMap)

	if mask is not None:
		ax.contour(mask[..., i], levels=[0], colors='r')

	plt.axis('off')
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	cb.anim = animation.FuncAnimation(fig, cb.nextSlice, interval=cb.interval, repeat=True,
								frames=img.shape[-1] if sidx is None else len(sidx))

	plt.show()

# loading case meta data
with open('./dataset1_meta.p', 'rb',) as df:
	dataMeta = pickle.load(df)


# This cases have displaced or incorrect annotation files
# Some maybe be corrected using the map below
skipCase = ['LUNG1-034',
            'LUNG1-040',
            'LUNG1-044',
            'LUNG1-068',
            'LUNG1-083',
            'LUNG1-084',
            'LUNG1-094',
            'LUNG1-096', ]

# corrections applied by using np.roll on the 3D mask along the y-axis(0) and x-axis(1)
# format = (dy,dx), None implies correction requires more than simple translation
corrections = {
    'LUNG1-034': (165, 0),
    'LUNG1-040': (190, 0),
    'LUNG1-044': (166, 0),
    'LUNG1-068': None,
    'LUNG1-083': (293, 0),
    'LUNG1-084': (314, 0),
    'LUNG1-094': (328, 0),
    'LUNG1-096': (318, 0),
}

for caseName in skipCase:
	print("--------------Case : %s-----------------\n" % caseName)
	img = stackSlices(dataMeta[caseName])
	mask, maskSlices = genMask(dataMeta[caseName])

	if caseName in corrections:
		if corrections[caseName] is not None:
			mask = np.roll(mask,corrections[caseName][0],0)
			mask = np.roll(mask,corrections[caseName][1],1)

	plotScanWithCorrection(img, mask, maskSlices, threshold=-1500)
	print("-----------------------------------------------\n")
