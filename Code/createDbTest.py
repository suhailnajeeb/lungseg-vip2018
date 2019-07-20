def createDatabase2D_Coronal(dbMetaData, dbPath, wavelet=False, onlyTumor=False, augments=[], crop=None, resizeTo = None, rescale = False, clipHU=None, expandMasks=False, minArea=100, applyGaussianFilter=False):

	''' This function creates an HDF5 database file with the 2D tumor slices
		under 'slice' and their coressponding masks under 'mask'. The name of 
		the case file from which the slice came from is under 'case' '''

	X = None         # slices
	Y = None         # masks
	hasTumor = []	 # whether slice has tumor(1) or not(0) 
	caseNames = []   # name of case where slice came from
	
	if crop is None:
		cz1, cz2 = (0, 120 if resizeTo is None else resizeTo[0])
		cy1, cy2 = (0, 512 if resizeTo is None else resizeTo[1])
	else:
		cy1, cz1, cy2, cz2 = crop
	
	if resizeTo is None:
		w = cy2 - cy1
		h = cz2 - cz1
	else:
		w = resizeTo[0]
		h = resizeTo[1]

	nTotalEntries,nBaseEntries = getTotalEntryCount(dbMetaData, onlyTumor, augments)

	try:
		print("\nCreating database with %d entries" % nTotalEntries)
		print("Each slice has shape %d x %d" % (h, w))
		ensureDir(dbPath)
		db = h5py.File(dbPath, mode='w')
		db.create_dataset("slice", (nTotalEntries, h, w,), np.float32)		
		db.create_dataset("mask",  (nTotalEntries, h, w,), np.float32 if applyGaussianFilter else np.uint8)
		db.create_dataset("case",  (nTotalEntries,), np.dtype('|S16'))
		db.create_dataset("tumor", (nTotalEntries,), np.uint8)			# binary label
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

		slices = stackSlices(case)
		masks, midx = genMask(case)

		slices = changePlane(slices,'yz')
		masks = changePlane(masks,'yz')
		midx =  getMaskIndex(masks)
		
		sliceLabels = np.uint8([0] * slices.shape[-1])
		sliceLabels[midx] = 1

		# adjusting HU scale for other manufacturer (CMS)
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

		slices, _ = resampleVoxels(slices, oldDim=(dy0,dx0,dz0), newDim=(0.97,0.97,3))
		masks, _ = resampleVoxels(masks, oldDim=(dy0, dx0, dz0), newDim=(0.97,0.97,3))
		
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

							if xRot is None:
								xRot = cv2.warpAffine(s, rotMat, (cols, rows))
								yRot = cv2.warpAffine(m, rotMat, (cols, rows))
							else:
								xRot = np.dstack((xRot,cv2.warpAffine(s, rotMat, (cols, rows))))
								yRot = np.dstack((yRot,cv2.warpAffine(m, rotMat, (cols, rows))))

					augSets.append( [xRot,yRot])

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
							xHorFlipped = np.dstack((xHorFlipped,(cv2.flip(s, 1))))
							yHorFlipped = np.dstack((yHorFlipped,(cv2.flip(m, 1))))

					augSets.append([xHorFlipped, yHorFlipped])

				if 'verFlip' in augments:
					xVerFlipped = None
					yVerFlipped = None

					for idx in range(x.shape[-1]):
						s = x[...,idx]
						m = y[...,idx]
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
							sliceLabels.extend(sliceLabelsOriginal)	# adding labels for previous augmentatioons

						for (xaug,yaug) in augSets:
							x = np.dstack((x, xaug))
							y = np.dstack((y, yaug))

					for idx in range(x.shape[-1]):
						s = x[..., idx]
						m = y[..., idx]
						
						sigma = np.random.uniform(0.07,0.15, 1)
						affineFactor = np.random.uniform(0.07,0.15,1)
						xe,ye = elasticTransform(s,m, sigma*s.shape[1], affineFactor*s.shape[1], 1234)
						
						if xElastic is None:
							xElastic = xe
							yElastic = ye
						else:
							xElastic = np.dstack((xElastic, xe))
							yElastic = np.dstack((yElastic, ye))

					if 'compound' in augments:
						x = np.dstack((x,xElastic))
						y = np.dstack((y,yElastic))
						sliceLabelsOriginal = sliceLabels[:]
						sliceLabels.extend(sliceLabelsOriginal)
					else:
						augSets.append([xElastic,yElastic])

				if 'compound' not in augments:			
					sliceLabelsOriginal = sliceLabels[:]
					for n in range(len(augments)):
						sliceLabels.extend(sliceLabelsOriginal)

					for (xaug,yaug) in augSets:
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
			print(name)
			print(err)
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

			area,bbox,centroid,solidity = calculateRegionProps(X,Y)


			db["slice"][didx:didx + flushSize, ...] = np.float32(X[...])
			db["mask"][didx:didx + flushSize, ...] = Y[...]
			db["case"][didx:didx + flushSize, ...] = caseNames[...]
			db["tumor"][didx:didx + flushSize, ...] = hasTumor[...]
			db["area"][didx:didx + flushSize, ...] = area[...]
			db["bbox"][didx:didx + flushSize, ...] = bbox[...]
			db["centroid"][didx:didx + flushSize, ...] = centroid[...]
			db["solidity"][didx:didx + flushSize, ...] = solidity[...]
			
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

			area,bbox,centroid,solidity = calculateRegionProps(X,Y)
			

			db["slice"][didx:didx + flushSize, ...] = np.float32(X[...])
			db["mask"][didx:didx + flushSize, ...] = Y[...]
			db["case"][didx:didx + flushSize, ...] = caseNames[...]
			db["tumor"][didx:didx + flushSize, ...] = hasTumor[...]
			db["area"][didx:didx + flushSize, ...] = area[...]
			db["bbox"][didx:didx + flushSize, ...] = bbox[...]
			db["centroid"][didx:didx + flushSize, ...] = centroid[...]
			db["solidity"][didx:didx + flushSize, ...] = solidity[...]

			if wavelet:
				wav = calculateWavelet(X)
				db["wavelet"][didx:didx + flushSize, ...] = wav[...]

	db.close()
