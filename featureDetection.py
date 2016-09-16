import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from preProcess import preProcess 

class scaleCornerDetection():
	def __init__(self, image, thickness):
		
		ret, image = cv.threshold(image, 180, 255, 0)####
		
		self.image = image
		self.size = thickness
		self.upperCorner = []
		self.downCorner = []
		self.jointPoints = []
		self.pipeline()
	
	def pipeline(self):
		kernel = self.createKernel(1)
		self.upperCorner = self.cornerDetect(kernel)
# 		kernel = self.createKernel(0)
# 		self.jointPoints = self.cornerDetect(kernel)
# 		kernel = self.createKernel(-1)
# 		self.downCorner = self.cornerDetect(kernel)
		self.displayCorners()

	def negateImage(self, image):
		image = 255-image

	def normalizeKernel(self, kernel):
		height, width = kernel.shape
		summ = np.sum(kernel)
		kernel = kernel / summ
		if kernel[0][0] != 0:
			pixel = kernel[0][0]
		else:
			pixel = kernel[height-1][width-1]

		return kernel, pixel

	def selectKernel(self, varList):
		selected = None
		for i in range(len(varList)):
			xVar, yVar = varList[i]
			if not selected and xVar > 10 and yVar > 10:
				selected =  i

		return i+1

	def sortByLeftTop(self, item):
		return (item[0], item[1])

	def removeRepeatCorner(self, cornerList):
		i=0
		margin = 5
		xList = []
		yList = []
		while i<len(cornerList):
			x, y = cornerList[i]
			xList.append(x)
			yList.append(y)
			j = i
			while j+1<len(cornerList) and x + margin > cornerList[j+1][0] and x-margin < cornerList[j+1][0]:
				if y+margin > cornerList[j+1][1] and y-margin < cornerList[j+1][1]:
					del cornerList[j+1]
				else:
					j+=1
			i +=1
		return cornerList

	def createKernel(self, mode):
		size = self.size
		width = size
		kernelSize = width *3
		kernel = np.zeros((kernelSize, kernelSize), np.float32)
		if mode==1:
			for i in range(width):
				for x in range(kernelSize):
					kernel[i][x] = 1
					kernel[x][i] = 1
		elif mode==0:
			for i in range(width):
				for x in range(kernelSize):
					kernel[x][kernelSize-1 - i] = 1
					kernel[width+i][x] = 1
		elif mode == -1:
			for i in range(width):
				for x in range(kernelSize):
					kernel[x][i] = 1
					kernel[kernelSize-1 - i][x] = 1
		kernelPackage = self.normalizeKernel(kernel)
		print 
		print "kernel: ", kernel
		print "kernel, pixel", kernelPackage
		return kernelPackage


	def cornerDetect(self, kernelPackage):

		size = self.size
		width = size
		kernelSize = width * 3	
		kernel, value = kernelPackage	
		filteredImage = cv.filter2D(self.image, -1, kernel)
		
		print "filterImage", filteredImage.dtype
		print filteredImage
		print "image"
		print self.image
		
		print size
		
		# plt.imshow(filteredImage, cmap='Greys_r')
		# plt.show()

		mmin = np.amin(filteredImage)
		# print mmin
		threshold= value * pow(size,1.5) * 255
		
		print "threshold: ", threshold
		# print threshold
		if mmin - threshold < 0:
			lowerBound = 0
		else:
			lowerBound = mmin - threshold
		upperBound = mmin + threshold
		print mmin, lowerBound, upperBound
		indices = np.where(filteredImage >lowerBound) and np.where( filteredImage < upperBound)
		
		# indices = np.where(filteredImage == mmin)
		cornerList = zip(indices[1], indices[0])
		
		new_cornerList = []
		
		for corner in cornerList:
			print corner 
			
			if(corner[0]+2 <= self.image.shape[0] and corner[1]+2 <= self.image.shape[1]):
				patch = self.image[corner[0]-1:corner[0]+2, corner[1]-1:corner[1]+2].copy().astype("float")/255	  
				patch_variance =  np.var(patch)
				patch_sum = max(np.amax(np.sum(patch, 0)), np.amax(np.sum(patch, 1)))
				print "patch"
				print patch
				print "sum, ", patch_sum
				
				print max(np.amax(np.sum(patch, 0)), np.amax(np.sum(patch, 1)))
				
				print "filter value,", filteredImage[corner[0], corner[1]]
				print "var,", patch_variance

				if patch_variance > 0.2 and patch_sum <= 2:
					new_cornerList.append(corner)
				
		print "cornerList", len(cornerList)
		print "new_cornerList", len(new_cornerList)
		
# 		print cornerList
		cornerList = new_cornerList
		tmpImage = self.image.copy()
		cornerList = sorted(cornerList, key=self.sortByLeftTop)
		cornerList = self.removeRepeatCorner(cornerList)

		return cornerList

		# for corner in cornerList:
		# 	x,y = corner
		# 	rad = 5
		# 	cv.rectangle(tmpImage, (x-rad, y-rad), (x+rad, y+rad), color=(0))
		# plt.imshow(tmpImage, cmap='Greys_r')
		# plt.show()

		# CornerValue = []
		# rad = 2
		# self.refUpperCorner = sorted(self.refUpperCorner, key = self.sortByLeftTop)
		# self.refUpperCorner = self.removeRepeatCorner(self.refUpperCorner)
		# for corner in self.refUpperCorner:
		# 	x, y = corner
		# 	x = x + self.size
		# 	y = y + self.size
		# 	cv.rectangle(filteredImage, (x-rad, y - rad), (x + rad, y +rad), color=(0), thickness=1)
		# 	CornerValue.append(filteredImage[y][x])
		# print cornerList
		# print self.refUpperCorner
		# plt.imshow(filteredImage, cmap='Greys_r')
		# plt.show()
		

	def parsing(self):
		varList = []
		for size in range(1,6):
			width = size
			kernelSize = width*3
			kernel = np.zeros((kernelSize, kernelSize), np.float32)
			for i in range(width):
				for x in range(kernelSize):
					kernel[i][x] = 1
					kernel[x][i] = 1

			kernel, value = self.normalizeKernel(kernel)
			# print kernel
			filteredImage = cv.filter2D(self.image, -1, kernel)
			plt.imshow(filteredImage, cmap='Greys_r')
			plt.show()
			# newKernel = np.zeros((kernelSize, kernelSize), np.float32)
			# for i in range(width):
			# 	for x in range(i, kernelSize):
			# 		if x < width:
			# 			if i==0:
			# 				newKernel[i][x] = summ - (kernelSize - width ) * x
			# 				newKernel[x][i] = summ - (kernelSize - width ) * x
			# 			elif x==i:
			# 				newKernel[i][x] = summ - (kernelSize - width) * x - (kernelSize - width + i) * x
			# 				newKernel[x][i] = summ - (kernelSize - width) * x - (kernelSize - width + i) * x
			# 			else:
			# 				newKernel[i][x] = newKernel[0][x] - (kernelSize - width + x) * i
			# 				newKernel[x][i] = newKernel[0][x] - (kernelSize - width + x) * i
			# 		else:
			# 			newKernel[i][x] = (kernelSize) * (width - i)
			# 			newKernel[x][i] = (kernelSize) * (width - i)
			# newKernel = 1.0 / (newKernel+0.00000001)
			# print newKernel
			# newKernel, summ = self.normalizeKernel(kernel)
			# tmpImage = 255 - filteredImage
			# doubleFilteredImage = cv.filter2D(tmpImage, -1, newKernel)
			# plt.imshow(doubleFilteredImage, cmap='Greys_r')
			# plt.show()
			# print np.amax(doubleFilteredImage)
			mmin = np.amin(filteredImage)
			# print mmin
			threshold= value * pow(size,1.2) * 255
			# print threshold
			if mmin - threshold < 0:
				lowerBound = 0
			else:
				lowerBound = mmin - threshold
			upperBound = mmin + threshold
			indices = np.where(filteredImage >lowerBound) and np.where( filteredImage < upperBound)
			# indices = np.where(filteredImage == mmin)
			cornerList = zip(indices[0], indices[1])
			tmpImage = self.image.copy()
			i=0
			xList = []
			yList = []
			while i<len(cornerList):
				x, y = cornerList[i]
				xList.append(x)
				yList.append(y)
				j = i
				# while j+1<len(cornerList) and x + 5 > cornerList[j+1][0] and  x-5 < cornerList[j+1][0] and y + 5 > cornerList[j+1][1] and y-5 < cornerList[j+1][1]:
				# 	del cornerList[j+1]
				# while j+1<len(cornerList) and x + 5 > cornerList[j+1][0] and  y + 5 > cornerList[j+1][1]:
				# 	del cornerList[j+1]
				while j+1<len(cornerList) and x + 5 > cornerList[j+1][0] and x-5 < cornerList[j+1][0]:
					if y+5 > cornerList[j+1][1] and y-5 < cornerList[j+1][1]:
						del cornerList[j+1]
					else:
						j+=1
				i +=1
			xAve = sum(xList)/len(xList)
			yAve = sum(yList)/len(yList)
			xVar = 0.0
			yVar = 0.0
			for x in xList:
				xVar += (pow(x - xAve, 2)+0.0)
			for y in yList:
				yVar += (pow(y - yAve, 2)+0.0)



			for corner in cornerList:
				y,x = corner
				rad = 5
				cv.rectangle(tmpImage, (x-rad, y-rad), (x+rad, y+rad), color=(0))
			# print cornerList
			# print len(cornerList)
			# xVar = (pow(xVar, 0.5)+0.0)/len(xList)
			# yVar = (pow(yVar, 0.5)+0.0)/len(yList)
			print (pow(xVar, 0.5)+0.0)/len(xList), (pow(yVar, 0.5)+0.0)/len(yList)
			# print mmin
			plt.imshow(tmpImage, cmap='Greys_r')
			plt.show()

			varList.append((xVar, yVar))

	def displayCorners(self):
		whatever = cv.cvtColor(self.image,cv.COLOR_GRAY2RGB)
		rad = 2
		for x, y in self.upperCorner:
			cv.rectangle(whatever, (x-rad, y - rad), (x + rad, y +rad), color=(255, 0 , 0), thickness=2)
		for x, y in self.jointPoints:
			cv.rectangle(whatever, (x-rad, y - rad), (x + rad, y +rad), color=(0, 255 , 0), thickness=2)
		for x, y in self.downCorner:
			cv.rectangle(whatever, (x-rad, y - rad), (x + rad, y +rad), color=(0, 0 , 255), thickness=2)
		print len(self.upperCorner), len(self.jointPoints), len(self.downCorner)
		plt.imshow(whatever)
		plt.show()

	def getCornerLists(self):
		return self.upperCorner, self.jointPoints, self.downCorner


			




class cornerDetection():
	def __init__(self, image):
		self.image = image

		self.cornerList = []
		self.upperList = []
		self.lowerList = []
		self.jointList = []
		self.cornerNum = 0
		(self.height, self.width) = image.shape
		self.buildKernel()
		self.negateImage()
		self.parsing()
		self.sortCorners()
		self.displayCorners()

	def negateImage(self, thres = 60):
		self.image = 255 - self.image
		ret, self.image = cv.threshold(self.image, thres, 255, 0)
		# for r in range(self.height):
		# 	for c in range(self.width):
		# 		tmp = 255 - self.image[r][c]
		# 		if tmp < thres:
		# 			self.image[r][c] = 0
		# 		else:
		# 			self.image[r][c] = 255


	def buildKernel(self):
		# build upper kernel and lower kernel
		self.upperKernel = np.array(([0,0,0,0,0], [0,0,0,0,0], [0,0,1.0/6,1.0/6,1.0/6], [0,0,1.0/6,1.0/6,0], [0,0,1.0/6,0,0]), np.float32)
		self.downKernel = np.array(([0,0,1.0/6,0,0], [0,0,1.0/6,1.0/6,0], [0,0,1.0/6,1.0/6,1.0/6], [0,0,0,0,0], [0,0,0,0,0]), np.float32)		
		self.jointKernel = np.array(([0,0,1.0/7,0,0], [0,0,1.0/7,0,0], [1.0/7,1.0/7,1.0/7,0,0], [0,0,1.0/7,0,0], [0,0,1.0/7,0,0]), np.float32)

		# for row in self.upperKernel:
		# 	for pixel in row:
		# 		print pixel,
		# 	print



	def parsing(self):
		# convolute the image with kernels
		self.radius = 2
		self.windowSize = 5

		self.upCorner = cv.filter2D(self.image, -1, self.upperKernel)
		self.downCorner = cv.filter2D(self.image, -1, self.downKernel)
		self.jointPt = cv.filter2D(self.image, -1, self.jointKernel)

		ret, self.upCorner = cv.threshold(self.upCorner, 210, 255, 0)
		ret, self.downCorner = cv.threshold(self.downCorner, 210, 255, 0)
		ret, self.jointPt = cv.threshold(self.jointPt, 254, 255, 0)

		tmpList = np.argwhere( self.upCorner == 255 )
		self.upperList = tmpList.tolist()

		tmpList = np.argwhere( self.downCorner == 255)
		self.downList = tmpList.tolist()

		tmpList = np.argwhere( self.jointPt == 255)
		self.jointList = tmpList.tolist()

	def sortCorners(self):
		self.upperList = sorted(self.upperList, key=self.cornerGetKey)
		self.downList = sorted(self.downList, key=self.cornerGetKey)
		self.jointList = sorted(self.jointList, key=self.cornerGetKey)

	def cornerGetKey(self, item):
		return (item[1], item[0])

	def getCornerLists(self):
		return (self.upperList, self.downList, self.jointList)

	def findCorners(self, thres=50):
		pass

	def postPro(self):
		# process the case that has more than two branches
		pass

	def printCorners(self):
		print (self.upperList, self.downList)

	def getCornerNum(self):
		self.cornerNum = (len(self.upperList), len(self.downList))
		return self.cornerNum

	def displayCorners(self, rad=3):
		for spot in self.downList:
			cv.rectangle(self.image, (spot[1]-rad, spot[0]-rad), (spot[1]+rad, spot[0]+rad), color=(255))
		
		for spot in self.upperList:
			cv.rectangle(self.image, (spot[1]-rad, spot[0]-rad), (spot[1]+rad, spot[0]+rad), color=(255))
			
		for spot in self.jointList:
			cv.rectangle(self.image, (spot[1]-rad, spot[0]-rad), (spot[1]+rad, spot[0]+rad), color=(255))

	def displayImage(self):
		plt.imshow(self.image, cmap='Greys_r')
		plt.show()

	def returnImage(self):
		return self.image


class lineDetection():
	def __init__(self, image, minLength = 10):
		self.image = image
		(self.height, self.width) = image.shape
		self.linesList = []
		self.horLines = []
		self.verLines = []

		self.negateImage()
		mode = 0
		self.lineDetection(mode, 12)		
		self.rotateImage()
		mode = 1
		self.lineDetection(mode, 7)
		self.rotateImage()
		self.rotateImage()
		self.rotateImage()
		self.sortLines()
		self.displayLines()

	def negateImage(self, thres = 30):
		self.image = 255 - self.image
		ret, self.image = cv.threshold(self.image, thres, 255, 0)

	# def laplacian(self, thres=150):
	# 	laplacian = cv.Laplacian(self.image, cv.CV_64F)
	# 	ret, laplacian = cv.threshold(laplacian, thres, 255, 0)
	# 	plt.imshow(laplacian, cmap='Greys_r')
	# 	plt.show()

	# def sobel(self):
	# 	image = self.image
	# 	sobelx64f = cv.Sobel(image,cv.CV_64F, 1,0,ksize=7)
	# 	abs_sobelx64f=np.absolute(sobelx64f)
	# 	sobel_8u=np.uint8(abs_sobelx64f)
	# 	plt.imshow(sobel_8u, cmap='Greys_r')
	# 	plt.show()

	def lineDetection(self, mode, minLength):
		tmp = cv.HoughLinesP(self.image, rho = 9, theta = np.pi, threshold = 10, minLineLength =minLength, maxLineGap = 0)
		
		if mode==0:
			for line in tmp:
				x1, y1, x2, y2 = list(line[0])
				lineList = x1, y2, x2, y1, abs(y2-y1)
				self.verLines.append(lineList)
		elif mode == 1:
			for line in tmp:
				x1, y1, x2, y2 = line[0]
				y1 = -y1 + self.width
				y2 = -y2 + self.width
				lineList = [y1, x2, y2, x1, abs(y2-y1)]
				self.horLines.append(lineList)

	def rotateImage(self):
		self.image = np.rot90(self.image)

	def displayLines(self, rad=2):
		for spot in self.verLines:
			cv.rectangle(self.image, (spot[0]-rad, spot[1]-rad), (spot[2]+rad, spot[3]+rad), color=(255), thickness=0)

		for spot in self.horLines:
			cv.rectangle(self.image, (spot[0]-rad, spot[1]-rad), (spot[2]+rad, spot[3]+rad), color=(255), thickness=0)

	def getLineLists(self):
		return (self.verLines, self.horLines)

	def getImage(self):
		return self.image
	
	def displayImage(self):
		plt.imshow(self.image, cmap='Greys_r')
		plt.show()

	def sortLines(self):
		self.verLines = sorted(self.verLines, key=self.lineGetKey)
		self.horLines = sorted(self.horLines, key=self.lineGetKey)

	def lineGetKey(self, item):
		return (pow(item[1], 2.0) + pow(item[0], 2.0) , item[0], item[1])


	# def removeRepeat(self):

	# 	index = 0
	# 	sinVer = []
	# 	sinHor = []
	# 	while index<len(self.verLines):
	# 		findi = index
	# 		xPos = self.verLines[index][0]
	# 		yPos = self.verLines[index][1]
	# 		while findi+1 < len(self.verLines) and abs(xPos-self.verLines[findi+1][0])<3 and yPos == self.verLines[findi+1][1]:
	# 			findi +=1
	# 		aveIndex = (findi + index)/2
	# 		sinVer.append(self.verLines[aveIndex])

	# 		index = findi + 1

	# 	index = 0
	# 	while index<len(self.horLines):
	# 		findi = index
	# 		xPos = self.horLines[index][0]
	# 		yPos = self.horLines[index][1]
	# 		while findi+1 < len(self.horLines) and abs(yPos-self.horLines[findi+1][1])<3 and xPos == self.horLines[findi+1][0]:
	# 			findi+=1
	# 		aveIndex = (findi + index) /2
	# 		sinHor.append(self.horLines[aveIndex])

	# 		index = findi + 1

	# 	print sinHor
	# 	print sinVer

	# 	self.horLines = sinHor
	# 	self.verLines = sinVer



class lineSelection():
	def __init__(verLines, horLines):
		self.verLines = verLines
		self.horLines = horLines

	def selection(thresh = 2, lenThreshRatio = 1.0/10):
		verLines = self.verLines
		horLines = self.horLines




if __name__ == '__main__':
	
	for index in range(2,6):
		i = str(index)
		filePath = "images/tree"+ i +"_wo.jpg"
		image = cv.imread(filePath,0)
		plt.imshow(image, cmap='Greys_r')
		plt.show()
		p = preProcess(image)
		p.bolding()
		p.displayImage()
		oriImage = p.returnImage()
		a = scaleCornerDetection(oriImage)
		a.parsing()

	# for index in range(1,5):
	# 	filePath = "images/tree%d_wo.jpg" %index
	# 	print filePath
	# 	image = cv.imread(filePath,0)
	# 	a = cornerDetection(image)
	# 	cv.imwrite('images/tree%d_corner.jpg', a, [cv.IMWRITE_JPEG_QUALITY, 100])

	# image = cv.imread("images/tree3.jpg", 0)

	# l = lineDetection(image)
	# l.displayImage()

	
	# a = cornerDetection(image)
	# a.displayImage()


	# b = np.array(([1, 2, 3], [4, 5, 6]))
	# for row in b:
	# 	for pixel in row:
	# 		pixel = pixel + 1

	# for row in b:
	# 	for pixel in row:
	# 		print pixel,
	# 	print


		# for index in range(len(self.downList)):
		# 	x = self.downList[index][1]
		# 	y = self.downList[index][0]
		# 	for rd in range(-self.radius, self.radius):
		# 		for cd in range(-self.radius, self.radius):
		# 			if y+rd >-1 and y+rd <self.height and x + cd>-1 and x+cd<self.width:
		# 				self.downCorner[y+rd][x+cd] = 255
		
			

		# eh, contours, hierarchy= cv.findContours(self.upCorner,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
		# cnt = contours[0]
		# cv.drawContours(self.image, [cnt], 0, (0,255,0), 3)

				# for row in range(self.height):
		# 	for col in range(self.width):
		# 		if self.upCorner[row][col] == 255:
		# 			self.upperList.append((row,col))
		# 		else:
		# 			self.upCorner[row][col] = 0

		# for item, value in upValue.items():
		# 	print item, value

		# plt.bar(upValue.keys(), upValue.values(), 1, color='g')
		# plt.show()
		# for r in range(self.radius, self.height-self.radius):
		# 	for c in range(self.radius, self.width-self.radius):
		# 		upSum = 0
		# 		downSum = 0
		# 		jointSum = 0
		# 		for rd in range(-self.radius, self.radius):
		# 			for cd in range(-self.radius,self.radius):
		# 				upSum += self.upperKernel[rd+self.radius][cd+self.radius] * self.image[r+rd][c+cd]
		# 				downSum += self.downKernel[rd+self.radius][cd+self.radius] * self.image[r+rd][c+cd]
		# 				jointSum += self.jointKernel[rd+self.radius][cd+self.radius] * self.image[r+rd][c+cd]
		# 		self.upCorner[r][c] = upSum/9
		# 		self.downCorner[r][c] = downSum/9
		# 		self.jointPt[r][c] = upSum/9
		# 		print upSum/9