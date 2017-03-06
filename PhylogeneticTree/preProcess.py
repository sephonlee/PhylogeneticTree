import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

class preProcess():
	def __init__(self, image):
		self.image = image
		(self.height, self.width) = self.image.shape
		print self.image.shape


	def sobelFilter(self, sigma=3, k=5):
		x = cv.GaussianBlur(self.image,(k,k),sigma)
		sobelx = cv.Sobel(self.image, cv.CV_64F, 1,0, ksize =5)
		sobely = cv.Sobel(self.image, cv.CV_64F, 0,1, ksize =5)
		plt.subplot(1,2,1), plt.imshow(sobelx, cmap = 'gray')
		plt.subplot(1,2,2), plt.imshow(sobely, cmap = 'gray')
		plt.show()

	def gaussianBlur(self, kernelSize = (3,3), sigma= 1.0):
		self.image = cv.GaussianBlur(self.image, kernelSize, sigma)

	def bolding(self, thres=180):
		ret, self.image = cv.threshold(self.image, thres, 255,3)
	
	def bilateralFilter(self, radius = 3, sigmaI = 30.0, sigmaS = 3.0):
		self.image = cv.bilateralFilter(self.image, radius, sigmaI, sigmaS)

	def downSample(self, parameter = 500.0):
		if self.height>700:
			if self.height>1000:
				parameter = 700.0
			ratio = (parameter+0.0)/self.height
			self.image = cv.resize(self.image, (0,0), fx=ratio, fy = ratio )
			self.height, self.width = self.image.shape

	def erosionKernel(self):

		kernel1 = np.array(([0,0,1,0,0], [0,0,1,0,0], [1,1,1,1,1], [0,0,1,0,0], [0,0,1,0,0]), np.uint8)
		kernel2 = np.array(([1,1,1], [1,1,1], [1,1,1]), np.uint8)
		kernel3= np.array(([1,1]), np.uint8)
		return kernel2

	def erosion(self):
		# size = 3
		# radius = size/2
		kernel = self.erosionKernel()
		image = self.image
		image = cv.erode(image, kernel, anchor = (1,1), iterations=1)
		image = cv.dilate(image, kernel, anchor = (1,1), iterations=1)

		plt.imshow(image, cmap='Greys_r')
		plt.show()



	def thinningIteration(self, thinImage, iteration):
		marker = np.zeros((3,3), np.uint8)

		for row in range(1,self.height-1):
			for col in range(1,self.width-1):
				marker = thinImage[row-1:row+2, col-1:col+2]
				pixel = thinImage[row][col]
				if cv.countNonZero(marker)>0:
					b = cv.sumElems(marker) - pixel
					a = 0
					if marker[0][1] - marker[0][0] == 1:
						a+=1
					if marker[0][2] - marker[0][1] == 1:
						a+=1
					if a<2 and marker[1][2] - marker[0][2] == 1:
						a+=1
					if a<2 and marker[2][2] - marker[1][2] == 1:
						a+=1
					if a<2 and marker[2][1] - marker[2][2] == 1:
						a+=1
					if a<2 and marker[2][0] - marker[2][1] == 1:
						a+=1
					if a<2 and marker[1][0] - marker[2][0] == 1:
						a+=1
					if a<2 and marker[0][0] - marker[1][0] == 1:
						a+=1

					if iteration == 0:
						#2,4,6 -- 2,4,8
						c = marker[0][1]*marker[1][2]*marker[2][1]
						d = marker[0][1]*marker[1][2]*marker[0][1]
					else:
						#4,6,8 -- 2,6,8
						c = marker[1][2]*marker[2][1]*marker[0][1]
						d = marker[0][1]*marker[2][1]*marker[0][1]

					if a==1 and b[0]>=2 and b[0]<=6 and c==0 and d == 0:
						thinImage[row][col] = 1

	def thinning(self):
		thinImg = self.image.copy()
		
		thinImg = thinImg / 255
		prev = np.zeros((self.height, self.width), np.uint8)
		diff = np.ones((self.height, self.width), np.uint8)
		while cv.countNonZero(diff) > 0:
			self.thinningIteration(thinImg, 0)
			self.thinningIteration(thinImg, 1)
			cv.absdiff(thinImg, prev, diff)

			prev = thinImg.copy()

		thinImg = thinImg * 255
		
		self.image = thinImg.copy()

	def threshold(self, thres=190):
		ret, self.image = cv.threshold(self.image, thres, 255, 0)

	def displayImage(self):
		plt.imshow(self.image, cmap='Greys_r')
		plt.show()

	def returnImage(self):
		return self.image


if __name__=='__main__':
	
	for i in range(1,5):
		filePath = "images/tree%d_wo.jpg" %i
		image = cv.imread(filePath, 0)
		a=preProcess(image)
		a.downSample()
		# a.sobelFilter()
		a.bolding()
		a.thinning()
		a.displayImage()


	# filePath = "images/tree2_wo.jpg"
	# image = cv.imread(filePath,0)
	# a = preProcess(image)
	# # a.downSample()
	# a.bolding()
	# a.erosion()

