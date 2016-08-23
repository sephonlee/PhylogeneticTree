import time
import requests
import cv2 
import numpy as np
import json
from matplotlib import pyplot as plt

class textDetection():
	def __init__(self):
		self.url = 'https://api.projectoxford.ai/vision/v1.0/ocr'
		self.key = '8d62e2bed27840579534554c04f9b721'
		self.maxNumRetries = 10
		# self.filePath = filePath
		# if(mode == 'disk'):
		# 	result = self.imageOnDisk()
		# elif(mode == 'online'):
		# 	result = self.imageOnline()

		# return result


	def processRequest(self, json, data, headers, params=None):
		retries = 0
		result = None

		while True:

			response = requests.request( 'post', self.url, data = data, headers = headers, params = params )
			if response.status_code == 429: 

				print "Message: %s" % ( response.json()['error']['message'] )

				if retries <= _maxNumRetries: 
					time.sleep(1) 
					retries += 1
					continue
				else: 
					print 'Error: failed after retrying!'
					break

			elif response.status_code == 200 or response.status_code == 201:

				if 'content-length' in response.headers and int(response.headers['content-length']) == 0: 
					result = None 
				elif 'content-type' in response.headers and isinstance(response.headers['content-type'], str): 
					if 'application/json' in response.headers['content-type'].lower(): 
						result = response.json() if response.content else None 
					elif 'image' in response.headers['content-type'].lower(): 
						result = response.content
			else:
				print "Error code: %d" % ( response.status_code )
				print "Message: %s" % ( response.json()['error']['message'] )

			break

		return result

	def renderResultOnImage(self, result, img):
		R = int(result['color']['accentColor'][:2],16)
		G = int(result['color']['accentColor'][2:4],16)
		B = int(result['color']['accentColor'][4:],16)

		cv2.rectangle( img,(0,0), (img.shape[1], img.shape[0]), color = (R,G,B), thickness = 25 )

		if 'categories' in result:
			categoryName = sorted(result['categories'], key=lambda x: x['score'])[0]['name']
			cv2.putText( img, categoryName, (30,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3 )


	def imageOnline(self, urlImage):
		params = {'visualFeatures': 'Color, Categories'}

		headers = dict()
		headers['Ocp-Apim-Subscription-Key'] = self.key
		headers['Content-Type'] = 'application/json' 

		json = {'url':urlImage}
		data = None

		result = processRequest(json, data, headers, params)

		arr = np.asarray(bytearray(requests.get(urlImage).content), dtype=np.uint8)
		img = cv2.cvtColor(cv2.imdecode(arr,-1), cv2.COLOR_BGR2RGB)

		renderResultOnImage (result,img)
		ig,ax = plt.subplots(figszize=(15,20))
		ax.imshow(img)

	def imageOnDisk(self, filePath):

		with open(filePath, 'rb') as f:
			data = f.read()

		params = {'visualFeatures': 'unk'}

		headers = dict()
		headers['Ocp-Apim-Subscription-Key'] = self.key
		headers['Content-Type'] = 'application/octet-stream'

		json=None

		result = self.processRequest(json, data, headers, params)

		# arr = np.asarray(bytearray(requests.get(urlImage).content), dtype=np.uint8)
		# img = cv2.cvtColor(cv2.imdecode(arr,-1), cv2.COLOR_BGR2RGB)

		# self.renderResultOnImage(result,img)
		# ig,ax = plt.subplots(figsize=(15,20))
		# ax.imshow(img) 
		# print result
		return result


class preProcess():
	def __init__(self,filePath):
		self.image = cv2.imread(filePath, 0)
		(self.height, self.width) = self.image.shape
		print self.height, self.width

	def padding(self, ratio=10):
		paddingWidth = self.width/ratio
		paddingHeight = self.height/ratio
		y = np.ones((self.height + 2*paddingHeight, self.width + 2*paddingWidth)) * 255
		h, w = y.shape
		print h,w
		# y[:,0:paddingWidth] = 255
		# y[:,self.width-paddingWidth : self.width] = 255
		# y[0:paddingHeight, :] = 255
		# y[self.height-paddingHeight:self.height, :] = 255

		y[paddingHeight:h-paddingHeight, paddingWidth:w-paddingWidth] = self.image.copy()
		self.image = y.copy()

		# self.image = cv2.copyMakeBorder(self.image,paddingHeight,paddingHeight,paddingWidth,paddingWidth,cv2.BORDER_CONSTANT,value=255)

	def resize(self):
		pass

	def negate(self, thres=40):
		self.image = 255 - self.image
		ret, self.image = cv2.threshold(self.image, thres, 255, 0)
		self.image = 255 - self.image
		
	def copy(self):
		y = self.image.copy()
		# print type(y)
		return y

	def returnImage(self):
		return self.image




if __name__ == '__main__':


	# filePath = 'test.png'

	# a = preProcess(filePath)
	# # a.negate()
	# a.padding()
	# image = a.returnImage()
	# # image  = a.copy()
	# cv2.imwrite('images/tree2_parsed.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 100])


	# for i in range(1,6):
	# 	filepath = 'images/tree%d.jpg' %i
	# 	a = textDetection()
	# 	result = a.imageOnDisk(filePath)
	# 	with open('textPos/tree%d_textPos.json'%i, 'w') as outfile:
	# 		json.dump(result, outfile)


	for index in range(1, 2):
		i = str(index)
		fileName = 'tree' + i
		filePath = 'images/' + fileName + '.jpg'

		a = textDetection()
		result = a.imageOnDisk(filePath)
		with open('textPos/' + fileName + '_textPos.json', 'w') as outfile:
			json.dump(result, outfile)


	

