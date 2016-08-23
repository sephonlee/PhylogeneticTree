import cv2 as cv
import numpy as np
import json
from matplotlib import pyplot as plt
from pprint import pprint



class getJson():
	def __init__(self, filePath):
		self.filePath = filePath

	def decodeJson(self):
		# with open("%s" %self.filePath) as jsonFile:

		# 	result = json.load(jsonFile)

		f = open("%s" %self.filePath)
		result = json.load(f)
		f.close()

		return result


class textRemover():
	def __init__(self):
		pass

	def textRemove(self, image, jsonData):
		self.image = image
		newJson = jsonData['regions']

		for lines in newJson:
			line = lines['lines']
			for data in line:
				pos = data['boundingBox']
				pos = pos.split(',')
				pos = map(int, pos)
				(x,y,w,h) = pos
				cv.rectangle(self.image, (x, y), (x+w, y+h), color=(0))

			
	def displayImage(self):
		plt.imshow(self.image, cmap='Greys_r')
		plt.show()	




if __name__ == '__main__':
	
	# for i in range(1,6):

	# 	image = cv.imread("images/tree%d.jpg" %i, 0) 
	# 	jsonPath = "textPos/tree%d_textPos.json" %i
	# 	a = getJson(jsonPath)
	# 	json = a.decodeJson()

	# 	b = textRemover()
	# 	b.textRemove(image, json)
	# 	b.displayImage()
	for i in range(2,25):
		index = str(i)
		fileName = 'tree' + index 


		image = cv.imread("images/" + fileName + ".jpg", 0)
		jsonPath = 'textPos/' + fileName + '_textPos.json'

		a = getJson(jsonPath)
		jsonData = a.decodeJson()

		b = textRemover()
		b.textRemove(image, jsonData)
		b.displayImage()



##################package process#####################

	# for index in range(1,25):
	# 	i = str(index)

	# 	fileName = 'tree' + i


	# 	image = cv.imread("images/" + fileName + ".jpg", 0)
	# 	jsonPath = 'textPos/' + fileName + '_textPos.json'
	# 	a = getJson(jsonPath)
	# 	json = a.decodeJson()

	# 	b = textRemover()
	# 	b.textRemove(image, json)
	# 	b.displayImage()


########################################################3
	# bounding = json['regions'][0]['lines'][0]['boundingBox']
	# print bounding
	# bounding = bounding.split(',')
	# bounding = map(int, bounding)
	# (x, y, width, height) = bounding
	

	
	# cv.rectangle(image, (x, y), (x+width, y+height), color=(0))





