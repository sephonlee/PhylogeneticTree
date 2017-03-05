import cv2 as cv
import numpy as np
import pickle
from matplotlib import pyplot as plt
import Image
import pytesseract



class treeRecover():
	def __init__(self):
		self.subTrees = []
		self.labelList = labelList
		self.cornerList = cornerList
		self.jointList = jointList
		self.horLines = horLines
		self.verLines = verLines

	def checkCorner(self):
		pass

	def extractTree(self, labelList, cornerList, jointList, horLines, verLines, w, h):
		#labelList needs to be sorted by y, (ID, location range, text, line)
		#cornerList: each corner expression should be a tuple: (type, corner location)
		tmpIndex = 0 #index for storing the subTree location
		for label in labelList:
			(ID, posRange, text, line) = label
			#check corner & line
			for cornerType, cornerPos in cornerList:
				pass


	def mergeTrees(self):
		pass






class readCorners():
	def __init__(self, fileName, upperList,  downList, jointList, labelList):
		self.image = cv.imread(fileName, 0)
		self.height, self.width = self.image.shape
		self.upperList = upperList
		self.downList = downList
		self.jointList = jointList
		self.labelList = labelList
		self.treeText = ""

	

	def findLabel(self, y):
		#return (boolen, text)

		for coverRange in self.labelList:
			y1, y2, x = coverRange

			if y >=y1 and y<y2:
				y1 = y1*2
				y2 = y2*2
				x = x*2 +1
				labelBox = self.image[y1:y2, x:x+95]
				labelBox = cv.resize(labelBox, (0,0), fx=2, fy = 2 )
				# ret, labelBox = cv.threshold(labelBox, 160, 255, 0)
				# plt.imshow(labelBox, cmap='Greys_r')
				# plt.show()
				cv.rectangle(self.image, (x, y1), (x+95, y2), color=(0), thickness=0)
				cv.imwrite("data/tmp.tiff", labelBox)
				label = pytesseract.image_to_string(Image.open('data/tmp.tiff'))
				# label = label.replace(' ', '')
				label = label.replace('-', '')
				label = label.replace(":", "")
				if label.isspace() or not label:
					print "haha", label, type(label)
					return(False, "")
				print label, type(label), label.isspace()
				return (True, label)
		
		return (False, "")

	def extractTree(self):
		# seen = []
		stack = []
		result = "("
		tmpString = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
		stringIndex = 0
		upCorner = self.upperList.pop(0)
		xPos = upCorner[1]
		yPos = upCorner[0]
		isDownCorner 	= False
		downCornerIndex = -1
		upperBound = 0
		lowerBound = 9999
		thresDist = self.height/60
		for i in range(len(self.downList)): #deal with the threePts on the same spots!!
			if self.downList[i][1] == xPos and not isDownCorner and self.downList[i][0]>yPos:
				# downCorner = self.downList.pop(i)
				lowerBound = self.downList[i][0]
				isDownCorner = True
				downCornerIndex = i

		# for i in range(downCornerIndex):
		# 	self.downList.pop(i)

		downCorner = self.downList.pop(downCornerIndex)
		# print downCorner


		stack.append((downCorner, 2))
		stack.append((upCorner, 1))

		# seen.append(downCorner)
		# seen.append(upCorner)

		#node type: 
		# 0: joint point
		# 1: upper corner open
		# 2: down corner open



		while stack:

			node = stack.pop()
			nodeType = node[1]
			
			if nodeType == 0:
				downCorner = node[2][1]
				upCorner = node[2][0]
				stack.append((downCorner, 2))
				stack.append((upCorner, 1))
				# xPos = node[0][1]
				# isUpCorner = False
				# isDownCorner = False
				# upCornerIndex = -1
				# downCornerIndex = -1
				# for i in range(len(self.downList)):
				# 	if self.downList[i][1] == xPos and not isDownCorner:
				# 		downCornerIndex = i
				# 		isDownCorner = True
				# downCorner = self.downList.pop(downCornerIndex)
				# for i in range(len(self.upperList)):
				# 	if self.upperList[i][1] == xPos and not isUpCorner:
				# 		upCornerIndex = i
				# 		isUpCorner = True
				# upCorner = self.upperList.pop(upCornerIndex)
				# if isDownCorner:
				# 	stack.append((downCorner, 2))
				# else:
				# 	pass

				# if isUpCorner:
				# 	stack.append((upCorner,1))
				# else:
				# 	pass

			elif nodeType ==1:
				result = result + "("
				stack.append((node[0], -nodeType))
				yPos = node[0][0]
				xPos = node[0][1]
				isJoint = False
				jointPtIndex = -1

				tmpDownCornerIndex = -1
				tmpUpCornerIndex = -1
				index = 0

				while index<len(self.jointList) and not isJoint:
					if self.jointList[index][0] == yPos and self.jointList[index][1]>xPos:
						isReal = 0
						yJoint, xJoint = self.jointList[index]
						isDownCorner = False
						isUpCorner = False
						tmpDownCornerIndex =-1
						tmpUpCornerIndex = -1

						for j in range(len(self.downList)):
							if self.downList[j][1] == xJoint and self.downList[j][0] > yJoint  and self.downList[j][0] < lowerBound and not isDownCorner:
								tmpDownCornerIndex = j
								isDownCorner = True
								isReal +=1
						for j in range(len(self.upperList)):
							if self.upperList[j][1] == xJoint and self.upperList[j][0] < yJoint and self.upperList[j][0] > upperBound and not isUpCorner:
								tmpUpCornerIndex = j
								isUpCorner = True
								isReal +=1

						downDist = abs(self.downList[tmpDownCornerIndex][0] - yJoint)
						upDist = abs(self.upperList[tmpUpCornerIndex][0] - yJoint)
						if isReal==2 and (downDist < thresDist or upDist < thresDist):
							if abs(downDist - upDist) <3:
								pass
							else:
								isReal = 0


						if isReal ==2:
							downCorner = self.downList.pop(tmpDownCornerIndex)
							upCorner = self.upperList.pop(tmpUpCornerIndex)
							
							if downCorner[0] > lowerBound:
								lowerBound = downCorner[0]
							jointPt = self.jointList[index]
							isJoint = True

					index +=1

				# jointPt = self.jointList.pop(jointPtIndex)

				if isJoint:
					stack.append((jointPt, 0, (upCorner, downCorner)))
				else:
					isLabel = self.findLabel(yPos)
					if isLabel[0]:
						result = result + isLabel[1]
					else:
						result = result + tmpString[stringIndex]
						stringIndex +=1			

			elif nodeType==2:
				isLowerBound = False
				stack.append((node[0], -nodeType))
				yPos = node[0][0]
				xPos = node[0][1]
				if yPos == lowerBound:
					isLowerBound = True
					lowerBound = 9999
				isJoint = False
				jointPtIndex = -1

				tmpDownCornerIndex = -1
				tmpUpCornerIndex = -1
				index = 0
				while index<len(self.jointList) and not isJoint:
					if self.jointList[index][0] == yPos  and self.jointList[index][1]>xPos:
						isReal = 0
						yJoint, xJoint = self.jointList[index]
						isDownCorner = False
						isUpCorner = False
						tmpDownCornerIndex =-1
						tmpUpCornerIndex = -1
						for j in range(len(self.downList)):
							if self.downList[j][1] == xJoint and self.downList[j][0] > yJoint  and self.downList[j][0] < lowerBound and not isDownCorner:
								tmpDownCornerIndex = j
								isDownCorner = True
								isReal +=1
						for j in range(len(self.upperList)):
							if self.upperList[j][1] == xJoint and self.upperList[j][0] < yJoint and self.upperList[j][0] > upperBound and not isUpCorner:
								tmpUpCornerIndex = j
								isUpCorner = True
								isReal +=1

						downDist = abs(self.downList[tmpDownCornerIndex][0] - yJoint)
						upDist = abs(self.upperList[tmpUpCornerIndex][0] - yJoint)

						if downDist <thresDist or upDist < thresDist:
							if abs(downDist - upDist) <3:
								pass
							else:
								isReal = 0

						if isReal ==2:
							downCorner = self.downList.pop(tmpDownCornerIndex)
							upCorner = self.upperList.pop(tmpUpCornerIndex)
							if downCorner[0] > lowerBound or isLowerBound:
								lowerBound = downCorner[0]
							jointPt = self.jointList[index]
							isJoint = True
					index+=1
				# jointPt = self.jointList.pop(jointPtIndex)
				if isJoint:
					stack.append((jointPt, 0, (upCorner, downCorner)))
				else:
					isLabel = self.findLabel(yPos)
					if isLabel[0]:
						result = result + isLabel[1]
					else:
						result = result + tmpString[stringIndex]
						stringIndex +=1
						upperBound = yPos

			elif nodeType == -1:
				result = result + ","
			elif nodeType == -2:
				result = result + ")"
		result = result + ")"
		# print self.downList
		# print self.upperList
		# print self.jointList
		self.treeText = result

	# def buildLists(self):

	# 	self.verLines = self.readLists("%s_verLines.p"%self.fileName)
	# 	self.horLines = self.readLists("%s_horLines.p"%self.fileName)
	# 	self.upperList = self.readLists("%s_upperList.p"%self.fileName)
	# 	self.downList = self.readLists("%s_downList.p"%self.fileName)
	# 	self.jointList = self.readLists("%s_jointList.p"%self.fileName)

	# def inputLists(self, upperList, downList, jointList):
	# 	self.upperList = upperList
	# 	self.downList = downList
	# 	self.jointList = jointList

	# def readLists(self,fileName):
	# 	with open(fileName, 'rb') as f:
	# 		tmpList = pickle.load(f)

	# 	return tmpList

	def getTreeText(self):
		return self.treeText

	def displayImage(self):
		plt.imshow(self.image, cmap='Greys_r')
		plt.show()



if __name__ == '__main__':


	fileName = 'images/tree3'
	r = readCorners(fileName)
	r.extractTree()
	print r.getTreeText()

	# fileName = ['verLines', 'horLines', 'upperList', 'downList', 'jointList']
	# index = 0
	# lists = [verLines, horLines, upperList, downList, jointList]
	# for name in fileName:
	# 	path = "images/tree%d_%s.p" %(i, name)
	# 	writeList(lists[index], path)
	# 	index +=1