# Editor: Sean Yang, Poshen Lee
from featureDetection import cornerDetection 
from featureDetection import lineDetection
from featureDetection import scaleCornerDetection
from preProcess import preProcess 
from readCorners import readCorners
# import textDetector
# import textRemover
import cv2 as cv
import numpy as np
import pickle
from matplotlib import pyplot as plt
from ete3 import Tree
import operator
import random
from os import listdir
from os.path import isfile, join



globalParent = []
globalChildren = []


def sortByLeftTop(item):
	criteria = pow(item[0],2) + pow(item[1], 2)
	return criteria, item[2]-item[0]

def sortByBtmRight(item):
	criteria = pow(item[2], 2) + pow(item[3],2)
	return -criteria, item[3] - item[1]

def sortByArea(item):
	return -item.area

def sortByY(item):
	return item[1]

def sortNodeByLeftEnd(item):
	return item.branch[0]

def isDotWithinLine(dot, line):
	margin = 5
	x, y = dot
	x1, y1, x2, y2, length = line
	if x1-margin < x and x < x2+margin and y1 - margin < y and y < y2 + margin:
		return True
	else:
		return False

def isLefter(branch, ref):
	x1 = branch[0]
	x2 = ref[0]

	if x2 < x1:
		return True
	return False

# def getNodeBranchOnTheLeft(nodeList, branch):
# 	potentialNodes = []
# 	x1, y1, x2, y2, length = branch
# 	for node in nodeList:
# 		x, y, xx, yy, length2 = node.branch
# 		if 

def getNodeBranchOnTheRight(dot, nodeList):
	x, y = dot
	potentialNodes = []
	for node in nodeList:
		x1, y1, x2, y2, length = node.branch
		if y1 < y and y2 > y and x1>x:
			potentialNodes.append(node)
	if len(potentialNodes) == 0:
		return False
	else:
		potentialNodes = sorted(potentialNodes, key = sortNodeByLeftEnd)
		return potentialNodes[0]


def fixTrees(rootList, parent, children, horLines, verLines):

	breakNodeList = []
	tmpList = rootList[:]
	for node in rootList:
		if node in tmpList:
			if not node.isComplete:
				for breakNode in node.breakSpot:
					print breakNode.branch, breakNode.to[0], breakNode.to[1]
					isFixed = False
					isUpper = True
					if not ((breakNode.to[0] or breakNode.upperLeave) or (breakNode.to[1] or breakNode.lowerLeave)):
						pass
					elif (breakNode.to[0] or breakNode.upperLeave) and not (breakNode.to[1] or breakNode.lowerLeave):
						print "upper"
						x1, y1, x2, y2, length = breakNode.branch
						result = getNodeBranchOnTheRight((x2,y2), rootList)
						if result:	
							to = list(breakNode.to)
							to[1] = result
							breakNode.to = tuple(to)
							result.whereFrom = breakNode
							result.origin = node
							if result.isComplete:
								print "remove"
								if result in tmpList:
									tmpList.remove(result)
							node.breakSpot.remove(breakNode)
							isFixed = True
							print result.branch
						else:
							isUpper = False

					elif not (breakNode.to[0] or breakNode.upperLeave) and (breakNode.to[1] or breakNode.lowerLeave):
						x1, y1, x2, y2, length = breakNode.branch
						result = getNodeBranchOnTheRight((x1,y1), rootList)
						if result:
							to = list(breakNode.to)
							to[0] = result
							breakNode.to = tuple(to)
							node.breakSpot.remove(breakNode)
							result.origin = node
							if result.isComplete:
								print "remove"
								if result in tmpList:
									tmpList.remove(result)
							result.whereFrom = breakNode
							isFixed = True
							print result.branch


					if isUpper:
						breakSpot = 'upper'
					else:
						breakSpot = 'lower'



			else:
				pass
	rootList = tmpList[:]
	for node in rootList:
		if node in tmpList:
			print node.branch, node.breakSpot, node.whereFrom
			if len(node.breakSpot) == 0 and node.whereFrom != None:
				print "hahahahahaha"
				tmpList.remove(node)
	rootList = tmpList[:]
	if len(rootList) == 1:
		rootList[0].isComplete = True


			



def checkDone(rootList):
	isDone = True
	rootNode = rootList[0]

	if not rootNode.isComplete:
		return False

	if rootNode.root:
		x1, y1, x2, y2, length = rootNode.root
		for node in rootList:
			if node != rootNode:
				if isDotWithinLine((x1, y1), node.branch) or isLefter(rootNode.branch, node.branch):
					return False
	else:
		for node in rootList:
			if node != rootNode:
				if isLefter(rootNode.branch, node.branch):
					return False

	return isDone

def removeFakeTree(rootList):
	for rootNode in rootList:

		if rootNode.area < 100 and rootNode.numNodes !=1:
			rootList.remove(rootNode)

	return rootList

def writeList(writeList, filePath):
	with open(filePath, 'wb') as f:
		pickle.dump(writeList, f)

def treeRecover(rootNode):

	return rootNode.getChildren()
	# defaultString = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	# stringIndex = 0
	# stack = []
	# result = ""
	# if rootNode.isUpperAnchor:
	# 	result = result + "("
	# 	if rootNode.upperLabel:
	# 		pass
	# 	else:
	# 		result = result + defaultString[stringIndex]
	# 		stringIndex +=1
	# else:
	# 	result = result + "("
	# 	rootNode.status = 1
	# 	stack.append(rootNode)
	# 	stack.append(rootNode.to[0])

	
	# while stack:
	# 	node = stack.pop()
	# 	if node.status == 0:
	# 		result = result + "("
	# 		node.status = 1
	# 		stack.append(node)
	# 		if node.isUpperAnchor:
				
	# 			if node.upperLabel:
	# 				pass
	# 			else:
	# 				result = result + defaultString[stringIndex]
	# 				stringIndex+=1
	# 		else:
	# 			stack.append(node.to[0])
	# 	elif node.status == 1:
	# 		result = result + ","
	# 		node.status = 2
	# 		stack.append(node)
	# 		if node.isLowerAnchor:
	# 			if node.lowerLabel:
	# 				pass
	# 			else:
	# 				result = result + defaultString[stringIndex]
	# 				stringIndex+=1
	# 		else:
	# 			stack.append(node.to[1])

	# 	elif node.status == 2:
	# 		result = result + ")"

	# return result


def isSameLine(aline, bline, margin = 5):
	ax1, ay1, ax2, ay2, alength = aline
	bx1, by1, bx2, by2, blength = bline

	if ay1 - margin < by1 and ay2 + margin> by2 and ax1 - margin < bx1 and ax2 + margin > bx2 and alength + margin > blength:
		return True
	elif by1 - margin < ay1 and by2 + margin > ay2 and bx1 - margin < ax1 and bx2 + margin > ax2 and blength + margin > alength:
		return True
	else:
		return False

def display(image, nodeList, cornerList):
	image = cv.cvtColor(image,cv.COLOR_GRAY2RGB)
	whatever=image.copy()
	whatever = displayNodesGroup(whatever, nodeList)
	# image = displayCorners(image, cornerList)
	plt.imshow(whatever)
	plt.show()


def displayCorners(image, cornerList):
	if len(image.shape) == 2:
		image = cv.cvtColor(image,cv.COLOR_GRAY2RGB)

	upperCorners, lowerCorners, jointPoints = cornerList
	rad = 5
	for x, y in jointPoints:
		cv.rectangle(image, (x-rad, y - rad), (x + rad, y +rad), color=(0, 255, 0), thickness=1)
	for x, y in upperCorners:
		cv.rectangle(image, (x-rad, y - rad), (x + rad, y +rad), color=(0, 255, 0), thickness=1)
	for x, y in lowerCorners:
		cv.rectangle(image, (x-rad, y - rad), (x + rad, y +rad), color=(0, 255, 0), thickness=1)

	return image

def displayRecoveredTree(image, rootNode):
	if len(image.shape) ==2:
		whatever = image.copy()
		whatever = cv.cvtColor(whatever, cv.COLOR_GRAY2RGB)

	color = (0,255,0)

	stack = []
	stack.append(rootNode)


	while stack:
		node = stack.pop()
		if node.to[0]:
			stack.append(node.to[0])
		if node.to[1]:
			stack.append(node.to[1])

		if node.root:
			x1, y1, x2, y2, length = node.root
			cv.rectangle(whatever , (x1, y1), (x2, y2), color=color, thickness=2)
		if node.branch:
			x1, y1, x2, y2, length = node.branch
			cv.rectangle(whatever , (x1, y1), (x2, y2), color=color, thickness=2)
		if node.upperLeave:
			x1, y1, x2, y2, length = node.upperLeave
			cv.rectangle(whatever , (x1, y1), (x2, y2), color=color, thickness=2)
		if node.lowerLeave:
			x1, y1, x2, y2, length = node.lowerLeave
			cv.rectangle(whatever , (x1, y1), (x2, y2), color=color, thickness=2)


	plt.imshow(whatever)
	plt.show()

def displayNodesGroupByOrder(image, nodeList):
	if len(image.shape) ==2:
		whatever = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
	for rootNode in rootList:
		stack = []
		stack.append(rootNode)

		image = whatever.copy()
		while stack:

			color = (0, 255, 0)
			node = stack.pop()
			if node.to[0]:
				stack.append(node.to[0])
			if node.to[1]:
				stack.append(node.to[1])				
			if node.root:
				x1, y1, x2, y2, length = node.root
				cv.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=2)
			if node.branch:
				x1, y1, x2, y2, length = node.branch
				cv.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=2)
			if node.upperLeave:
				x1, y1, x2, y2, length = node.upperLeave
				cv.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=2)
			if node.lowerLeave:
				x1, y1, x2, y2, length = node.lowerLeave
				cv.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=2)
		plt.imshow(image)
		plt.show()

def displayNodesGroup(image, rootList):
	if len(image.shape) ==2:
		whatever = image.copy()
		whatever = cv.cvtColor(whatever, cv.COLOR_GRAY2RGB)
	else:
		whatever = image.copy()
	count = 0
	for rootNode in rootList:

		stack = []
		stack.append(rootNode)
		if count%3 == 0:
			color = (255, 0 , 0)
		elif count%3 == 1:
			color = (0, 255, 0)
		else:
			color = (0, 0, 255)
		while stack:
			node = stack.pop()
			if node.to[0]:
				stack.append(node.to[0])
			if node.to[1]:
				stack.append(node.to[1])
			if not node.isBinary:
				for to in node.otherTo:
					if to:
						stack.append(to)
			if node.root:
				x1, y1, x2, y2, length = node.root
				cv.rectangle(whatever , (x1, y1), (x2, y2), color=color, thickness=2)
			if node.branch:
				x1, y1, x2, y2, length = node.branch
				cv.rectangle(whatever , (x1, y1), (x2, y2), color=color, thickness=2)
			if node.upperLeave:
				x1, y1, x2, y2, length = node.upperLeave
				cv.rectangle(whatever , (x1, y1), (x2, y2), color=color, thickness=2)
			if node.lowerLeave:
				x1, y1, x2, y2, length = node.lowerLeave
				cv.rectangle(whatever , (x1, y1), (x2, y2), color=color, thickness=2)
			if not node.isBinary:
				for line in node.interLeave:
					x1, y1, x2, y2, length = line
					cv.rectangle(whatever , (x1, y1), (x2, y2), color=color, thickness=2)

		count +=1
	return whatever



def displayNodes(image,nodeList):
	if len(image.shape) ==2:
		image = cv.cvtColor(image,cv.COLOR_GRAY2RGB)
	count = 0
	for node in nodeList:

		if count%3 == 0:
			color = (255, 0 , 0)
		elif count%3 == 1:
			color = (0, 255, 0)
		else:
			color = (0, 0, 255)

		if node.root:
			x1, y1, x2, y2, length = node.root
			cv.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=2)
		if node.branch:
			x1, y1, x2, y2, length = node.branch
			cv.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=2)
		if node.upperLeave:
			x1, y1, x2, y2, length = node.upperLeave
			cv.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=2)
		if node.lowerLeave:
			x1, y1, x2, y2, length = node.lowerLeave
			cv.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=2)
		if not node.isBinary:
			for line in node.interLeave:
				x1, y1, x2, y2, length = line
				cv.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=2)
		count +=1
	plt.imshow(image)
	plt.show()
	# return image

# def connectParentAndChildren(parent, children):

# 	for parentNode in parent:
# 		if not parentNode.to:	
# 			for childrenNode in children:
# 				if isSameLine(parentNode.branch, childrenNode.branch):
# 					parentNode.to = childrenNode
# 					childrenNode.fromRoot = parentNode
# 	for childrenNode in children:
# 		if not childrenNode.upperTo and not childrenNode.upperTo:
# 			for parentNode in parent:
# 				if isSameLine(childrenNode.upperLeave, parentNode.root):
# 					childrenNode.upperTo = parentNode
# 					parentNode.whereFrom = childrenNode
# 				elif isSameLine(childrenNode.lowerLeave, parentNode.root):
# 					childrenNode.lowerTo = parentNode
# 					parentNode.whereFrom = childrenNode


def checkError(node, mode , anchorLines):

	if mode == 'upper':
		isAnchorLine = False
		if node.upperLeave:
			for line in anchorLines:
				if isSameLine(line, node.upperLeave):
					node.isUpperAnchor = True
					isAnchorLine = True
					return True
			if not isAnchorLine:
				for package in globalParent:
					lines, dist = package

					if isSameLine(lines[0], node.upperLeave):
						newNode = Node(node.upperLeave, lines[1])
						nodeTo = list(node.to)
						nodeTo[0] = newNode
						node.to = tuple(nodeTo)

				return False
		else:
			if node.isRoot:
				node.breakSpot.append(node)
			else:
				rootNode = node.origin
				rootNode.breakSpot.append(node)
			return False

	elif mode == 'lower':
		isAnchorLine = False
		if node.lowerLeave:
			for line in anchorLines:
				if isSameLine(line, node.lowerLeave):
					node.isLowerAnchor = True
					isAnchorLine = True
					return True
			if not isAnchorLine:
				for package in globalParent:
					lines, dist = package
					if isSameLine(lines[0], node.lowerLeave):
						newNode = Node(node.lowerLeave, lines[1])
						nodeTo = list(node.to)
						nodeTo[1] = newNode
						node.to = tuple(nodeTo)
						return False
		else:
			if node.isRoot:
				node.breakSpot.append(node)
			else:
				rootNode = node.origin
				rootNode.breakSpot.append(node)
			return False
	elif mode[:5] == 'inter':
		isAnchorLine = False
		index = int(mode[5])
		print index
		if node.interLeave[index]:
			for line in anchorLines:
				if isSameLine(line, node.interLeave[index]):
					node.isInterAnchor[index] = True
					isAnchorLine = True
					return True
			if not isAnchorLine:
				for package in globalParent:
					lines, dist = package
					if isSameLine(lines[0], node.interLeave[index]):
						newNode = Node(node.interLeave[index], lines[1])
						node.otherTo[index] = newNode
						return False
		else:
			if node.isRoot:
				node.breakSpot.append(node)
			else:
				rootNode = node.origin
				rootNode.breakSpot.append(node)
			return False

def countArea(lineList):

	lineList = sorted(lineList, key = sortByLeftTop)

	leftTop = lineList[0]
	x1 = leftTop[0]
	y1 = leftTop[1]
	lineList = sorted(lineList, key = sortByBtmRight)
	btmRight = lineList[0]
	x2 = btmRight[2]
	y2 = btmRight[3]
	area = abs(y2-y1) *abs(x2-x1) 
	if area==0:
		x1 = leftTop[2]
		y1 = leftTop[3]
		x2 = btmRight[2]
		y2 = btmRight[3]
		area = abs(y2-y1) *abs(x2-x1) 

	return area


def groupNodes(rootNode, seen, anchorLines):
	stack = []
	visit = []
	lineList = []
	visit.append(rootNode)
	lineList.append(rootNode.branch)
	rootNode.origin = rootNode
	isComplete = True
	if rootNode.to[0]:
		if rootNode.branch != rootNode.to[0].branch:
			stack.append(rootNode.to[0])
		else:
			tmpNode = list(rootNode.to)
			tmpNode[0] = None
			rootNode.to = tuple(tmpNode)			
	else:
		isAnchorLine = checkError(rootNode, 'upper', anchorLines)
		if isAnchorLine:
			lineList.append(rootNode.upperLeave)
		else:
			if rootNode.to[0]:
				stack.append(rootNode.to[0])
			else:
				isComplete = False
				lineList.append(rootNode.branch)
	if rootNode.to[1]:
		if rootNode.branch != rootNode.to[1].branch:
			stack.append(rootNode.to[1])
		else:
			tmpNode = list(rootNode.to)
			tmpNode[1] = None
			rootNode.to = tuple(tmpNode)
	else:
		isAnchorLine = checkError(rootNode, 'lower', anchorLines)
		if isAnchorLine:
			lineList.append(rootNode.lowerLeave)
		else:
			if rootNode.to[1]:
				stack.append(rootNode.to[1])
			else:
				isComplete = False
				lineList.append(rootNode.branch)
	if not rootNode.isBinary:
		for index, to in enumerate(rootNode.otherTo):
			if to:
				if rootNode.branch != to.branch:
					stack.append(to)
				else:
					rootNode.otherTo[index] = None
			else:
				isAnchorLine = checkError(rootNode, 'inter%s' %str(index), anchorLines)
				if isAnchorLine:
					anchorLines.append(rootNode.interLeave[index])
					lineList.append(rootNode.interLeave[index])
				else:
					if rootNode.otherTo[index]:
						stack.append(rootNode.otherTo[index])
					else:
						if isComplete:
							isComplete = False
							lineList.append(rootNode.branch)
	numNodes = 1

	while stack:
		numNodes +=1
		node = stack.pop()
		visit.append(node)
		node.origin = rootNode
		if node.to[0] :

			if node.to[0] not in seen:
				seen.append(node.to[0])
			if node.to[0] not in visit and node.branch != node.to[0].branch:
				stack.append(node.to[0])
			else:
				loop = [True, node]
				tmpNode = list(node.to)
				tmpNode[0] = None
				node.to = tuple(tmpNode)
				return (seen, loop)
		else:
			if not checkError(node,'upper',anchorLines):
				isComplete = False
				if node.to[0]:
					stack.append(node.to[0])
				else:
					lineList.append(node.branch)
			else:
				lineList.append(node.upperLeave)
			

		if node.to[1]:

			if node.to[1] not in seen:
				seen.append(node.to[1])
			if node.to[1] not in visit and node.branch != node.to[1].branch:

				stack.append(node.to[1])
			else:


				loop = [True, node]
				tmpNode = list(node.to)
				tmpNode[1] = None
				node.to = tuple(tmpNode)

				return (seen, loop)
		else:

			if not checkError(node, 'lower', anchorLines):
				isComplete = False
				if node.to[1]:
					stack.append(node.to[1])
				else:
					lineList.append(node.branch)
			else:
				lineList.append(node.lowerLeave)

		if not node.isBinary:
			for index, to in enumerate(node.otherTo):

				if to:
					if to not in seen:
						seen.append(to)
					if to not in visit and node.branch != to.branch:
						stack.append(to)
					else:
						node.otherTo[index] = None
				else:
					isAnchorLine = checkError(node, 'inter%s' %str(index), anchorLines)
					if isAnchorLine:
						anchorLines.append(node.interLeave[index])
						lineList.append(node.interLeave[index])
					else:
						if node.otherTo[index]:
							stack.append(node.otherTo[index])
						else:
							if isComplete:
								isComplete = False
								lineList.append(node.branch)


	rootNode.numNodes = numNodes

	area = countArea(lineList)
	rootNode.area = area


	if isComplete:
		rootNode.isComplete = True
	loop = False, None
	return (seen, loop)


def getRootList(nodeList, anchorLines):
	seen = []
	stack = []
	rootList = []
	for node in nodeList:
		if node not in seen:
			rootNode = None
			stack.append(node)
			foundRoot = False
			while stack:

				subnode = stack.pop()
				if subnode in seen:
					break				
				else:
					seen.append(subnode)
					if subnode.whereFrom:
						stack.append(subnode.whereFrom)
					else:
						foundRoot = True
						subnode.isRoot = True
						rootNode = subnode
						rootList.append(subnode)
			if foundRoot:
				(seen, loop) = groupNodes(rootNode, seen, anchorLines)
				# isLoop = loop[0]
				# if isLoop:
				# 	rootList.remove(rootNode)
					# removeNode = loop[1]
					# previousNode = removeNode.whereFrom
					# to1, to2 = previousNode.to
					# if to1 == removeNode:
					# 	previousNode.to = (None, to2)
					# if to2 == removeNode:
					# 	previousNode.to = (to1, None)




	return rootList



def createNodes(parent, children):
	nodeList = []
	for item in children:
		lines, dist = item
		(branch, hlines) = lines
		match = False
		for pitem in parent:
			((root, pbranch), pdist) = pitem
			if isSameLine(branch, pbranch):
				match = True
				if len(hlines) <= 2:
					upperLeave, lowerLeave = hlines
					a = Node(root, branch, upperLeave, lowerLeave)					
				else:
					hlines = list(hlines)
					interLine = []
					for index, line in enumerate(hlines):
						
						if index== 0:
							upperLeave = line
						elif index == 1:
							lowerLeave = line
						else:
							interLine.append(line)
					a = Node(root,branch,upperLeave,lowerLeave)

					a.interLeave = interLine
					numInterLeave = len(hlines) - 2
					for index in range(numInterLeave):
						a.otherTo.append(None)
						a.isInterAnchor.append(False)
						a.interLabel.append(None)
					a.isBinary = False
				nodeList.append(a)

		if not match:
			if len(hlines) <=2:
				upperLeave, lowerLeave = hlines
				a = Node(None, branch ,upperLeave, lowerLeave)
			else:
				hlines = list(hlines)
				for index, line in enumerate(hlines):
					interLine = []
					if index == 0:
						upperLeave = line
					elif index == 1:
						lowerLeave = line
					else:
						interLine.append(line)
				a = Node(root, branch, upperLeave, lowerLeave)
				a.interLeave = interLine
				numInterLeave = len(hlines) - 2
				for index in range(numInterLeave):
					a.interLabel.append(None)
					a.otherTo.append(None)
					a.isInterAnchor.append(False)

				a.isBinary = False

			nodeList.append(a)

	for node in nodeList:
		if node.root:
			for subNode in nodeList:
				if subNode != node and subNode.branch != node.branch:					
					if subNode.upperLeave:
						if isSameLine(node.root, subNode.upperLeave):
							node.whereFrom = subNode
							tmp = list(subNode.to)
							tmp[0] = node
							subNode.to = tuple(tmp)
							break
					if subNode.lowerLeave:
						if isSameLine(node.root, subNode.lowerLeave):
							node.whereFrom = subNode
							tmp = list(subNode.to)
							tmp[1]= node
							subNode.to = tuple(tmp)
							break
					if not subNode.isBinary :
						for index, line in enumerate(subNode.interLeave):
							if isSameLine(node.root, line):
								node.whereFrom = subNode
								subNode.otherTo[index] = node
								break



	return nodeList






# def createParentNodes(parent):
# 	newParent = []
# 	for item in parent:
# 		lines, dist = item
# 		root, branch = lines
# 		a = parentNode(root, branch)
# 		newParent.append(a)

# 	return newParent

# def createChildrenNodes(children):
# 	newChildren = []
# 	for item in children:
# 		lines, dist = item
# 		(branch, (upperLeave, lowerLeave)) = lines

# 		a = childrenNode(branch, upperLeave, lowerLeave)
# 		newChildren.append(a)

# 	return newChildren

class Node():
	def __init__(self, root = None, branch = None, upperLeave = None, lowerLeave = None):
		self.root = root
		self.branch = branch
		self.upperLeave = upperLeave
		self.interLeave = []
		self.lowerLeave = lowerLeave
		self.to = (None, None)
		self.otherTo = [] 
		self.whereFrom = None
		self.origin = None
		self.isRoot = False
		self.isBinary = False
		self.numNodes = None
		self.isUpperAnchor = False
		self.isLowerAnchor = False
		self.isInterAnchor = []
		self.isComplete = False
		self.upperLabel = None
		self.lowerLabel = None
		self.interLabel = []
		self.area = None
		self.breakSpot = []
		self.status = 0

	def isAnchor(self, anchorLines):
		if self.upperLeave in anchorLines:
			self.isUpperAnchor = True
			self.getLabel(self.upperLeave)
		if self.lowerLeave in anchorLines:
			self.isLowerAnchor = True
			self.getLabel(self.lowerLeave)

	def getNodeInfo(self):
		print '------node Information-------'
		print 'root:', self.root
		print 'branch: ', self.branch
		print 'upperLeave: ', self.upperLeave
		print 'lowerLeave: ', self.lowerLeave
		if self.to[0]:
			print 'upperLeave goes to:', self.to[0].branch
		if self.to[1]:
			print 'lowerLeave goes to:', self.to[1].branch
		if self.origin:
			print 'origin node is:', self.origin.branch


	def getLabel(self):
		pass

	def sortByY(self, item):
		return item[0][1]

	def getChildren(self):

		if self.to[0]:
			upperChildren = self.to[0].getChildren()
		else:
			if self.upperLabel:
				upperChildren = self.upperLabel
			elif self.isUpperAnchor:
				upperChildren = "%"
			else:
				upperChildren = "**"
		if self.to[1]:
			lowerChildren = self.to[1].getChildren()
		else:
			if self.lowerLabel:
				lowerChildren = self.lowerLabel
			elif self.isLowerAnchor:
				lowerChildren = "%"
			else:
				lowerChildren = "**"

		if self.isBinary:
			return "(%s, %s)" %(upperChildren, lowerChildren)
		else:
			result = "(%s," %upperChildren

			# newList = sorted(zip(self.interLeave, self.otherTo, self.interLabel, self.), key = self.sortByY)
			# tmp1 = []
			# tmp2 = []
			# for a,b in newList:
			# 	tmp1.append(a)
			# 	tmp2.append(b)
			# self.interLeave = tmp1
			# self.otherTo = tmp2
			for index, to in enumerate(self.otherTo):
				if to:
					interChildren = to.getChildren()
				else:
					if self.interLabel[index]:
						interChildren = self.interLabel
					elif self.isInterAnchor[index]:
						interChildren = "%"
					else:
						interChildren = "**"
				result += interChildren + ','

			return result + '%s)' %lowerChildren

# class parentNode():
# 	def __init__(self, root, branch):
# 		self.type = 1
# 		self.root = root
# 		self.branch = branch
# 		self.dist = None
# 		self.to = None
# 		self.whereFrom = None

	

# class childrenNode():
# 	def __init__(self, branch, upperLeave, lowerLeave, anchorLines):
# 		self.type = 0
# 		self.branch = branch
# 		self.upperLeave = upperLeave
# 		self.lowerLeave = lowerLeave
# 		self.to = None
# 		self.lowerTo = None
# 		self.fromRoot = None
# 		self.isUpperAnchor = False
# 		self.upperLabel = None
# 		self.lowerLabel = None
# 		self.isLowerAnchor = False
# 		self.isAnchor(anchorLines)

# 	def isAnchor(self, anchorLines):
# 		if self.upperLeave in anchorLines:
# 			self.isUpperAnchor = True
# 			self.getLabel(self.upperLeave)
# 		if self.lowerLeave in anchorLines:
# 			self.isLowerAnchor = True
# 			self.getLabel(self.lowerLeave)

# 	def getLabel(self, line):
# 		pass





class matchLines():
	def __init__(self, image, verLines, horLines):
		self.image = image
		self.height, self.width = image.shape
		self.verLines = verLines
		self.horLines = horLines
		self.parent= []
		self.children = []
		self.anchorLines = []
		self.interLines = []
		self.jointLines = []
		self.jointPoints = []
		self.upperCorners=[]
		self.lowerCorners = []
		self.length = []
		self.thickness = []

	def sortByXEnd(self, item):
		return (-item[2])

	def sortByXEndFromLeft(self, item):
		return (item[2])

	def sortByXHead(self,item):
		return (item[0], item[1])

	def sortByDist(self, item):
		return (-item[1])
	def sortByY(self, item):
		return item[1]

	def sortByLengthAndDist(self, item):
		return (item[0][0][4], item[1])

	def sortByLength(self, item):
		return (-item[4])

	def sortByNodeNum(self, item):
		return -item.numNodes

	def matchParent(self):
		self.horLines = sorted(self.horLines, key=self.sortByXEndFromLeft)	
		self.verLines = sorted(self.verLines, key=self.sortByXEndFromLeft)	
		margin = 5
		parent = []

		for line in self.horLines:
			x1, y1, x2, y2, length= line
			probParent = []
			for vline in self.verLines:
				vx1, vy1, vx2, vy2, vlength= vline
				if y1 > vy1 and y1 < vy2 and x2 > vx1 - margin and x2 < vx2 + margin:
					probParent.append(vline)
			minDist = None
			target = None
			if len(probParent) >0:
				for vline in probParent:
					vx1, vy1, vx2, vy2, vlength= vline
					dist = abs(vx2-x2)
					if dist == 0:
						score = vlength
					else:
						score = vlength * (1 - int(dist/margin)*0.1)
					if not target or score > minDist:
						target = vline
						minDist = score
				tupleLine = tuple(line)
				self.interLines.append(tupleLine)
				self.jointLines.append(target)
				parent.append(((tupleLine, target), minDist))
				self.jointPoints.append((x2-1,y2))
		parent = sorted(parent, key=self.sortByDist)

		self.parent = self.removeRepeatLines(parent)
		self.removeRepeatLinesBasic(self.interLines)
			

	def matchChildren(self):
		self.verLines = sorted(self.verLines, key=self.sortByXEndFromLeft)	
		self.horLines = sorted(self.horLines, key=self.sortByXHead)
		margin = 5
		children = []
		for line in self.verLines:
			x1, y1, x2, y2, length = line
			upperLine = []
			lowerLine = []
			interLine = []
			for hline in self.horLines:
				hx1, hy1, hx2, hy2, hlength = hline
				isUpperLine = False
				isLowerLine = False
				if x1 > hx1 - margin and x1 < hx1 + margin and y1 > hy1 - margin and y1 < hy1 + margin:
					upperLine.append(hline)
					isUpperLine = True
				if x1 > hx1 - margin and x1 < hx1 + margin and y2 > hy1 - margin and y2 < hy1 + margin:
					lowerLine.append(hline)
					isLowerLine = True
				if x1 -margin < hx1  and x1 + margin > hx1  and y1 -margin < hy1 and y2 + margin > hy1 :
					if not (isUpperLine or isLowerLine):
						interLine.append(hline)
			if len(upperLine)>0 or len(lowerLine)>0 or len(interLine)>0:
				# if line in self.jointLines:
				totalDist = 0
				minDist = None
				upTarget = None
				isBinary = True
				for upLine in upperLine:
					upx1, upy1, upx2, upy2, uplength = upLine
					dist = abs(upx1-x1) + abs(upy1 - y1)
					if dist == 0:
						score = uplength
					else:
						score = uplength * (1 - int(dist/margin)*0.1)
					if not minDist or score > minDist:
						minDist = score
						upTarget = upLine
				if minDist:
					totalDist +=minDist
				minDist = None
				downTarget = None
				for downLine in lowerLine:
					downx1, downy1, downx2, downy2, downlength = downLine
					dist = abs(y2-downy1) + abs(downx1 - x1)
					if dist == 0:
						score = downlength
					else:
						score = downlength * (1 - int(dist/margin)*0.1)
					if not minDist or score > minDist:
						minDist = score
						downTarget = downLine
				if minDist:
					totalDist +=minDist
				if len(interLine)>0:

					isBinary = False
					interLine = self.removeRepeatLinesBasic(interLine)
					for subline in interLine:
						subline = tuple(subline)
						if subline not in self.interLines:
							self.anchorLines.append(subline)
				if upTarget:
					upTarget = tuple(upTarget)
					self.upperCorners.append((upTarget[0], upTarget[1]))
					if upTarget not in self.interLines:
						self.anchorLines.append(upTarget)
				if downTarget:
					downTarget = tuple(downTarget)
					self.lowerCorners.append((downTarget[0], downTarget[1]))
					if downTarget not in self.interLines:
						self.anchorLines.append(downTarget)
				if not (upTarget or downTarget):
					if isBinary:
						children.append(((line, (upTarget, downTarget)), totalDist-100))
					else:
						tmpLineList = [upTarget, downTarget]
						for subline in interLine:
							tmpLineList.append(subline)
						children.append(((line, tuple(tmpLineList)), totalDist -100))
				else:
					if isBinary:
						children.append(((line, (upTarget, downTarget)), totalDist))
					else:
						tmpLineList = [upTarget, downTarget]
						for subline in interLine:
							tmpLineList.append(subline)
						children.append(((line, tuple(tmpLineList)), totalDist))
		children = sorted(children, key=self.sortByDist)

		self.children = self.removeRepeatLines(children)
		self.anchorLines = self.removeRepeatLinesBasic(self.anchorLines)


	def removeRepeatLinesBasic(self, lineList):
		lineList = sorted(lineList, key=sortByLength)
		margin = 5
		i=0	
		while i<len(lineList):
			x1, y1, x2, y2, length= lineList[i]

			for j in xrange(len(lineList)-1, i, -1):
				if x1 - margin < lineList[j][0] and x2 + margin > lineList[j][2]:
					if y2+margin > lineList[j][3] and y1-margin < lineList[j][1]:
						del lineList[j]
			i +=1
		return lineList

	def removeRepeatLines(self, lineList):
		margin = 5
		i=0
		num = 1
		while i<len(lineList):
			lines, dist = lineList[i]
			x1, y1, x2, y2, length =lines[0]
			# parent = False
			# if len(lines[1])==5:
			# 	parent = True
			# 	bx1, by1, bx2, by2, blength = lines[1]

			for j in xrange(len(lineList)-1, i, -1):

				if x1 - margin < lineList[j][0][0][0] and x2 + margin > lineList[j][0][0][2]:
					if y2+margin > lineList[j][0][0][3] and y1-margin < lineList[j][0][0][1]:
						if lineList[j][0][0][4] <= length+ (margin-2):
							num +=1
							del lineList[j]
				# if parent:
				# 	if bx1 - margin < lineList[j][0][1][0] and bx2 + margin > lineList[j][0][1][2]:
				# 		if by2+margin > lineList[j][0][1][3] and by1-margin < lineList[j][0][1][1]:
				# 			if lineList[j][0][1][4] <= blength+ (margin-2):
				# 				num +=1
				# 				del lineList[j]
			i +=1
			self.thickness.append(num)
			num = 1
		# print (sum(thickness)+0.0) / len(thickness)
		return lineList

	def makeCleanAnchorLines(self):
		self.anchorLines = sorted(self.anchorLines, key = self.sortByDist)
		margin = 5
		cleanLines = []
		i = 0
		while i<len(self.anchorLines):
			line = self.anchorLines[i]
			x1, y1, x2, y2, length = line
			maxlen = length
			target = line
			j = i
			lowerBound = 0
			upperBound = self.height-1
			if y1 - margin > 0:
				lowerBound = y1 - margin
			if y1 + margin < self.height-1:
				upperBound = y1 + margin
			while j<len(self.anchorLines) and lowerBound < self.anchorLines[j][1] and upperBound > self.anchorLines[j][1]:
				if self.anchorLines[j][4] > maxlen:
					maxlen = self.anchorLines[j][4]
					target = self.anchorLines[j]
				j +=1
			cleanLines.append(target)
			i=j
		self.anchorLines = cleanLines[:]

	def removeText(self):
		if self.height < 500:
			rad = 10
		elif self.height >= 500 and self.height < 700:
			rad = 12
		elif self.height >=700:
			rad = 15
		tmpImage = self.image.copy()
		for line in self.anchorLines:
			x1, y1, x2, y2, length = line
			lowerBound = 0
			upperBound = self.height
			if y1 - rad > 0:
				lowerBound = y1 -rad
			if y1 + rad < self.height:
				upperBound = y1 + rad
			tmpImage[lowerBound:upperBound, x2+1:] = 255
		
		self.cleanImage = tmpImage.copy()

	def advancedSelection(self):
		if len(self.anchorLines) >=10:
			sameEnd = True
			thres = 5
			endPoint = self.anchorLines[0][2]
			for index in range(int(len(self.anchorLines)*0.4)):
				if self.anchorLines[index][2] < endPoint - thres or self.anchorLines[index][2] > endPoint + thres:
					sameEnd = False

			if sameEnd:
				for i in xrange(len(self.anchorLines)-1, -1, -1):
					x1, y1, x2, y2, length = self.anchorLines[i]
					if x2 < endPoint - thres or x2 > endPoint + thres:
						del self.anchorLines[i]

		return sameEnd



	def displayChildren(self):
		whatever = cv.cvtColor(self.image,cv.COLOR_GRAY2RGB)
		count = 0
		for ((line, hlines), dist) in self.children:
			
			x1, y1, x2, y2, length= line
			
			count +=1
			if count%3 == 0:
				color = (255, 0 , 0)
			elif count%3 == 1:
				color = (0, 255, 0)
			else:
				color = (0, 0, 255)

			cv.rectangle(whatever, (x1, y1), (x2, y2), color=color, thickness=2)

			for hline in hlines:
				if hline:
					hx1, hy1, hx2, hy2, hlength = hline
					cv.rectangle(whatever, (hx1, hy1), (hx2, hy2), color=color, thickness=2)
		plt.imshow(whatever)
		plt.show()


	def displayParent(self):
		whatever = cv.cvtColor(self.image,cv.COLOR_GRAY2RGB)
		count = 0
		for ((line, vline), dist) in self.parent:

			x1, y1, x2, y2, length= line
			vx1, vy1, vx2, vy2, vlength = vline
			count +=1
			if count%3 == 0:
				color = (255, 0 , 0)
			elif count%3 == 1:
				color = (0, 255, 0)
			else:
				color = (0, 0, 255)

			cv.rectangle(whatever, (x1, y1), (x2, y2), color=color, thickness=2)
			cv.rectangle(whatever, (vx1, vy1), (vx2, vy2), color=color, thickness=2)

		plt.imshow(whatever)
		plt.show()

	def displayCorners(self):
		whatever = cv.cvtColor(self.image,cv.COLOR_GRAY2RGB)
		rad = 3
		for x, y in self.jointPoints:
			cv.rectangle(whatever, (x-rad, y - rad), (x + rad, y +rad), color=(0, 255, 0), thickness=1)
		for x, y in self.upperCorners:
			cv.rectangle(whatever, (x-rad, y - rad), (x + rad, y +rad), color=(0, 255, 0), thickness=1)
		for x, y in self.lowerCorners:
			cv.rectangle(whatever, (x-rad, y - rad), (x + rad, y +rad), color=(0, 255, 0), thickness=1)
		plt.imshow(whatever)
		plt.show()

	def getAnchorLines(self):
		return self.anchorLines

	def getParent(self):
		return self.parent

	def getChildren(self):
		return self.children

	def displayAnchorLines(self):
		whatever = cv.cvtColor(self.image,cv.COLOR_GRAY2RGB)
		for line in self.anchorLines:
			x1, y1, x2, y2, length= line
			cv.rectangle(whatever, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
		plt.imshow(whatever)
		plt.show()

	def displayLines(self):
		whatever = cv.cvtColor(self.image,cv.COLOR_GRAY2RGB)

		for line in self.horLines:
			x1, y1, x2, y2, length= line
			cv.rectangle(whatever, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

		for line in self.verLines:
			x1, y1, x2, y2, length= line
			cv.rectangle(whatever, (x1, y1), (x2, y2), color=(0,0,255), thickness = 2)
		plt.imshow(whatever)
		plt.show()

	def displayInterLines(self):
		whatever = cv.cvtColor(self.image,cv.COLOR_GRAY2RGB)
		for line in self.interLines:
			x1, y1, x2, y2, length= line
			cv.rectangle(whatever, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
		plt.imshow(whatever)
		plt.show()		

	def getCornerLists(self):
		return (self.upperCorners, self.lowerCorners, self.jointPoints)

	def getCleanImage(self):
		return self.cleanImage

	def getThickness(self):
		return (sum(self.thickness)) / len(self.thickness) + 1


def sortByLength(item):
	return (-item[4])

def getThickness(lineList, mode):
	lineList = sorted(lineList, key=sortByLength)
	margin = 10
	i=0
	ave = []
	if mode == 'hor':		
		while i<len(lineList):
			num = 1
			x1, y1, x2, y2, length= lineList[i]

			for j in xrange(len(lineList)-1, i, -1):
				if x1 - margin < lineList[j][0] and x2 + margin > lineList[j][2]:
					if y2+margin > lineList[j][3] and y2-margin < lineList[j][3]:
						num +=1
						del lineList[j]
			ave.append(num)
			i +=1
	elif mode == 'ver':
		while i<len(lineList):
			num = 1
			x1, y1, x2, y2, length= lineList[i]

			for j in xrange(len(lineList)-1, i, -1):
				if x2 + margin > lineList[j][2] and x2 - margin < lineList[j][2]:
					if y2 + margin > lineList[j][3] and y1 -margin < lineList[j][3]:
						num+=1
						del lineList[j]
			ave.append(num)
			i +=1

	return (sum(ave) + 0.0)/len(ave)



class removeText():
	def __init__(self, image, oriImage, verLines, horLines):
		self.image = image
		self.oriImage = oriImage
		self.height, self.width = image.shape
		self.verLines = verLines
		self.horLines = horLines
		self.sortLineList()

		# fileName = "images/tree3_wo.jpg"
		# image1 = cv.imread(fileName,0)
		# p = preProcess(image1)
		# p.downSample()
		# p.bolding()
		# p.thinning()
		# image1 = p.returnImage()

		# l = lineDetection(image1)
		# (verLines1, self.horLines1) = l.getLineLists()

		self.classify()
		self.advancedSelection()
		# self.removeText()

	def sortByLength(self, item):
		return (-item[4])

	def sortByEnd(self,item):
		return (-item[3])

	def sortKey(self, item):
		return (-item[2],(item[3]))

	def sortByY(self, item):
		return (item[1])


	def sortLineList(self):
		self.horLines = sorted(self.horLines, key=self.sortKey)		

	def classify(self, aveThres = 20, lenThres = 9):
		rad = 2

		self.realLine = []
		self.fakeLine = []
		for line in self.horLines:
			x1, y1, x2, y2, length = line
			if y1-rad< 0:
				lineBox = self.oriImage[0:y2+rad, x1+1:x2]
			elif  y2+rad>self.height:
				lineBox = self.oriImage[y1-rad:self.height,x1+1:x2]
			else:
				lineBox = self.oriImage[y1-rad:y2+rad, x1+1:x2]
			upperBound = y2-2
			lowerBound = y2+3
			if upperBound<0:
				upperBound = 0
			if lowerBound>self.height:
				lowerBound = self.height
			if x2+1 <self.width and x2+1+1 < self.width:
				verLineCheck = self.oriImage[upperBound:lowerBound, x2+1:x2+1+1]
				doubleCheck = self.oriImage[upperBound:lowerBound, x2-1:x2]
				# tripleCheck = self.oriImage[upperBound:lowerBound, x2-1:x2]

				varLineBox = np.sqrt(np.var(lineBox, 1))
				ave = np.average(varLineBox)
				varLineBox = list(varLineBox)

				index, value = max(enumerate(varLineBox), key=operator.itemgetter(1))
				varLineBox.pop(index)
				newAve = (sum(varLineBox) + 0.0) / len(varLineBox)

				varDoubleCheck = np.sqrt(np.var(doubleCheck,0))
				double = True
				if np.sqrt(np.var(doubleCheck,0)) < 10 :
					double = False

				if (newAve < aveThres/2 or ave<aveThres) and length >= lenThres and np.sqrt(np.var(verLineCheck,0))[0]<40 and double:
					# print varDoubleCheck
					saveLine = (x1, y1, x2, y2, length)
					self.realLine.append(saveLine)
					# print lineBox
					# print doubleCheck
					
					
					# np.sqrt(np.var(doubleCheck,0))[0]>10
					# self.realLine.append((line, length, varLineBox))
					# print ave , "real", (line, length, varLineBox)
				# else:
				# 	self.fakeLine.append((line, length, varLineBox))
					# print "fake", (line, length, varLineBox)
		self.realLine = sorted(self.realLine, key=self.sortByLength)

		i=0
		while i<len(self.realLine):
			x1, y1, x2, y2, length = self.realLine[i]
			
			# while j+1<len(cornerList) and x + 5 > cornerList[j+1][0] and  x-5 < cornerList[j+1][0] and y + 5 > cornerList[j+1][1] and y-5 < cornerList[j+1][1]:
			# 	del cornerList[j+1]
			# while j+1<len(cornerList) and x + 5 > cornerList[j+1][0] and  y + 5 > cornerList[j+1][1]:
			# 	del cornerList[j+1]
			for j in xrange(len(self.realLine)-1, i, -1):
				if x2 + 5 > self.realLine[j][2] and x2 - 5 < self.realLine[j][2]:
					if y2+5 > self.realLine[j][3] and y2-5 < self.realLine[j][3]:
						del self.realLine[j]
			i +=1



		# for line in self.realLine:
		# 	x1, y1, x2, y2 = line
		# 	x = 187
		# 	rad = 2
		# 	y = 8
		# 	if x1 < x and x2 > x and y1 < y + rad and y1 > y - rad :
		# 		lineBox = self.oriImage[y1-rad:y2+rad, x1:x2]
		# 		doubleCheck = self.oriImage[y1-2:y1+3, x2-1:x2]
		# 		print lineBox
		# 		print doubleCheck
		# 		plt.imshow(lineBox, cmap='Greys_r')
		# 		plt.show()						

	# def recoverMissingLabels(self):
	# 	self.realLine = sorted(self.realLine, key=self.sortByY)
	# 	print self.realLine
	# 	yList = []
	# 	for line in self.realLine:
	# 		x1, y1, x2, y2, length = line
	# 		yList.append(y1)
	# 	yGap = []
	# 	for i in range(len(yList)-1):
	# 		yGap.append(yList[i+1] - yList[i])
		
	# 	for i in range(100):
	# 		rand = int(random.uniform(0, len(yGap)))
	# 		if len(yGap) < 20:
	# 			iteration = 4
	# 		else:
	# 			iteration = len(yGap) /5
	# 		for j in range(rand, rand+iteration):		


	def advancedSelection(self):
		if len(self.realLine) >=10:
			sameEnd = True
			thres = 5
			endPoint = self.realLine[0][2]
			for index in range(int(len(self.realLine)*0.4)):
				if self.realLine[index][2] < endPoint - thres or self.realLine[index][2] > endPoint + thres:
					sameEnd = False

			if sameEnd:
				for i in xrange(len(self.realLine)-1, -1, -1):
					x1, y1, x2, y2, length = self.realLine[i]
					if x2 < endPoint - thres or x2 > endPoint + thres:
						del self.realLine[i]

			# self.recoverMissingLabels()


	def removeText(self):
		ratio = 0 

		rad = int(self.height/(len(self.realLine)*1.3))
		self.coverRange = []
		index =0
		self.realLine = sorted(self.realLine, key=self.sortKey)

		while index<len(self.realLine):
			# print index
			line = self.realLine[index]
			x1, y1, x2, y2, length = line
			# cv.rectangle(self.image, (x1, y1-1), (x2, y2+1), color=(0), thickness=3)
			# print x1, x2, y1, y2
			if y1-rad<0:
				upperBound = 0
				downBound = y2+rad
			elif y2+rad>self.height:
				upperBound = y1-rad
				downBound = self.height
			else:
				upperBound = y1-rad
				downBound = y2+rad
			# print upperBound, downBound
			isCover = self.checkRange(upperBound, downBound, x2)
			if not isCover[0]:
			
				self.oriImage[isCover[1]:isCover[2], x2+2:] = 255   #remove labels
				# cv.rectangle(self.image, (x2+3, isCover[1]), (x2+95, isCover[2]), color=(0), thickness=0)
			index +=1
		# print self.coverRange


	def checkRange(self, y1, y2, x2):

		index = 0
		if not self.coverRange:
			self.coverRange.append((y1,y2, x2))
			# print y1, y2
			return (False,y1,y2)
		else:
			for index in range(len(self.coverRange)):
				if self.coverRange[index][0]>y1:
					if index!= 0 and self.coverRange[index][0] > y2 and y1> self.coverRange[index-1][1]:
						self.coverRange.insert(index, (y1, y2, x2))
						# print y1, y2
						return (False, y1, y2)
					elif index == 0:						
						if self.coverRange[index][0] < y2:
							y2 = self.coverRange[index][0] 
						self.coverRange.insert(index, (y1,y2, x2))
						# print y1, y2
						return (False, y1, y2)
					else:
						if self.coverRange[index][0] == self.coverRange[index-1][1]:
							return (True, y1, y2)
						else:
							if self.coverRange[index][0] < y2:
								y2 = self.coverRange[index][0] 
							if self.coverRange[index-1][1] >y1:
								y1 = self.coverRange[index-1][1] 
							self.coverRange.insert(index, (y1, y2, x2))
							# print y1, y2
							return (False, y1, y2)

			if self.coverRange[len(self.coverRange)-1][1] >y1:
				y1 = self.coverRange[len(self.coverRange)-1][1]			
			self.coverRange.append((y1,y2, x2))
			# print y1, y2
			return (False, y1, y2)
	def displayImage(self):
		plt.imshow(self.image, cmap='Greys_r')
		plt.show()		

	def displayAnchorLines(self):
		whatever = cv.cvtColor(self.oriImage,cv.COLOR_GRAY2RGB)

		# for line in self.horLines:
		# 	x1, y1, x2, y2, length = line
		# 	cv.rectangle(whatever, (x1, y1-1), (x2, y2+1), color=(0, 255, 0), thickness=2)
		for line in self.realLine:
			x1, y1, x2, y2, length = line
			cv.rectangle(whatever, (x1, y1-1), (x2, y2+1), color=(255, 0, 0), thickness=3)
		# for line in self.verLines:
		# 	x1, y1, x2, y2, length= line
		# 	cv.rectangle(whatever, (x1-1, y1), (x2+1, y2), color=(0,0,255), thickness = 2)
		plt.imshow(whatever)
		plt.show()

	def returnCoverRange(self):
		return self.coverRange

	def returnImage(self):
		return self.oriImage

	def returnAnchorLines(self):
		return self.realLine




		# for line in self.realLine:
		# 	((x1, y1, x2, y2), length, varLineBox) = line
		# 	cv.rectangle(self.image, (x1, y1-rad), (x2, y2+rad), color=(0), thickness=3)
		
		# plt.imshow(self.image, cmap='Greys_r')
		# plt.show()
		# 		if line in self.horLines1:
		# 			longLine.append((line, length, varLineBox))
		# 		else:
		# 			if length>10:
		# 				isOver = False
		# 				for var in varLineBox:
		# 					if var>100:
		# 						isOver = True
		# 				if not isOver:
		# 					longLine.append((line, length, varLineBox))
		# 				# print length, varLineBox
		# 				# plt.imshow(lineBox, cmap='Greys_r')
		# 				# plt.show()
		# 			else:						
		# 				shortLine.append((line, length, varLineBox))
		# 		# print length
		# 		# print lineBox
		# 		# print "Variance:", varLineBox
		# 		# cv.rectangle(self.image, (x1-rad, y1-rad), (x2+rad, y2+rad), color=(0), thickness=0)
		# print longLine
		# print shortLine
		# for line in longLine:


		# for lines in shortLine:
		# 	line, length, var = lines
		# 	isOver = False
		# 	for item in var:
		# 		if item > 100:
		# 			isOver = True

		# 	if not isOver:
		# 		print 'Over', lines
		# 		x1, y1, x2, y2 = line
		# 		lineBox = self.image[y1-rad:y2+rad, x1:x2]
		# 		plt.imshow(lineBox, cmap='Greys_r')
		# 		plt.show()
		

def plotDots(lines, w, h):
	image = np.ones((h,w),dtype=np.uint8)
	image = image * 255
	for line in lines:
		x1, y1, x2, y2 = line
		image[y1][x1]= 0

	plt.imshow(image,cmap='Greys_r')
	plt.show()

def getFilesInFolder(folderPath):
	fileNameList = [f for f in listdir(folderPath) if isfile(join(folderPath, f))]


	return fileNameList


def printGroupNode(image, rootList):
	image = cv.cvtColor(image,cv.COLOR_GRAY2RGB)
	for root in rootList:
		stack = []
		stack.append(root)
		while stack:
			

			node = stack.pop()
			node.getNodeInfo()
			print '-----------------------------------------'


			x1, y1, x2, y2, length = node.branch

			if node.to[0]:
				stack.append(node.to[0])
			if node.to[1]:
				stack.append(node.to[1])
			for subNode in stack:
				print 'main', subNode.branch, 
				if subNode.to[0]:
					print 'upperNode', subNode.to[0].branch
				if subNode.to[1]:
					print 'lowerNode', subNode.to[1].branch
			cv.rectangle(image , (x1, y1), (x2, y2), color=(255, 0,0), thickness=2)
			plt.imshow(image,cmap='Greys_r')
			plt.show()

def cutLines(horLines, verLines):
	newList = []
	margin = 5
	for line in horLines:
		if line[4] <20:
			newList.append(line)
		else:
			x1, y1, x2, y2, length = line
			isnotcut = True
			for vx1, vy1, vx2, vy2, vlength in verLines:
				if x1+margin<vx1 and vx1<x2-margin and vy1<y1 and y1<vy2:
					newline1 = [x1, y1, vx1, y2, vx1-x1]
					newline2 = [vx1, y1, x2, y2, x2-vx1]
					newList.append(newline1)
					newList.append(newline2)
					isnotcut = False
					break
			if isnotcut:
				newList.append(line)

	return newList


if __name__=='__main__':
	

	# filePath = "images/tree2_wo.jpg"
	# oriImage = cv.imread(filePath,0)
	# image = oriImage
	# fig = plt.figure()
	# a = fig.add_subplot(1,2,1)
	# plt.imshow(image,cmap='Greys_r')
	# a = fig.add_subplot(1,2,2)
	# plt.imshow(oriImage,cmap='Greys_r')
	# plt.show()
	# for index in range(1,2):
	# 	i = str(index)
	# 	fileName = 'tree' + i
	# 	filePath = 'images/' + fileName +'.jpg'
	# 	image = cv.imread(filePath,0)
	# 	plt.imshow(image, cmap='Greys_r')
	# 	plt.show()
	# 	# p = preProcess(image)
	# 	# p.downSample()
	# 	# image1 = p.returnImage()
	# 	# p.bolding()
	# 	# image2 = p.returnImage()

	# 	l = lineDetection(image)
	# 	# l.displayImage()
	# 	(verLines, horLines) = l.getLineLists()
	# 	print verLines

	# 	h,w = image.shape
	# 	plotDots(horLines, w,h)
	# print horLines

######################################## line matching pipeline ########### 	########


	folderPath = 'images/'
	fileNameList = getFilesInFolder(folderPath)
	print fileNameList
	for index in range(5, len(fileNameList)):
		print index
		filePath = folderPath + fileNameList[index]

		if isfile(filePath) :
			image = cv.imread(filePath,0)

			oriImage = image.copy()
			plt.imshow(image, cmap='Greys_r')
			plt.show()

			# image_x = cv.Sobel(image, -1, dx = 1, dy = 0, ksize = 3)
			# image_y = cv.Sobel(image, -1, dx = 0, dy = 1, ksize = 3)
			# image = cv.addWeighted(image_x, 0.5, image_y, 0.5, 0)

			# plt.imshow(image, cmap='Greys_r')
			# plt.show()



			p = preProcess(image)
			p.downSample()
			
			p.bilateralFilter()
			p.bolding()

			# p.displayImage()
			image = p.returnImage()

			l = lineDetection(image, 10)
			# l.displayImage()
			(verLines, horLines) = l.getLineLists()
			horLines = cutLines(horLines, verLines)
			# verLines = removeRepeatLines(verLines, 'ver')
			# horLines = removeRepeatLines(horLines, 'hor')
			# r = removeText(oriImage, image, verLines, horLines)
			ml = matchLines(image, verLines, horLines)
			ml.displayLines()
			ml.matchParent()
			ml.displayParent()
			ml.matchChildren()
			# ml.makeCleanAnchorLines()
			ml.removeText()
			ml.displayChildren()
			# ml.displayCorners()
			parent = ml.getParent()
			children = ml.getChildren()
			globalParent = parent
			globalChildren = children
			anchorLines = ml.getAnchorLines()
			cleanImage = ml.getCleanImage()
			cornerLists = ml.getCornerLists()
			# thickness1 = getThickness(verLines, 'ver')
			# thickness2 = getThickness(horLines, 'hor')	
			# thickness = int(round((thickness1+thickness2)/2))
			# if thickness == 1:
			# 	thickness =2
			# # print thickness
			# sp = preProcess(cleanImage)
			# # sp.gaussianBlur()
			# sp.bolding()
			# sp.displayImage()
			# cleanImage = sp.returnImage()
			# sc = scaleCornerDetection(cleanImage, thickness)
			# cornerLists = sc.getCornerLists()


			nodeList = createNodes(parent, children)
			print 'yes'
			rootList = getRootList(nodeList, anchorLines)
			print 'yes'
				

			# r.displayAnchorLines()
			ml.displayInterLines()
			ml.displayAnchorLines()
			
			for node in rootList:
				print node.isComplete, node.numNodes, len(node.breakSpot), node.area
			displayNodes(image, nodeList)
			# rootList = removeFakeTree(rootList)
			print rootList[0].breakSpot	

			# printGroupNode(image,rootList)
			isDone = checkDone(rootList)
			if isDone:
				print treeRecover(rootList[0])
			else:
				fixTrees(rootList, parent, children, horLines, verLines)
				checkDone(rootList)
				print treeRecover(rootList[0])

			# displayRecoveredTree(image,rootList[0])
			
			display(image, rootList, cornerLists)
			# displayNodesGroupByOrder(image, nodeList)

			# if isDone:
			# 	TreeRecover(isDone)
			# else:
				





		# sl = lineDetection(cleanImage, 7)
		# (verLines, horLines) = sl.getLineLists()
		# sml = matchLines(cleanImage, verLines, horLines)
		# sml.displayLines()
		# sml.matchParent()
		# sml.displayParent()
		# sml.matchChildren()
		# sml.makeCleanAnchorLines()
		# sml.displayChildren()
		# sml.displayCorners()






##########################################end here##########################################

	# for i in range(13, 26):
	# 	index = str(i)
	# 	filePath = "images/tree" + index + ".jpg"
	# 	image = cv.imread(filePath,0)
	# 	oriImage = image.copy()
	# 	plt.imshow(image, cmap='Greys_r')
	# 	plt.show()
	# 	p = preProcess(image)
	# 	p.downSample()
	# 	p.bolding()
	# 	p.displayImage()
	# 	image = p.returnImage()

	# 	l = lineDetection(image)
	# 	l.displayImage()
	# 	(verLines, horLines) = l.getLineLists()


	# 	r = removeText(oriImage, image, verLines, horLines)
	# 	r.returnAnchorLines()


##########################################################################################
		# oriImage = r.returnImage()
		# ll = lineDetection(image)
		# (verLines, horLines) = ll.getLineLists()
		# rr = removeText(oriImage, image, verLines, horLines)
		# rr.displayAnchorLines()

		# oriImage = r.returnImage()
		# p = preProcess(oriImage)
		# p.bolding()
		# oriImage = p.returnImage()
		# a = scaleCornerDetection(oriImage)
		# a.parsing()
		

	# a = scaleCornerDetection(oriImage)
	# a.parsing()
	# c = cornerDetection(oriImage)
	# c.displayCorners()
	# c.displayImage()


#############################pipline_start here##########################

	# filePath = "images/tree17.jpg"
	# image = cv.imread(filePath,0)
	# plt.imshow(image, cmap='Greys_r')
	# plt.show()
	# p = preProcess(image)
	# p.downSample()
	# oriImage = p.returnImage()
	# p.bolding()
	# p.thinning()
	# p.displayImage()
	# image = p.returnImage()
	# # plt.imshow(image, cmap='Greys_r')
	# # plt.show()

	# l = lineDetection(image)
	# (verLines, horLines) = l.getLineLists()
	# r = removeText(image, oriImage, verLines, horLines)
	# image = r.returnImage()
	# r.displayImage()
	
	# labelList = r.returnCoverRange()
	# c = cornerDetection(image)
	# c.displayCorners()
	# image = c.returnImage()

	# (upperList, downList, jointList) = c.getCornerLists()
	# rc = readCorners(filePath,upperList, downList, jointList, labelList)
	# rc.extractTree()
	# treeText = rc.getTreeText()
	
	# t = Tree(treeText+";")
	
	# t.show()


##############################pipeline ends here#####################################


	# filePath = "images/tree2_wo.jpg"
	# image = cv.imread(filePath,0)
	# p = preProcess(image)
	# p.downSample()
	# p.bolding()
	# p.thinning()
	# image = p.returnImage()
	# c = cornerDetection(image)
	# c.displayImage()
	# lists = c.getCornerLists()
	# print lists

############################################################

	# for i in range(1,6):
	# 	filePath = "images/tree%d_wo.jpg" %i
	# 	oriImage = cv.imread(filePath,0)
	# 	image = cv.imread(filePath,0)

	# 	# plt.imshow(image, cmap='Greys_r')
	# 	# plt.show()

	# 	p = preProcess(image)
	# 	p.downSample()
	# 	p.bolding()
	# 	p.thinning()

	# 	image = p.returnImage()
	# 	# plt.imshow(image, cmap='Greys_r')
	# 	# plt.show()


	# 	l = lineDetection(image)
	# 	(verLines, horLines) = l.getLineLists()
	# 	# l.displayImage()

	# 	c = cornerDetection(image)
	# 	# c.displayImage()
	# 	(upperList, downList, jointList) = c.getCornerLists()
		
	# 	fileName = ['verLines', 'horLines', 'upperList', 'downList', 'jointList']
	# 	index = 0
	# 	lists = [verLines, horLines, upperList, downList, jointList]
	# 	for name in fileName:
	# 		path = "images/tree%d_wo_%s.p" %(i, name)
	# 		writeList(lists[index], path)
	# 		index +=1
##################################################################

		# image = l.returnImage()

		# fig = plt.figure()
		# f = fig.add_subplot(1,2,1)
		# plt.imshow(oriImage,cmap='Greys_r')

		# f = fig.add_subplot(1,2,2)
		# plt.imshow(image,cmap='Greys_r')
		# plt.show()





	# # for i in range(1,10):
	
	# filePath = "images/tree1_wo.jpg"
	# oriImage = cv.imread(filePath, 0)
	# image = oriImage


	# a = preProcess(image)
	# a.downSample()
	# a.bolding()
	# a.thinning()
	# image = a.returnImage()
	
	# # plt.show()

	# # b = cornerDetection(image)
	# # image = b.returnImage()
	# # b.printCorners()

	# c = lineDetection(image)
	# # # c.laplacian()
	# # # c.sobel()
	# image = c.returnImage()
	# plt.imshow(image, cmap='Greys_r')
	# plt.show()

