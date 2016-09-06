import cv2 as cv
import numpy as np
from Node import *
from ImageData import *






class PhyloParser():
    
    def __init__(self):
        return 
    
    def preprocces(self, image_data, debug = False):

        image_data.image, image_data.varianceMask = self.purifyBackGround(image_data)

        image_data = self.findContours(image_data)
        
        image_data = self.downSample(image_data)
        if debug:
            self.displayImage(image_data.image)
            
        image_data = self.bilateralFilter(image_data)
        
        if debug:
            self.displayImage(image_data.image)
            
        image_data = self.binarize(image_data, 180, 3)
        
        if debug:
            self.displayImage(image_data.image)
            
        return image_data
        
    ## static method for preprocessing ##
    

    @staticmethod
    def sobelFilter(image, k=5, sigma = 3):
        image  = PhyloParser.gaussianBlur(image, (k,k), sigma)
        sobelx = cv.Sobel(image, cv.CV_64F, 1, 0, ksize = k)
        sobely = cv.Sobel(image, cv.CV_64F, 0, 1, ksize = k)
        image = cv.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
        return image
    
    @staticmethod 
    def gaussianBlur(image, kernelSize = (3,3), sigma= 1.0):
        image = cv.GaussianBlur(image, kernelSize, sigma)
        return image
    @staticmethod 
    ## Old bolding, duplicate with threshold
    def binarize(image, thres=180, mode = 0):
        ret, image = cv.threshold(image, thres, 255, mode)
        return image
    
    @staticmethod
    def bilateralFilter(image, radius = 3, sigmaI = 30.0, sigmaS = 3.0):
        image = cv.bilateralFilter(image, radius, sigmaI, sigmaS)
        return image

    @staticmethod
    def downSample(image, parameter = 500.0):
        height, width = image.shape
        if height>700:
            if height > 1000:
                parameter = 700.0
            ratio  = (parameter + 0.0) / height
            image = cv.resize(image, (0,0), fx = ratio, fy=ratio)

        return image

    @staticmethod
    # Make this static
    # you don't need self.height,  just get the image size again from image.shape
    def thinningIteration(image, iteration):
        return image

    @staticmethod
    # make this static too
    def thinning(image):
        return image
        
    @staticmethod
    def displayImage(image):
        plt.imshow(image, cmap='Greys_r')
        plt.show()

    @staticmethod
    def sortByCntsLength(item):
        return -len(item)


    def findContours(self, image):

        height, width = image.shape
        _, contours, hierarchy= cv.findContours(image.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        contours = sorted(contours, key=self.sortByCntsLength)

        mask = np.zeros((height,width), dtype=np.uint8)
        cv.drawContours(mask, contours, 0, (255), thickness = -1)
        nonTreeMask = np.zeros((height, width), dtype = np.uint8)
        for index in range(1, len(contours)):
            cv.drawContours(nonTreeMask, contours, index, (255), thickness = -1)

        plt.imshow(mask, cmap='Greys_r')
        plt.show()
        plt.imshow(nonTreeMask, cmap='Greys_r')
        plt.show()

        return mask, nonTreeMask


    # def purifyBackGround(image, threshold_var = 0.01, threshold_pixel = 5, kernel_size = (3,3)):

    #     dim = image.shape
    #     mask = np.zeros(dim, dtype=np.uint8)   # 1:keep 0:remove
        
    #     for i in range(0, dim[0] - kernel_size[0] + 1):
    #         for j in range(0, dim[1] - kernel_size[1] + 1):

    #             patch = image[i:i+kernel_size[0], j:j+kernel_size[1]].copy().astype("float")/255      
    #             patch_variance =  np.var(patch)
    #             patch_sum = np.sum(patch)*255
                
    #             threshold_sum = kernel_size[0] * kernel_size[1] * threshold_pixel
                
    #             if patch_variance < threshold_var and patch_sum > threshold_sum:
    #                 mask[i:i+kernel_size[0], j:j+kernel_size[1]] = 255

    #     image[np.where(mask == 255)] = 255
    #     return image, mask
    def purifyBackGround(self, image_data, threshold = 0.01, kernel_size = (3,3)):
        image = image_data.image
        dim = image.shape
        mask = np.zeros(dim, dtype=np.uint8)   # 1:keep 0:remove
        test = np.zeros(dim, dtype=np.uint8)
        h,w = image.shape

        for i in range(0, dim[0] - kernel_size[0] + 1):
            for j in range(0, dim[1] - kernel_size[1] + 1):

                patch = image[i:i+kernel_size[0], j:j+kernel_size[1]].copy().astype("float")/255      
                patch_variance =  np.var(patch)
                if j < w and i < h:
                    if patch_variance < threshold:
                        test[i][j] = 0
                    else:
                        test[i][j] = 255

                if patch_variance < threshold:                
                    mask[i:i+kernel_size[0], j:j+kernel_size[1]] = 255


        boxList, contours = self.findContours(test)
        # search(test, contours)
        # drawContours(boxList, test)
        kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
        eroImage = cv.morphologyEx(test,cv.MORPH_OPEN, kernel)
        plt.imshow(test, cmap = 'Greys_r')
        plt.show()
        plt.imshow(eroImage, cmap = 'Greys_r')
        plt.show()
        image[np.where(mask == 255)] = 255
        return image
    ## end static method for preprocessing ##

    def detectLines(self, image_data, debug = False):
               
        image = self.negateImage(image_data.image)
        
        mode = 0
        image_data = self.getLines(image, image_data, mode, 12)  
        
        image = self.rotateImage(image)
        
        mode = 1
        image_data = self.getLines(image, image_data, mode, 7)

        image_data = self.cutLines(image_data)
        
        if debug:
            print "detectLines debugging"
            image_data.displayTargetLines('horLines')
            image_data.displayTargetLines('verLines')
        
        image_data.lineDetected = True # Update image_data status
        return image_data
        
        
    ## static method for detectLine ##
    @staticmethod
    def negateImage(image, thres = 30):
        image = 255 - image
        rat, image = cv.threshold(image, thres, 255, 0)
        return image
    
    @staticmethod
    def rotateImage(image):
        return np.rot90(image)
    
    
    @staticmethod
    # the image is in the image data
    # fill up the list in image_data
    # return the image_data
    def getLines(image, image_data, mode, minLength = 10):
        
        tmp = cv.HoughLinesP(image, rho = 9, theta = np.pi, threshold = 10, minLineLength =minLength, maxLineGap = 0)
        
        if mode==0:
            for line in tmp:
                x1, y1, x2, y2 = list(line[0])
                lineList = x1, y2, x2, y1, abs(y2-y1)
                image_data.addVerticleLine(lineList)
        elif mode == 1:
            for line in tmp:
                x1, y1, x2, y2 = line[0]
                y1 = -y1 + image_data.image_width
                y2 = -y2 + image_data.image_width
                lineList = [y1, x2, y2, x1, abs(y2-y1)]
                image_data.addHorizontalLine(lineList)
                
                
        return image_data
    

    def cutLines(self, image_data):

        newList = []
        margin = 5
        for line in image_data.horLines:
            if line[4] <20:
                newList.append(line)
            else:
                x1, y1, x2, y2, length = line
                isnotcut = True
                for vx1, vy1, vx2, vy2, vlength in image_data.verLines:
                    if x1+margin<vx1 and vx1<x2-margin and vy1<y1 and y1<vy2:
                        newline1 = [x1, y1, vx1, y2, vx1-x1]
                        newline2 = [vx1, y1, x2, y2, x2-vx1]
                        length1 = vx1 - x1
                        length2 = x2 - vx1
                        isTooShort = False
                        if length1/length < 0.35 or length2/length< 0.35:
                            isTooShort = True

                        if not isTooShort:
                            newList.append(newline1)
                            newList.append(newline2)
                            isnotcut = False
                            break


                if isnotcut:
                    newList.append(line)
        image_data.horLines = newList
        return image_data

    ## end static method for detectLine ##
    
    # Maybe you have implemented this. If positive, you can put it here
    # TODO: Sean, put your old code here
    def detectCorners(self, image_data):
        return image_data
    
    # Not implemented yet
    # TODO
    def refineLinesByCorners(self, image_data):
        return image_data
    
    ## static method for detectCorners ##
    
    ## end static method for detectCorners ##
    
    def matchLines(self, image_data, debug = True):
        
        print hasattr(image_data, 'horLines')
        print len(image_data.horLines)
        
        if image_data.lineDetected:
            image_data = self.matchParent(image_data)
            image_data = self.matchChildren(image_data)
            # image_data = self.removeText(image_data)
            
            if debug:
                image_data.displayTargetLines('parent')
                image_data.displayTargetLines('children')
                image_data.displayTargetLines('interLines')
                image_data.displayTargetLines('anchorLines')
            
            image_data.lineMatched = True
            
        else:
            print "Error! Please do detectLines before this method"
        
        
        return image_data
    
    @staticmethod
    # your display() method
    def displayTree(root):
        return
    
    
    def matchParent(self, image_data):
        horLines = sorted(image_data.horLines, key = image_data.sortByXEndFromLeft)
        verLines = sorted(image_data.verLines, key = image_data.sortByXEndFromLeft)
        margin = 5
        parent = []
        interLines = []
        jointPoints = []

        for line in horLines:
            x1, y1, x2, y2, length = line
            probParent = []
            for vline in verLines:
                vx1, vy1, vx2, vy2, vlength= vline
                if y1 > vy1 and y1 < vy2 and x2 > vx1 - margin and x2 < vx2 + margin:                   
                    probParent.append(vline)
            # print probParent
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
                interLines.append(tupleLine)
                parent.append(((tupleLine, target), minDist))
                jointPoints.append((x2-1,y2))
        parent = sorted(parent, key = image_data.sortByDist)

        parent = self.removeRepeatLines(parent)
        interLines = self.removeRepeatLinesBasic(interLines)

        image_data.parent = parent
        image_data.interLines = interLines



        return image_data
    
    
    def matchChildren(self, image_data):
        verLines = sorted(image_data.verLines, key = image_data.sortByXEndFromLeft)
        horLines = sorted(image_data.horLines, key = image_data.sortByXHead)
        margin = 5
        children= []
        anchorLines = []
        for line in verLines:
            x1, y1, x2, y2, length = line
            upperLeave = []
            lowerLeave = []
            interLeave = []
            for hline in horLines:
                hx1, hy1, hx2, hy2, hlength = hline
                isUpperLeave = False
                isLowerLeave = False
                if x1 > hx1 - margin and x1 < hx1 + margin and y1 > hy1 - margin and y1 < hy1 + margin:
                    upperLeave.append(hline)
                    isUpperLeave = True
                if x1 > hx1 - margin and x1 < hx1 + margin and y2 > hy1 - margin and y2 < hy1 + margin:
                    lowerLeave.append(hline)
                    isLowerLeave = True
                if x1 -margin < hx1  and x1 + margin > hx1  and y1 + margin < hy1 and y2 - margin > hy1:
                    if not (isUpperLeave or isLowerLeave):
                        interLeave.append(hline)
            if len(upperLeave) > 0 or len(lowerLeave) > 0 or len(interLeave) > 0:
                totalDist = 0
                minDist = None
                upTarget = None
                for upLine in upperLeave:
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
                for downLine in lowerLeave:
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

                if len(interLeave) > 0:
                    image_data.isBinary = False
                    interLeave = self.removeRepeatLinesBasic(interLeave)
                    for subline in interLeave:
                        subline = tuple(subline)
                        isAnchor = True
                        for interline in image_data.interLines:
                            if self.isSameLine(subline, interline):
                                isAnchor = False
                                break
                        if isAnchor:
                            anchorLines.append(subline)
                if upTarget:
                    upTarget = tuple(upTarget)
                    isAnchor = True
                    for interline in image_data.interLines:
                        if self.isSameLine(upTarget, interline):
                            isAnchor = False
                            break
                    if isAnchor:
                        anchorLines.append(upTarget)
                if downTarget:
                    downTarget = tuple(downTarget)
                    isAnchor = True
                    for interline in image_data.interLines:
                        if self.isSameLine(downTarget, interline):
                            isAnchor = False
                            break
                    if isAnchor:
                        anchorLines.append(downTarget)
                if not (upTarget or downTarget):
                    if image_data.isBinary:
                        children.append(((line, (upTarget, downTarget)),totalDist - 100))
                    else:
                        tmpLineList = [upTarget, downTarget]
                        for subline in interLeave:
                            tmpLineList.append(subline)
                        children.append(((line, tuple(tmpLineList)), totalDist-100))
                else:
                    if image_data.isBinary:
                        children.append(((line, (upTarget, downTarget)), totalDist))
                    else:
                        tmpLineList = [upTarget, downTarget]
                        for subline in interLeave:
                            tmpLineList.append(subline)
                        children.append(((line, tuple(tmpLineList)), totalDist))
        children = sorted(children, key = image_data.sortByDist)
        children = self.removeRepeatLines(children)
        anchorLines = self.removeRepeatLinesBasic(anchorLines)
        image_data.children = children
        image_data.anchorLines = anchorLines



        return image_data


    @staticmethod     
    def removeText(image_data):
        return image_data
    

    def removeRepeatLinesBasic(self, lineList):
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
        while i<len(lineList):
            lines, dist = lineList[i]
            x1, y1, x2, y2, length =lines[0]

            for j in xrange(len(lineList)-1, i, -1):

                if x1 - margin < lineList[j][0][0][0] and x2 + margin > lineList[j][0][0][2]:
                    if y2+margin > lineList[j][0][0][3] and y1-margin < lineList[j][0][0][1]:
                        if lineList[j][0][0][4] <= length+ (margin-2):
                            del lineList[j]

            i +=1

        return lineList


    ## end static method for matchLines ##
    
    # TODO: not implemented yet, if you have, put it here
    def getSpecies(self, image_data, debug = False):
        image_data.speciesNameReady 
        return image_data
    
    
    def makeTree(self, image_data, debug = False):
        
        if image_data.lineDetected and image_data.lineMatched:
        
            # Create node from matched lines
            image_data = self.createNodes(image_data)
            if debug:
                for node in image_data.nodeList:
                    node.getNodeInfo()
                image_data.displayNodes()


            # Gather trees from nodes
            image_data = self.createRootList(image_data)
            if debug:
                image_data.displayTrees('regular')
            image_data.displayTrees('regular')
            # Check if it's perfectly recovered
            image_data = self.checkDone(image_data)
            
            
            if not image_data.treeReady:
                ## Fix false-positive sub-trees and mandatorily connect sub-trees
                image_data = self.fixTrees(image_data)
                image_data = self.checkDone(image_data)
                image_data.displayTrees('regular')
                image_data.defineTreeHead()
                print self.treeRecover(image_data.treeHead)
                
            # fixTrees fixed everything
            if image_data.treeReady:
                image_data.defineTreeHead()
                print self.treeRecover(image_data.treeHead)
                if debug:
                    print "TODO: draw something here"
                    image_data.displayTrees('final')
            
            # something bad happens
            else:
                print "unknown bad happens"
                image_data.defineTreeHead()## For now, it still defines the tree head. However, we need something else returned to notice it's not perfect
                if debug:
                    print "TODO: draw something here"
        
        else:
            print "Error! Please do detectLine and matchLine before this method"

        return image_data
    
    
    
    @staticmethod
    def isDotWithinLine(dot, line):
        margin = 5
        x, y = dot
        x1, y1, x2, y2, length = line
        if x1-margin < x and x < x2+margin and y1 - margin < y and y < y2 + margin:
            return True
        else:
            return False
    @staticmethod
    def isLefter(branch, ref):
        x1 = branch[0]
        x2 = ref[0]

        if x2 < x1:
            return True
        return False

    @staticmethod
    def sortNodeByLeftEnd(item):
        return item.branch[0]

    def getNodeBranchOnTheRight(self, dot, nodeList):
        x, y = dot
        potentialNodes = []
        for node in nodeList:
            x1, y1, x2, y2, length = node.branch
            if y1 < y and y2 > y and x1>x:
                potentialNodes.append(node)
        if len(potentialNodes) == 0:
            return False
        else:
            potentialNodes = sorted(potentialNodes, key = self.sortNodeByLeftEnd)
            return potentialNodes[0]

    @staticmethod
    def treeRecover(rootNode):
        return rootNode.getTreeString()

    def fixTrees(self,image_data):
        rootList = image_data.rootList
        parent = image_data.parent
        children = image_data.children
        horLines = image_data.horLines
        verLines = image_data.verLines
        breakNodeList = []
        tmpList = rootList[:]
        for node in rootList:
            if node in tmpList:
                if not node.isComplete:
                    for breakNode in node.breakSpot:
                        isFixed = False
                        isUpper = True
                        if not ((breakNode.to[0] or breakNode.upperLeave) or (breakNode.to[1] or breakNode.lowerLeave)):
                            pass
                        elif (breakNode.to[0] or breakNode.upperLeave) and not (breakNode.to[1] or breakNode.lowerLeave):
                            x1, y1, x2, y2, length = breakNode.branch
                            result = self.getNodeBranchOnTheRight((x2,y2), rootList)
                            if result:  
                                to = list(breakNode.to)
                                to[1] = result
                                breakNode.to = tuple(to)
                                result.whereFrom = breakNode
                                result.origin = node
                                if result.isComplete:
                                    if result in tmpList:
                                        tmpList.remove(result)
                                node.breakSpot.remove(breakNode)
                                isFixed = True
                            else:
                                isUpper = False

                        elif not (breakNode.to[0] or breakNode.upperLeave) and (breakNode.to[1] or breakNode.lowerLeave):
                            x1, y1, x2, y2, length = breakNode.branch
                            result = self.getNodeBranchOnTheRight((x1,y1), rootList)
                            if result:
                                to = list(breakNode.to)
                                to[0] = result
                                breakNode.to = tuple(to)
                                node.breakSpot.remove(breakNode)
                                result.origin = node
                                if result.isComplete:
                                    if result in tmpList:
                                        tmpList.remove(result)
                                result.whereFrom = breakNode
                                isFixed = True


                        if isUpper:
                            breakSpot = 'upper'
                        else:
                            breakSpot = 'lower'

                else:
                    pass
        rootList = tmpList[:]
        for node in rootList:
            if node in tmpList: 
                if len(node.breakSpot) == 0 and node.whereFrom != None:

                    tmpList.remove(node)
        rootList = tmpList[:]
        if len(rootList) == 1:
            rootList[0].isComplete = True

        image_data.rootList = rootList
        image_data.parent = parent
        image_data.children = children
        return image_data


    def checkDone(self, image_data):
        rootList = image_data.rootList
        isDone = True
        rootNode = rootList[0]

        if not rootNode.isComplete:
            isDone =  False

        if rootNode.root:
            x1, y1, x2, y2, length = rootNode.root
            for node in rootList:
                if node != rootNode:
                    if self.isDotWithinLine((x1, y1), node.branch) or self.isLefter(rootNode.branch, node.branch):
                        isDone = False
        else:
            for node in rootList:
                if node != rootNode:
                    if self.isLefter(rootNode.branch, node.branch):
                        isDone = False

        image_data.treeReady = isDone
        
        return image_data


    @staticmethod
    def countArea(lineList,image_data):

        lineList = sorted(lineList, key = image_data.sortByLeftTop)

        leftTop = lineList[0]
        x1 = leftTop[0]
        y1 = leftTop[1]
        lineList = sorted(lineList, key = image_data.sortByBtmRight)
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

    def checkError(self, node, mode , image_data):
        anchorLines = image_data.anchorLines
        parent = image_data.parent
        if mode == 'upper':
            isAnchorLine = False
            if node.upperLeave:
                for line in anchorLines:
                    if self.isSameLine(line, node.upperLeave):
                        node.isUpperAnchor = True
                        isAnchorLine = True
                        return True
                if not isAnchorLine:
                    for package in parent:
                        lines, dist = package

                        if self.isSameLine(lines[0], node.upperLeave):
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
                    if self.isSameLine(line, node.lowerLeave):
                        node.isLowerAnchor = True
                        isAnchorLine = True
                        return True
                if not isAnchorLine:
                    for package in parent:
                        lines, dist = package
                        if self.isSameLine(lines[0], node.lowerLeave):
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
            if index < len(node.interLeave) and node.interLeave[index]:
                for line in anchorLines:
                    if self.isSameLine(line, node.interLeave[index]):
                        node.isInterAnchor[index] = True
                        isAnchorLine = True
                        return True
                if not isAnchorLine:
                    for package in parent:
                        lines, dist = package
                        if self.isSameLine(lines[0], node.interLeave[index]):
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

    def createRootList(self, image_data):
        nodeList = image_data.nodeList
        anchorLines = image_data.anchorLines
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
                    (seen, loop) = self.groupNodes(rootNode, seen, image_data)

        image_data.rootList = rootList
        return image_data

    def groupNodes(self, rootNode, seen, image_data):
        anchorLines = image_data.anchorLines
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
            isAnchorLine = self.checkError(rootNode, 'upper', image_data)
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
            isAnchorLine = self.checkError(rootNode, 'lower', image_data)
            if isAnchorLine:
                lineList.append(rootNode.lowerLeave)
            else:
                if rootNode.to[1]:
                    stack.append(rootNode.to[1])
                else:
                    isComplete = False
                    lineList.append(rootNode.branch)
        if not rootNode.isBinary:
            # print rootNode.getNodeInfo()
            for index, to in enumerate(rootNode.otherTo):
                if to:
                    if rootNode.branch != to.branch:
                        stack.append(to)
                    else:
                        rootNode.otherTo[index] = None
                else:
                    isAnchorLine = self.checkError(rootNode, 'inter%s' %str(index), image_data)
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
                if not self.checkError(node,'upper',image_data):
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

                if not self.checkError(node, 'lower', image_data):
                    isComplete = False
                    if node.to[1]:
                        stack.append(node.to[1])
                    else:
                        lineList.append(node.branch)
                else:
                    lineList.append(node.lowerLeave)

            if not node.isBinary:
                # print rootNode.getNodeInfo()
                for index, to in enumerate(node.otherTo):

                    if to:
                        if to not in seen:
                            seen.append(to)
                        if to not in visit and node.branch != to.branch:
                            stack.append(to)
                        else:
                            node.otherTo[index] = None
                    else:
                        isAnchorLine = self.checkError(node, 'inter%s' %str(index), image_data)
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

        area = self.countArea(lineList, image_data)
        rootNode.area = area


        if isComplete:
            rootNode.isComplete = True
        loop = False, None
        return (seen, loop)    


    def createNodes(self,image_data):
        
        if not image_data.speciesNameReady:
            print "Species names not found! Use dummy names."

        parent = image_data.parent
        children = image_data.children
        nodeList = []
        for item in children:
            lines, dist = item
            (branch, hlines) = lines
            match = False
            for pitem in parent:
                ((root, pbranch), pdist) = pitem
                if self.isSameLine(branch, pbranch):
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
                    a = Node(None, branch, upperLeave, lowerLeave)
                    a.interLeave = interLine
                    numInterLeave = len(hlines) - 2
                    for index in range(numInterLeave):
                        a.interLabel.append(None)
                        a.otherTo.append(None)
                        a.isInterAnchor.append(False)

                    a.isBinary = False

                nodeList.append(a)
        

        for node in nodeList:
            potentialUpper = []
            potentialLower = []
            potentialInter = []

            if not node.isBinary:
                for leave in node.interLeave:
                    potentialInter.append([])


            for subNode in nodeList:
                if subNode.root:
                    if node.upperLeave and not node.isUpperAnchor and self.isSameLine(node.upperLeave, subNode.root):
                        score = self.evaluateNode(subNode)
                        potentialUpper.append((subNode, score))
                    elif node.lowerLeave and not node.isLowerAnchor and self.isSameLine(node.lowerLeave, subNode.root):
                        score = self.evaluateNode(subNode)
                        potentialLower.append((subNode,score))
                    if not node.isBinary:
                        for index, leave in enumerate(node.interLeave):
                            if not node.isInterAnchor[index] and self.isSameLine(leave, subNode.root):
                                score = self.evaluateNode(subNode)
                                potentialInter[index].append((subNode, score))
            # if node.lowerLeave:
            #     if node.lowerLeave[0] > 213 and node.lowerLeave[0] < 223 and node.lowerLeave[1] >295 and node.lowerLeave[1] <305:
            #         print potentialUpper, potentialLower, potentialInter

            if len(potentialUpper) != 0:
                potentialUpper = sorted(potentialUpper, key = self.sortByScore)

                tmpTo = list(node.to)
                tmpTo[0] = potentialUpper[0][0]
                node.to = tuple(tmpTo)
                print node.to
            if len(potentialLower) != 0:
                potentialLower = sorted(potentialLower, key = self.sortByScore)

                tmpTo = list(node.to)
                tmpTo[1] = potentialLower[0][0]
                node.to = tuple(tmpTo)
            if not node.isBinary:
                for index, leave in enumerate(node.interLeave):
                    potentialInter[index] = sorted(potentialInter[index], key = self.sortByScore)
                    if len(potentialInter[index])!=0:
                        node.otherTo[index] = potentialInter[index][0]

        image_data.nodeList = nodeList
        return image_data
    @staticmethod
    def sortByScore(item):
        return -item[1]


    @staticmethod
    def evaluateNode(node):
        score = 0
        if node.upperLeave:
            score+=1
        if node.lowerLeave:
            score+=1
        if node.to[0]:
            score+=1
        if node.to[1]:
            score+=1

        return score

    @staticmethod
    def isSameLine(aline, bline, margin = 5):
        ax1, ay1, ax2, ay2, alength = aline
        bx1, by1, bx2, by2, blength = bline

        if ay1 - margin < by1 and ay2 + margin> by2 and ax1 - margin < bx1 and ax2 + margin > bx2 and alength + margin > blength:
            return True
        elif by1 - margin < ay1 and by2 + margin > ay2 and bx1 - margin < ax1 and bx2 + margin > ax2 and blength + margin > alength:
            return True
        else:
            return False        
        