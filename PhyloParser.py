import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

class PhyloParser():
    
    EROSION_KERNEL5 = np.array(([0,0,1,0,0], [0,0,1,0,0], [1,1,1,1,1], [0,0,1,0,0], [0,0,1,0,0]), np.uint8)
    EROSION_KERNEL3 = np.array(([1,1,1], [1,1,1], [1,1,1]), np.uint8)
    EROSION_KERNEL1 = np.array(([1,1]), np.uint8)
        
    def __init__(self):
        print "ddd"
        
        
    def preprocces(self, image, debug = False):
        
        image = self.downSample(image)
        if debug:
            self.displayImage(image)
            
        image = self.bilateralFilter(image)
        
        if debug:
            self.displayImage(image)
            
        image = self.bolding(image)
        
        if debug:
            self.displayImage(image)
            
        return image
        
    ## static method for preprocessing ##
    
    @staticmethod
    def sobelFilter(image, sigma=3, k=5):
        return image
    
    @staticmethod 
    def gaussianBlur(image, kernelSize = (3,3), sigma= 1.0):
        return image
    
    @staticmethod 
    ## Old bolding, duplicate with threshold
    def binarize(self, image, thres=180):
        return self.threshold(image, thres=thres)
    
    @staticmethod
    def bilateralFilter(image, radius = 3, sigmaI = 30.0, sigmaS = 3.0):
        return image

    @staticmethod
    def downSample(image, parameter = 500.0):
        return image

    @staticmethod
    def erosion(image, kernel = PhyloParser.EROSION_KERNEL3):
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
    # this is duplicate with binarize(bolding), keep only one. 
    # i have changed the binarize to call this method
    def threshold(image, thres=190):
        return cv.threshold(image, thres, 255, 0)
        
    @staticmethod
    def displayImage(image):
        plt.imshow(image, cmap='Greys_r')
        plt.show()

    ## end static method for preprocessing ##
    
    
    
    def detectLines(self, image, debug = False):
        
        image_data = ImageData(image)
        
        self.negateImage(image)
        
        mode = 0
        image_data = self.getLines(image_data, mode, 12)  
        
        self.rotateImage(image)
        
        mode = 1
        image_data = self.getLines(image_data, mode, 7)
        
        self.rotateImage()
        self.rotateImage()
        self.rotateImage()
        
        image_data.sortLines()
        
        if debug:
            image_data.displayLines()
        
        return image_data
        
        
    ## static method for detectLine ##
    @staticmethod
    def negateImage(image, thres = 30):
        return image
    
    @staticmethod
    def rotateImage(image):
        return np.rot90(image)
    
    
    @staticmethod
    # the image is in the image data
    # fill up the list in image_data
    # return the image_data
    def getLines(image_data, mode, minLength):
        
        tmp = cv.HoughLinesP(image_data.getImage(), rho = 9, theta = np.pi, threshold = 10, minLineLength =minLength, maxLineGap = 0)
        
        if mode==0:
            for line in tmp:
                x1, y1, x2, y2 = list(line[0])
                lineList = x1, y2, x2, y1, abs(y2-y1)
                image_data.addHorizontalLine(lineList)
        elif mode == 1:
            for line in tmp:
                x1, y1, x2, y2 = line[0]
                y1 = -y1 + image_data.image_width
                y2 = -y2 + image_data.image_width
                lineList = [y1, x2, y2, x1, abs(y2-y1)]
                image_data.addVerticleLine(lineList)
                
        return image_data
    
    ## end static method for detectLine ##
    
    # Maybe you have implemented this. If positive, you can put it here
    def detectCorners(self, image_data):
        return image_data
    
    # Not implemented yet
    def refineLinesByCorners(self, image_data):
        return image_data
    
    ## static method for detectCorners ##
    
    ## end static method for detectCorners ##
    
    def matchLines(self, image_data, debug = True):
        
        image_data = self.matchParent(image_data)
        image_data = self.matchChildren(image_data)
        image_data = self.removeText(image_data)
        
        #..... Keep going as this guideline ....
        
        
        if debug:
            self.displayTree(root)
        
        treeString = root.toString()
        
        return treeString
    
    @staticmethod
    # your display() method
    def displayTree(root):
        return
    
    @staticmethod     
    def matchParent(image_data):
        return image_data
    
    @staticmethod     
    def matchChildren(image_data):
        return image_data
    
    @staticmethod     
    def removeText(image_data):
        return image_data
    
    ## end static method for matchLines ##
    
    
# All data about the image stored in this instance
# Methods importing, exporting, sorting internal data placed in this instance
# Methods, processing image, extracting data from image placed in PhyloParser and store the ouput in this instande
class ImageData():
    
    # filled by detectLine
    image = None
    image_height = None
    image_width = None
    linesList = []
    horLines = []
    verLines = []
    
    # filled by matchLine
    cleanImage = None
    parent= []
    children = []
    anchorLines = []
    interLines = []
    jointLines = []
    jointPoints = []
    upperCorners=[]
    lowerCorners = []
    length = []
    thickness = []
    
    def __init__(self, image):
        self.image = image
        (self.image_height, self.image_width) = image.shape
        self.linesList = [] #not use??
        self.horLines = []
        self.verLines = []
        
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

        
    # make a staticmethod sort everything you want depending on the input
    # ex: original sortLines will likely call as
    # sort("verlines", ImageData.lineGetKey)
    # then you need only different keys as methods, you don't need many sort methods.
    @staticmethod    
    def sort(self, target, key):
        try:
            list = getattr(self, target)
            try:
                list = sorted(list, key=key)
            except:
                print "Given key is not valid"
            setattr(self, target, list)
        except:
            print "ImageData has no attribute %s"%target

    def sortLines(self): #change the method that calls this method as sort(target, key) 
        self.verLines = sorted(self.verLines, key=self.lineGetKey) # change the method name, consistent with other sort by
        self.horLines = sorted(self.horLines, key=self.lineGetKey)

    # change the method name, consistent with other sort by
    def lineGetKey(self, item):
        return (pow(item[1], 2.0) + pow(item[0], 2.0) , item[0], item[1])
    
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
    
    def displayLines(self, rad=2):
        for spot in self.verLines:
            cv.rectangle(self.image, (spot[0]-rad, spot[1]-rad), (spot[2]+rad, spot[3]+rad), color=(255), thickness=0)
    
        for spot in self.horLines:
            cv.rectangle(self.image, (spot[0]-rad, spot[1]-rad), (spot[2]+rad, spot[3]+rad), color=(255), thickness=0)
    
    def addHorizontalLine(self, line):
        self.horLines.append(line)
        
    def addVerticleLine(self, line):
        self.verLines.append(line)  
    
    def getLineLists(self):
        return (self.verLines, self.horLines)

    def getImage(self):
        return self.image
    
    def displayImage(self):
        plt.imshow(self.image, cmap='Greys_r')
        plt.show()
        
## Need modified
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
            #     tmp1.append(a)
            #     tmp2.append(b)
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
        
        