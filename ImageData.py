# All data about the image stored in this instance
# Methods importing, exporting, sorting internal data placed in this instance
# Methods, processing image, extracting data from image placed in PhyloParser and store the ouput in this instande
import cv2 as cv
from matplotlib import pyplot as plt

class ImageData():
    
    originalImage = None
    
    # filled by detectLine
    image = None
    image_height = None
    image_width = None
    linesList = []
    horLines = []
    verLines = []
    lineDetected = False
    
    # filled by detectCorner
    image_preproc_for_corner = None
    upCornerList = [] # top left corners
    downCornerList = [] # bottom left corners
    jointUpList = [] # joint points to parent
    jointDownList = [] # joint points to children
    pointSet_ver = [] #vertical line between corners
    upPointSet_hor = [] #horizontal line between top left corners and corresponding joints
    downPointSet_hor = [] #horizontal line between bottom left corners and corresponding joints
    cornerDetected = False
    lineDetectedFromCorners = False
    
    # filled by refineLine
    horLineGroup = []
    verLineGroup = []
    lineRefined = False
    
    # filled by matchLine
    cleanImage = None
    parent= []
    children = []
    anchorLines = []
    interLines = []
    jointLines = []
    isBinary = True
    lineMatched = False

    # filled by getSpecies
    species = []
    speciesNameReady = False

    # filled by makeTree
    nodeList = []
    rootList = []
    treeHead = None # Head of the tree
    treeReady = False

    
    def __init__(self, image):
        self.image = image
        self.originalImage = image.copy()
        (self.image_height, self.image_width) = image.shape
        self.horLines = []
        self.verLines = []       
        self.parent= []
        self.children = []
        self.anchorLines = []
        self.interLines = []
        self.nodeList = []
        self.rootList = []
        self.varianceMask = None
        self.isBinary = True

    def __str__(self):
        return "ImageData (%d,%d)"%(self.image_height, self.image_width)
        
        
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

    @staticmethod
    def sortByLeftTop(item):
        criteria = pow(item[0],2) + pow(item[1], 2)
        return criteria, item[2]-item[0]
    @staticmethod
    def sortByBtmRight(item):
        criteria = pow(item[2], 2) + pow(item[3],2)
        return -criteria, item[3] - item[1]
    @staticmethod
    def sortByXEnd(item):
        return (-item[2])
    @staticmethod
    def sortByXEndFromLeft(item):
        return (item[2])
    @staticmethod
    def sortByXHead(item):
        return (item[0], item[1])
    @staticmethod
    def sortByDist(item):
        return (-item[1])
    @staticmethod    
    def sortByY(item):
        return item[1]
    @staticmethod
    def sortByLengthAndDist(item):
        return (item[0][0][4], item[1])
    @staticmethod
    def sortByLength(item):
        return (-item[4])
    @staticmethod
    def sortByNodeNum(item):
        return -item.numNodes
   
    def addHorizontalLine(self, line):
        self.horLines.append(line)
        
    def addVerticleLine(self, line):
        self.verLines.append(line)  
    
    def updateImage(self, image):
        self.image = image
        (self.image_height, self.image_width) = self.image.shape
        
    def updateImageDimension(self):
        self.image_height, self.image_width = self.image.shape

    def getLineLists(self):
        return (self.verLines, self.horLines)

    def getImage(self):
        return self.image

    @staticmethod 
    def getColor(count):
        if count %5 == 0:
            return (255, 0, 0)
        elif count % 5 == 1:
            return (0, 255, 0)
        elif count % 5 == 2:
            return (0, 0, 255)
        elif count % 5 == 3:
            return (255, 0, 255)
        else:
            return (0, 255, 255)


    def displayTrees(self, target = 'regular'):
        if len(self.image.shape) ==2:
            whatever = self.image.copy()
            whatever = cv.cvtColor(whatever, cv.COLOR_GRAY2RGB)
        else:
            whatever = image.copy()
        count = 0
        for rootNode in self.rootList:

            stack = []
            stack.append(rootNode)
            color = self.getColor(count)
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
            if target == 'final':
                break
            count +=1
        plt.imshow(whatever)
        plt.show()

    def displayNodes(self):
        if len(self.image.shape) ==2:
            whatever = self.image.copy()
            whatever = cv.cvtColor(whatever, cv.COLOR_GRAY2RGB)
        else:
            whatever = image.copy()

        count = 0
        for node in self.nodeList:
            count +=1
            color = self.getColor(count)
            if node.root:
                x1, y1, x2, y2, length = node.root
                cv.rectangle(whatever, (x1, y1), (x2, y2), color=color, thickness=2)
            if node.branch:
                x1, y1, x2, y2, length = node.branch
                cv.rectangle(whatever, (x1, y1), (x2, y2), color=color, thickness=2)
            if node.upperLeave:
                x1, y1, x2, y2, length = node.upperLeave
                cv.rectangle(whatever, (x1, y1), (x2, y2), color=color, thickness=2)
            if node.lowerLeave:
                x1, y1, x2, y2, length = node.lowerLeave
                cv.rectangle(whatever, (x1, y1), (x2, y2), color=color, thickness=2)
            if not node.isBinary:
                for line in node.interLeave:
                    x1, y1, x2, y2, length = line
                    cv.rectangle(whatever, (x1, y1), (x2, y2), color=color, thickness=2)
            print node.getNodeInfo()
            plt.imshow(whatever)
            plt.show()




    def displayTargetLines(self, target):
        
        print "Display %s"%(target)
        
        if len(self.image.shape) == 2:
            whatever = cv.cvtColor(self.image, cv.COLOR_GRAY2RGB)

        list = getattr(self, target)
        if target == 'parent' or target == 'children':
            count = 0
            for ((line, rlines), dist) in list:
                x1, y1, x2, y2, length = line
                count +=1
                color = self.getColor(count)
                cv.rectangle(whatever, (x1, y1), (x2, y2), color=color, thickness=2)
                if isinstance(rlines[0], tuple) or not rlines[0]:
                    for subline in rlines:
                        if subline:
                            hx1, hy1, hx2, hy2, hlength = subline
                            cv.rectangle(whatever, (hx1, hy1), (hx2, hy2), color=color, thickness=2)
                else:
                    hx1, hy1, hx2, hy2, hlength = rlines
                    cv.rectangle(whatever, (hx1, hy1), (hx2, hy2), color=color, thickness=2)
        else:
            for line in list:
                x1, y1, x2, y2, length = line
                cv.rectangle(whatever, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
        plt.imshow(whatever)
        plt.show()
   
    def displayImage(self):
        plt.imshow(self.image, cmap='Greys_r')
        plt.show()
        
    
    def defineTreeHead(self):
        self.treeHead = self.rootList[0]
        
    # return final tree structure described by string
    def getTreeString(self):
        if self.treeReady and self.treeHead is not None:
            treeString = self.treeHead.getTreeString()
            return treeString
        else:
            print "something bad happens" 
            treeString = self.treeHead.getTreeString()## For now, it still defines the tree head. However, we need something else returned to notice it's not perfect
            return treeString ## For now, it still defines the tree head. However, we need something else returned to notice it's not perfect
                