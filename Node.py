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
        if not self.isBinary:
            for index, leave in enumerate(self.interLeave):
                print 'one of the interleavs: ', leave
                print 'this leave goes to the node: ', self.otherTo[index]
        print '------------------------------'


    def getLabel(self):
        pass

    def sortByY(self, item):
        return item[0][1]

    
    def getTreeString(self):
        return self.printTree(0)[0]

    def printTree(self, speciesIndex):

        if self.to[0]:
            upperChildren, speciesIndex = self.to[0].printTree(speciesIndex)
        else:
            if self.upperLabel:
                upperChildren = self.upperLabel
            elif self.isUpperAnchor:
                upperChildren = "%s" %str(speciesIndex)
                speciesIndex+=1
            else:
                upperChildren = "**"
        if self.to[1]:
            lowerChildren, speciesIndex = self.to[1].printTree(speciesIndex)
        else:
            if self.lowerLabel:
                lowerChildren = self.lowerLabel
            elif self.isLowerAnchor:
                lowerChildren = "%s" %str(speciesIndex)
                speciesIndex+=1
            else:
                lowerChildren = "**"

        if self.isBinary:
            return "(%s, %s)" %(upperChildren, lowerChildren), speciesIndex
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
                    interChildren, speciesIndex = to.printTree(speciesIndex)
                else:
                    if self.interLabel[index]:
                        interChildren = self.interLabel
                    elif self.isInterAnchor[index]:
                        interChildren = "%s" %str(speciesIndex)
                        speciesIndex+=1
                    else:
                        interChildren = "**"
                result += interChildren + ','

            return result + '%s)' %lowerChildren, speciesIndex
        