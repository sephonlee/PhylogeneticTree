## Need modified
class Node():
    def __init__(self, root = None, branch = None, upperLeave = None, lowerLeave = None):
        self.root = root
        self.branch = branch #vertical line
        self.upperLeave = upperLeave  #top horizontal line
        self.interLeave = [] #connected horizontal line, indicating non-binary tree
        self.lowerLeave = lowerLeave #bottom horizontal line
        self.to = (None, None) #(top children node, bottom children node)
        self.otherTo = [] #list of connected children (not top bot bottom)
        self.whereFrom = None #parent node
        self.origin = None
        self.isRoot = False
        self.isBinary = False
        self.numNodes = 1
        self.isUpperAnchor = False
        self.isLowerAnchor = False
        self.isInterAnchor = []
        self.isComplete = False
        self.upperLabel = None
        self.lowerLabel = None
        self.interLabel = []
        self.area = None
        self.breakSpot = [] #only root has something in the list, first created by createRootList, also used by 
        self.status = 0
        self.isConnected = False
        self.score = None # update by method evaluateNode
        self.nodesNetwork = [] #update by method createNodes

        # verification
        self.interAnchorVerification = []
        self.biAnchorVerification = (None, None) #(top children node, bottom children node) 1: suspicious, -1: unsure, 0: verified positive, 7: not leave
 

        # Only enable for root node
        self.verifiedAnchorLines = None #checked by PhyloParser.getSuspiciousAnchorLline
#         self.unsureAnchorLines = None #checked by PhyloParser.getSuspiciousAnchorLline
        self.suspiciousAnchorLines = None #checked by PhyloParser.getSuspiciousAnchorLline
        
       

        self.nodesIncluded = [] # updated by fixTree




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
        print 'upperLeave: ', self.upperLeave, 'isAnchor? ', self.isUpperAnchor
        print 'lowerLeave: ', self.lowerLeave, 'isAnchor? ', self.isLowerAnchor
        if self.to[0]:
            print 'upperLeave goes to:', self.to[0].branch
        if self.to[1]:
            print 'lowerLeave goes to:', self.to[1].branch
        if self.origin:
            print 'origin node is:', self.origin.branch
        if not self.isBinary:
            for index, leave in enumerate(self.interLeave):
                print 'one of the interleavs: ', leave, 'isAnchor? ', self.isInterAnchor[index]
                print 'this leave goes to the node: ', self.otherTo[index]
                if leave[0] == 75 and leave[1] == 354:
                    self.otherTo[index].getNodeInfo()
        if self.whereFrom:
            print 'Connected from the node (branch): ', self.whereFrom.branch
        print 'node score: ', self.score
        print 'node isComplete? ', self.isComplete
        print 'node breakSpot:', self.breakSpot
        print '------------------------------'

    def getAllLines(self):
        lines = []
        if self.upperLeave:
            lines.append(self.upperLeave)
        if self.lowerLeave:
            lines.append(self.lowerLeave)
        if self.branch:
            lines.append(self.branch)
        if self.root:
            lines.append(self.root)
        if not self.isBinary:
            lines = lines + self.interLeave

        return lines


    def getLabel(self):
        pass

    def sortByY(self, item):
        return item[0][1]

#     def getTreeSpecies(self, speciesList):
# 
#         if self.to[0]:
#             upperChildren, speciesIndex = self.to[0].getTreeSpecies(speciesList)
#         else:
#             if self.upperLabel:
#                 upperChildren = self.upperLabel
#             elif self.isUpperAnchor:
#                 if self.upperLeave in speciesList:
#                     upperChildren = speciesList[self.upperLeave]
#                 else:
#                     upperChildren = "%s" %str(speciesIndex)
#                     speciesIndex+=1
#             else:
#                 upperChildren = "**"
#         if self.to[1]:
#             lowerChildren, speciesIndex = self.to[1].getTreeSpecies(speciesList)
#         else:
#             if self.lowerLabel:
#                 lowerChildren = self.lowerLabel
#             elif self.isLowerAnchor:
#                 if self.lowerLeave in speciesList:
#                     lowerChildren = speciesList[self.lowerLeave]
#                 else:
#                     lowerChildren = "%s" %str(speciesIndex)
#                     speciesIndex+=1
#             else:
#                 lowerChildren = "**"
# 
#         if self.isBinary:
#             return "(%s, %s)" %(upperChildren, lowerChildren), speciesIndex
#         else:
#             result = "(%s," %upperChildren
# 
#             for index, to in enumerate(self.otherTo):
#                 if to:
#                     interChildren, speciesIndex = to.getTreeSpecies(speciesList)
#                 else:
#                     if self.interLabel[index]:
#                         interChildren = self.interLabel
#                     elif self.isInterAnchor[index]:
#                         interChildren = "%s" %str(speciesIndex)
#                         speciesIndex+=1
#                     else:
#                         interChildren = "**"
#                 result += interChildren + ','
# 
#             return result + '%s)' %lowerChildren, speciesIndex        

    def getTreeSpecies(self, speciesList):
        return

    def getTreeString(self, useText = False):
#         print self.printTree(0)
        return self.printTree(0, useText=useText)[0]



    def getParentChildrenStyle(self):
        treeString = self.getTreeString()
        count_labels = treeString.count(',') + 1
        count_relations = treeString.count('(')
        data = {'count_labels': count_labels,
                'count_relations': count_relations,
                'label_index':0,
                'relation_index':count_labels,
                'children':[],
                'current_cluster':'',
                'cluster_index':None,
                'cluster_string':'',
                'cluster':{}} 

        return self.outputDBStyle(data)       

    def outputDBStyle(self, data, useText = True):

        # if relation_index==None:
        #     relation_index == count_labels

#         print "branch", self.branch
#         print self.upperLeave
#         print self.interLeave
#         print self.lowerLeave
        children = []
        isRelation = False
        
        if data['cluster_index'] == None:

            data['cluster_index'] = 0
            data['cluster']['root'] = '0'
            data['current_cluster'] = '0'
            cluster_string = '0'
        else:
            cluster_string = data['current_cluster'] + ':' + str(data['cluster_index'])
            data['current_cluster'] = cluster_string
        if self.to[0]:
            data['cluster_index'] = 0
            data = self.to[0].outputDBStyle(data)
            data[data['relation_index']] = data['children'][:]
            data['cluster'][data['relation_index']] = data['cluster_string']
            children.append(data['relation_index'])
            data['relation_index'] +=1 
        else:
            if self.upperLabel:
                upperChildren = self.upperLabel

            # elif self.isUpperAnchor:
            #     upperChildren = "%s" %str(speciesIndex)
            #     speciesIndex+=1
            else:
#                 print cluster_string + ':1'
#                 print 'here', self.biAnchorVerification[0]
#                 print self.upperLeave
#                 print
                upperChildren = "**"
            data[data['label_index']] = upperChildren
            children.append(data['label_index'])
            data['cluster'][data['label_index']] = cluster_string + ':0'
            data['label_index']+=1 

        if self.to[1]:
            data['cluster_index'] = 1
            data = self.to[1].outputDBStyle(data)
            data[data['relation_index']] = data['children'][:]
            children.append(data['relation_index'])
            data['cluster'][data['relation_index']] = data['cluster_string']
            data['relation_index'] +=1 
        else:
            if self.lowerLabel and useText:
                lowerChildren = self.lowerLabel
            # elif self.isLowerAnchor:
            #     lowerChildren = "%s" %str(speciesIndex)
            #     speciesIndex+=1
            else:
                lowerChildren = "**"
            data[data['label_index']] = lowerChildren
            data['cluster'][data['label_index']] = cluster_string + ':1'
            children.append(data['label_index'])
            data['label_index'] +=1
        if self.isBinary:
            data['children'] = children[:]
            data['cluster_string'] = cluster_string
            if data['relation_index'] +1  == data['count_labels'] + data['count_relations']:
                data[data['relation_index']] = children[:]

            return data
        else:
            
            # # result = "(%s," %upperChildren
            cluster_index = 2
            for index, to in enumerate(self.otherTo):
                if to:
                    data['cluster_index'] =cluster_index
                    cluster_index +=1
                    data = to.outputDBStyle(data)
                    data[data['relation_index']] = data['children'][:]
                    data['cluster'][data['relation_index']] = data['cluster_string']
                    children.append(data['relation_index'])
                    data['relation_index'] +=1 
                else:
                    if self.interLabel[index] and useText:
                        interChildren = self.interLabel[index]
                    else:
                        interChildren = '**'
                    children.append(data['label_index'])
                    data[data['label_index']] = interChildren
                    data['cluster'][data['label_index']] = cluster_string + ':' + str(cluster_index)
                    cluster_index +=1
                    data['label_index'] +=1

            # for index, to in enumerate(self.otherTo):
                
            #     if to:
            #         interChildren, speciesIndex = to.printTree(speciesIndex, useText=useText)
            #     else:
            #         if self.interLabel[index] and useText:
            #             interChildren = self.interLabel[index]
            #         # elif self.isInterAnchor[index]:
            #         #     interChildren = "%s" %str(speciesIndex)
            #         #     speciesIndex+=1
            #         else:
            #             interChildren = "**"
            #         data[data['label_index']] = interChildren
            #         data['label_index']+=1    
            #     # result += interChildren + ','
            if data['relation_index'] +1  == data['count_labels'] + data['count_relations']:
                data[data['relation_index']] = children[:]

            data['cluster_string'] = cluster_string
            data['children'] = children[:]
            return data


    def printTree(self, speciesIndex, useText = False, removeSuspicious = False):

#         print "branch", self.branch
#         print self.upperLeave
#         print self.interLeave
#         print self.lowerLeave
        
        if self.to[0]:
            upperChildren, speciesIndex = self.to[0].printTree(speciesIndex, useText=useText)
        else:
            if self.upperLabel and useText:
                upperChildren = self.upperLabel
            elif self.isUpperAnchor:
                upperChildren = "%s" %str(speciesIndex)
                speciesIndex+=1
            else:
#                 print 
#                 print 'here', self.biAnchorVerification[0]
#                 print self.upperLeave
#                 print
                upperChildren = "**"
        if self.to[1]:
            lowerChildren, speciesIndex = self.to[1].printTree(speciesIndex, useText=useText)
        else:
            if self.lowerLabel and useText:
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

            for index, to in enumerate(self.otherTo):
                
                if to:
                    interChildren, speciesIndex = to.printTree(speciesIndex, useText=useText)
                else:
                    if self.interLabel[index] and useText:
                        interChildren = self.interLabel[index]
                    elif self.isInterAnchor[index]:
                        interChildren = "%s" %str(speciesIndex)
                        speciesIndex+=1
                    else:
                        interChildren = "**"
                        
                result += interChildren + ','

            return result + '%s)' %lowerChildren, speciesIndex
        
