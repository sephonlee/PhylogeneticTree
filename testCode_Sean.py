# Phylogenetic Tree Parser Tutorial

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
from GroundTruthConverter import *

from PhyloParser import *

def getFilesInFolder(folderPath):
    fileNameList = [f for f in listdir(folderPath) if isfile(join(folderPath, f))]

    return fileNameList

# class multiParsing():
#     def __init__():
#         clfPath = 'RF.pkl'
#         image = ''
#     def 


if __name__ == '__main__':



    ############ run experiment ######################

    # a = ExperimentExecutor()

    # a.autoRun('high_quality_tree/', 'groundTruth.csv', 'compareResult_0124.csv')


    ###################################################

# PMC2323573_pone.0002033.g002.jpg

    clfPath = 'RF.pkl'
    phyloParser = PhyloParser(clfPath = clfPath)
    
    folderPath = 'images/'
    folderPath = 'high_quality_tree/'

    fileNameList = getFilesInFolder(folderPath)

    fileNameList.sort()
    # with open('image_fileName_list.txt', 'r')as f:
    #     fileNameList = f.readlines()

    # testingFilePath = [join(folderPath, x) for x in fileNameList]
    # result = {}
    # groundTruth = {}
    # a = ExperimentExecutor()
    # a.execute(result, groundTruth, testingFilePath)



    for index, fileName in enumerate(fileNameList):
        print index, fileName
            # f.write(fileName + '\n')
    fileNameList = [x.rstrip() for x in fileNameList]
    # fileNameList = [folderPath + x for x in fileNameList]

    for index in range(15, len(fileNameList)):
        # try:
        print index, fileNameList[index]
        filePath = folderPath + fileNameList[index]
        fileName = fileNameList[index]
        # fileName = 'PMC2614190_ebo-04-181-g01.jpg'
        # fileName = 'PMC2674049_1471-2148-9-74-4.jpg'
        # fileName = 'PMC1326215_1471-2148-5-71-6.jpg'
        # fileName = 'PMC2644698_1471-2229-8-133-4.jpg'
        # fileName = 'PMC2644698_1471-2229-8-133-4.jpg'
        # fileName = 'PMC2775678_IJMB2009-701735.001.jpg'
        # fileName = 'PMC2697986_1471-2148-9-107-3.jpg'
        # fileName = 'PMC1182362_1471-2148-5-38-3.jpg'
        filePath = folderPath + fileName
        # filePath = 'treeset/images/multi/1471-2148-10-52-2-l.jpg'
        if isfile(filePath) :
            image = cv.imread(filePath,0)
        # PhyloParser.displayImage(image)





        # imageData = ImageData(image)
        # imageData = PhyloParser.preprocces(imageData, debug = False)
        # # imageData = PhyloParser.detectLines__v2(imageData, debug=True)
        # imageData = PhyloParser.testing(imageData)

        image_data = ImageData(image)
        image_data = phyloParser.preprocces(image_data, debug= False)
        # PhyloParser.displayImage(image)


        image_data = phyloParser.detectLines(image_data, debug = False)

        
        # image_data = phyloPaser.traceTree(image_data, debug = True)
        image_data = phyloParser.getCorners(image_data, debug = False)        
        image_data = phyloParser.makeLinesFromCorner(image_data, debug = False)
        image_data = phyloParser.includeLinesFromCorners(image_data)

        image_data = phyloParser.postProcessLines(image_data)



        # image_data = phyloParser.removeRepeatedLines(image_data)
        image_data = phyloParser.groupLines(image_data, debug = False)
        image_data = phyloParser.matchLineGroups(image_data, debug = False)

        

#         # image_data = phyloParser.detectCorners(image_data)
#         # image_data = phyloParser.refineLinesByCorners(image_data)
#         # image_data = phyloPaser.getSpecies_v2(image_data, debug = True)
#         # image_data = phyloPaser.getSpecies(image_data, debug = True)
        # image_data = phyloParser.matchLines(image_data, debug = True, useNew = False)
        # image_data = phyloParser.getSpecies_v2(image_data, debug = True)
#         # print image_data.orphanBox2Text
        image_data = phyloParser.getSpecies_v3(image_data, debug = False)
        # testString = phyloParser.makeTree(image_data, debug = True, tracing = False)
        # treeString = phyloParser.constructTreeByTracing(image_data, debug = False)
        treeString = phyloParser.constructTree_eval(image_data, debug = False, tracing = False)
        print treeString.treeStructure

        dbdata = treeString.treeHead.getParentChildrenStyle()
        print dbdata
        # print dbdata['cluster']

        with open('dbdata_2.csv', 'ab') as f:
            csvwriter = csv.writer(f, delimiter='\t')
            for i in range(dbdata['relation_index'] + 1):
                if i >= dbdata['count_labels']:
                    string = ''
                    for node in dbdata[i]:
                        string += '#' + str(node)
                    csvwriter.writerow([fileName, str(i), string])
                else:
                    csvwriter.writerow([fileName, str(i), dbdata[i]])
        with open('dbdata_cluster.csv','ab') as f:
            csvwriter = csv.writer(f, delimiter = '\t')
            if 'root' in dbdata['cluster']:
                print dbdata['cluster']['root']
                csvwriter.writerow([fileName, str(dbdata['relation_index']), dbdata['cluster']['root']])
            for i in range(dbdata['relation_index']):
                csvwriter.writerow([fileName, str(i), dbdata['cluster'][i]])
                    
        # except:
        #     pass
#         truth_string = string2TreeString(ground_truth[fileNameList[index]], rename = True)
#         print image_data.treeStructure
#         t1 = PhyloTree(image_data.treeStructure + ";")
#         t2 = PhyloTree(truth_string + ";")
#         PhyloTree.rename_node(t1, rename_all=True)
#         PhyloTree.rename_node(t2, rename_all=True)
        
#         print t1
# #             t1.drawTree()
#         print t2
#         distance = PhyloTree.zhang_shasha_distance(t1, t2)
#         num_node = PhyloTree.getNodeNum(t2)
#         num_leaf = t2.getLeafCount()
#         score = distance/float(num_node)
        
#         print "distance %d/%d = %f, leave=%d" %(distance, num_node, score, num_leaf)
#         print "contour count=%d , sharebox count=%d"%(image_data.count_contourBoxes, image_data.count_shareBox)
#         results.append([fileNameList[index], distance, num_node, score, num_leaf, image_data.count_contourBoxes, image_data.count_shareBox])

        

# #         # image_data = phyloPaser.getSpecies(image_data)
 
        
        # a = ExperimentExecutor()

        # resultTree = ExperimentExecutor.getTreeObject(image_data.treeStructure)

        # groundTruthDict = ExperimentExecutor.getGroundTruthDict('groundTruth.csv')

        # groundTruthTree = ExperimentExecutor.getTreeObject(groundTruthDict[fileName])

        # score = ExperimentExecutor.getEditDistance(resultTree, groundTruthTree)        
 

        # distance = PhyloTree.zhang_shasha_distance(resultTree, groundTruthTree)
        # num_node = PhyloTree.getNodeNum(groundTruthTree)
        # num_leaf = groundTruthTree.getLeafCount()   

        # print distance, num_node, num_leaf     
# # #         print treeString
        
#         