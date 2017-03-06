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


from PhyloParser import *

def getFilesInFolder(folderPath):
    fileNameList = [f for f in listdir(folderPath) if isfile(join(folderPath, f))]

    return fileNameList




if __name__ == '__main__':



    ############ run experiment ######################

    # a = ExperimentExecutor()

    # a.autoRun('high_quality_tree/', 'groundTruth.csv', 'compareResult_0124.csv')


    ###################################################

# PMC2323573_pone.0002033.g002.jpg


    phyloParser = PhyloParser()
    
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

    for index in range(48, len(fileNameList)):
        print index, fileNameList[index]
        filePath = folderPath + fileNameList[index]
        fileName = fileNameList[index]
        fileName = 'PMC2614190_ebo-04-181-g01.jpg'
        # fileName = 'PMC2674049_1471-2148-9-74-4.jpg'
        # fileName = 'PMC1326215_1471-2148-5-71-6.jpg'
        # fileName = 'PMC2640363_1471-2148-9-21-5.jpg'
        filePath = folderPath + fileName
        if isfile(filePath) :
            image = cv.imread(filePath,0)
        PhyloParser.displayImage(image)

        # imageData = ImageData(image)
        # imageData = PhyloParser.preprocces(imageData, debug = False)
        # # imageData = PhyloParser.detectLines__v2(imageData, debug=True)
        # imageData = PhyloParser.testing(imageData)

        image_data = ImageData(image)
        image_data = phyloParser.preprocces(image_data, debug= False)
        # PhyloParser.displayImage(image)


        image_data = phyloParser.detectLines(image_data, debug = True)

        
        # image_data = phyloPaser.traceTree(image_data, debug = True)
        image_data = phyloParser.getCorners(image_data, debug = False)        
        image_data = phyloParser.makeLinesFromCorner(image_data, debug = False)
        image_data = phyloParser.includeLinesFromCorners(image_data)
        image_data = phyloParser.removeRepeatedLines(image_data)

        

#         # image_data = phyloParser.detectCorners(image_data)
#         # image_data = phyloParser.refineLinesByCorners(image_data)
#         # image_data = phyloPaser.getSpecies_v2(image_data, debug = True)
#         # image_data = phyloPaser.getSpecies(image_data, debug = True)
        image_data = phyloParser.matchLines(image_data, debug = True, useNew = True)
#         # image_data = phyloPaser.getSpecies_v2(image_data, debug = True)
#         # print image_data.orphanBox2Text
#         # image_data = phyloParser.getSpecies(image_data, debug = True)
        treeString = phyloParser.constructTree(image_data, debug = True, tracing = False)
# #         # image_data = phyloPaser.getSpecies(image_data)
 

#         a = ExperimentExecutor()

#         resultTree = ExperimentExecutor.getTreeObject(image_data.treeStructure)

#         groundTruthDict = ExperimentExecutor.getGroundTruthDict('groundTruth.csv')

#         groundTruthTree = ExperimentExecutor.getTreeObject(groundTruthDict[fileName])

#         score = ExperimentExecutor.getEditDistance(resultTree, groundTruthTree)        
        
# # #         print treeString
        
#         