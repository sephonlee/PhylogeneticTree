# Phylogenetic Tree Parser Tutorial

# from readCorners import readCorners
# import textDetector
# import textRemover
import cv2 as cv
import time
import os
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
    
#     startTime = time.time()
#     
#     ss = 0
#     for i in range(0, 1000000):
#         ss +=i
#     
#     mm = 1
#     for i in range(0, 0000000):
#         mm *= i
#         
#     kk = 1
#     for i in range(0, 1000000):
#         kk += i*10
#     
#     print time.time()-startTime
#     
#     
#     startTime = time.time()
#     ss = 0
#     mm = 1
#     kk = 1
#     for i in range(0, 1000000):
#         ss +=i
#         mm *= i
#         kk += i*10
#     
#     print time.time()-startTime
    
    boundaries = [[335, 351, 185, 408], [351, 365, 212, 470]]
    boxes = [[[336, 344, 382, 385], [335, 344, 315, 333], [336, 345, 337, 342], [336, 345, 389, 408], [335, 345, 185, 257], [335, 349, 262, 310], [335, 349, 346, 404]], [[355, 362, 446, 448], [355, 363, 452, 470], [354, 363, 410, 435], [355, 365, 212, 292], [354, 365, 298, 347], [354, 365, 346, 404]]]

    boundaries, boxes = PhyloParser.stitchBoundries(boundaries, boxes)
    print boundaries
    print boxes
    
#     new_boundaries, boxes =  PhyloParser.splitBoundaries(boundaries, boxes)

    phyloPaser = PhyloParser()
    
    filename = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/PMC1474148_ijbsv02p0133g05.jpg'
    filename = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/PMC2324105_1471-2148-8-100-1.jpg'

    filename = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/PMC2323573_pone.0002033.g002.jpg'
    filename = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/image_5569.jpg'

    filename = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/PMC2323573_pone.0002033.g002.jpg'
#     filename = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/PMC2761929_1471-2180-9-208-3.jpg'




    # bg example
#     filename = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/PMC2287175_1471-2148-8-57-2.jpg'
#     filename = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/image_336.jpg'
#     filename = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/image_337.jpg'
#     filename = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/tree16.jpg'
    ##
#     filename = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/PMC2397417_1471-2164-9-215-5.jpg'
#     filename = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/high_quality_tree/PMC1660551_1471-2148-6-95-3.jpg'
#     filename = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/PMC2259423_1471-2105-9-S1-S22-5.jpg'
    # wide line example
#     filename = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/PMC2467406_1471-2148-8-193-4.jpg'
    
    # wide line example
    filename = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/PMC184354_1471-2148-3-16-2.jpg"
    
#     filename = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/PMC184354_1471-2148-3-16-7.jpg"
    
    #perfect case
#     filename = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/image_5217.jpg"
    #hard case
#     filename = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/PMC2174949_1471-213X-7-118-3.jpg"
#     filename = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/PMC2614493_gb-2008-9-11-r161-3.jpg"
#     filename = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/image_6477.jpg"
#     filename = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/PMC428588_1741-7007-2-13-4.jpg"
#     filename = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/PMC535349_1471-2148-4-46-1.jpg"
#     filename = "/Users/sephon/Downloads/tree3.jpg"

#     mask = np.ones((663,600), dtype = np.uint8)
#     mask[0:,385:] = 0
    
    
#     filename = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/14087683_10153822638021658_1412380893_o.jpg"

    filename = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/PMC2614190_ebo-04-181-g01.jpg"

#     print "here"
#     image = np.zeros([12, 20]) + 255
#     image[5:7, 0:16] = 0
#     image[:, 16:19] = 0
#     PhyloParser.displayImage(image)
#     
#     PhyloParser.detectCorners(image, 2)

    path = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/high_quality_tree"
    
    fileList = []
    for dirPath, dirNames, fileNames in os.walk(path):   
        for f in fileNames:
            extension = f.split('.')[-1].lower()
            if extension in ["jpg", "png"]:
                fileList.append(os.path.join(dirPath, f))
    
    fileList = [filename]
    for i in range(0, len(fileList)):    
        filename =  fileList[i]
        print i, filename
        image = cv.imread(filename, 0)
    #     image = cv.resize(image, None, fx=2, fy=2, interpolation = cv.INTER_CUBIC)
        print image.shape
    
    
#         image = np.zeros((35,22),dtype=np.uint8)
#         array= [255, 255, 0,0,0,0,0,0,255,255,0,0,255,0,0,255,0,0,0,0,255,0,255,0,255,255]
#         print PhyloParser.getMaxLengthInLine(array)
#         PhyloParser.getFeatures(image)
    
        
        print "original image"
        PhyloParser.displayImage(image)
#         print PhyloParser.image2text(image[33:, 366:453])
    
        image_data = ImageData(image)
    
        image_data = phyloPaser.preprocces(image_data, debug = False)
        
#         image_data = phyloPaser.clusterPixels(image_data, debug = True)
            
        image_data = phyloPaser.getCorners(image_data, debug = False)   
        image_data = phyloPaser.detectLines(image_data, debug = False)
        
        image_data = phyloPaser.traceTree(image_data, debug = True)
        
        
        
#         image_data = phyloPaser.makeLinesFromCorner(image_data, debug = False)
#         image_data = phyloPaser.includeLinesFromCorners(image_data)
    
    #     image_data = phyloPaser.refineLines(image_data, debug = False)
    
    
#         image_data = phyloPaser.matchLines(image_data, debug = False)
#     
#     
#         image_data = phyloPaser.getSpecies_v2(image_data, debug = True)
#         image_data = phyloPaser.getSpecies(image_data, debug = True)
# #      
# #      
#     image_data = phyloPaser.makeTree(image_data, debug = True)
#     print "final string: ", image_data.getTreeString()
