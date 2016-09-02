# Phylogenetic Tree Parser Tutorial

# from readCorners import readCorners
# import textDetector
# import textRemover
import cv2 as cv
import time

import pickle
from matplotlib import pyplot as plt
from ete3 import Tree
import operator
import random
from os import listdir
from os.path import isfile, join
from featureDetection import  *


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
    
    
    phyloPaser = PhyloParser()
    
#     filename = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/PMC1474148_ijbsv02p0133g05.jpg'
#     filename = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/PMC2324105_1471-2148-8-100-1.jpg'
#     filename = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/PMC2323573_pone.0002033.g002.jpg'



    # bg example
#     filename = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/PMC2287175_1471-2148-8-57-2.jpg'
#     filename = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/image_336.jpg'
#     filename = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/image_337.jpg'
#     filename = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/PMC2397417_1471-2164-9-215-5.jpg'
    
    # wide line example
#     filename = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/PMC2467406_1471-2148-8-193-4.jpg'
    
    # wide line example
    filename = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/PMC184354_1471-2148-3-16-2.jpg"
    
#     filename = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/PMC184354_1471-2148-3-16-7.jpg"
    
    
    #hard case
#     filename = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/PMC535349_1471-2148-4-46-1.jpg"
    
    mask = np.ones((663,600), dtype = np.uint8)
    mask[0:,385:] = 0
    
    print filename
    image = cv.imread(filename, 0)
    print image.shape
    
#     




#     image = np.ones((18,18), dtype = np.uint8)
#     image = image * 255
# #     
#     image[10:13,7:] = 0
#     image[3:,7:10] = 0
#      
#      
# #     image[0:,10:13] = 0
# #     image[8:11,0:10] = 0
#     image[0:1, 0:1] = 190
# #     image[7,11] = 0


#     image = image[15:70, 320:350]
    
#     image = image[10:50,  330:340]
    PhyloParser.displayImage(image)
    
    
#     PhyloParser.displayImage(image[280:300, 240:250])
#     print image[280:300, 240:250]
    
#     startTime = time.time()
#     image = PhyloParser.purifyBackGround(image, kernel_size=(3, 3))
#     print time.time() - startTime
    
#     PhyloParser.displayImage(image)


#     thickness1 = getThickness(verLines, 'ver')
#     thickness2 = getThickness(horLines, 'hor')    
#     thickness = int(round((thickness1+thickness2)/2))
            
    print "ori"
#     PhyloParser.displayImage(image)
#     image = image[50:125, 230:280]
    image_data = ImageData(image)

#     image_data.updateImage(phyloPaser.preprocces(image_data.image))
    
#     print "pre"
#     image_data.displayImage()
    
#     print "ori"
#     PhyloParser.displayImage(image_data.originalImage)

#     image_data = phyloPaser.getCorners(image_data, mask = mask, debug = False)    
    image_data = phyloPaser.detectLines(image_data, debug = False)

#     image_data = phyloPaser.makeLinesFromCorner(image_data, debug = False)
    
#     image_data = phyloPaser.includeLinesFromCorners(image_data)
    
    
#     image_data = phyloPaser.refineLinesByCorners(image_data)
#  
#    
    print "match line"
    image_data = phyloPaser.matchLines(image_data, debug = True)
#      
#     image_data = phyloPaser.getSpecies(image_data)
#      
#      
    image_data = phyloPaser.makeTree(image_data, debug = True)
    print "final string: ", image_data.getTreeString()
