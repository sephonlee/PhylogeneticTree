# Phylogenetic Tree Parser Tutorial

# from readCorners import readCorners
# import textDetector
# import textRemover
import cv2 as cv

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
    phyloPaser = PhyloParser()
    
    filename = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/PMC1474148_ijbsv02p0133g05.jpg'
    filename = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/phylogenetic_tree_for_parsing/PMC2324105_1471-2148-8-100-1.jpg'
    
    
    print filename
    image = cv.imread(filename,0)
#     PhyloParser.displayImage(image)

    image_data = ImageData(image)
    
    image_data.updateImage(phyloPaser.preprocces(image_data.image))

    image_data = phyloPaser.detectLines(image_data, debug = False)
    
    #Not implemented 
#     image_data = phyloPaser.detectCorners(image_data)
#     image_data = phyloPaser.refineLinesByCorners(image_data)

    
    image_data = phyloPaser.matchLines(image_data, debug = False)
    
    image_data = phyloPaser.getSpecies(image_data)
    
    
    image_data = phyloPaser.makeTree(image_data, debug = True)
    print "final string: ", image_data.getTreeString()

        