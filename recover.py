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
    phyloPaser = PhyloParser()
    
    folderPath = 'images/'
    fileNameList = getFilesInFolder(folderPath)
    
    for index, fileName in enumerate(fileNameList):
        print index, fileName

    for index in range(24, len(fileNameList)):
        print index
        filePath = folderPath + fileNameList[index]
        if isfile(filePath) :
            image = cv.imread(filePath,0)
            
        image_data = ImageData(image)
        image_data = phyloPaser.preprocces(image_data)
        image_data.updateImageDimension()
        
        image_data = phyloPaser.detectLines(image_data, True)
        
#         image_data = phyloPaser.detectCorners(image_data)
#         image_data = phyloPaser.refineLinesByCorners(image_data)
        
        image_data = phyloPaser.matchLines(image_data)
        treeString = phyloPaser.makeTree(image_data)
        image_data = phyloPaser.getSpecies(image_data)
        
        
        
        print treeString
        
        