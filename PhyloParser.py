import cv2 as cv
import numpy as np
from Node import *
from ImageData import *
from sets import Set
import pytesseract
import time
# from ete2.phylo.phylotree import PhyloTree
import math


try:
    import Image
except:
    from PIL import Image



class PhyloParser():
    
    def __init__(self):
        return 
    
    
    @staticmethod
    def preprocces(image_data, debug = False):
        
        image = image_data.image

        if debug:
            print "Preprocessing image ..."
            print "Input image:"
            PhyloParser.displayImage(image)

        #save original image
        image_data.originalImage = image.copy() 

        #purify background
        image, image_data.varianceMask, image_data.varMap = PhyloParser.purifyBackGround(image, kernel_size = (3,3))
        if debug:
            print "Display image with removed background"
            PhyloParser.displayImage(image)
            print "Display variance mask"
            PhyloParser.displayImage(image_data.varianceMask)

        #determine effective area and save the masks into image_data        
        image_data.treeMask, image_data.nonTreeMask, image_data.contours, image_data.hierarchy = PhyloParser.findContours(image_data.varianceMask)
#         image_data.treeMask, image_data.nonTreeMask, image_data.contours = PhyloParser.findContours(image_data.textMask)
        
        
        if debug:
            print "display tree mask"
            PhyloParser.displayImage(image_data.treeMask)
            print "display non-tree mask"
            PhyloParser.displayImage(image_data.nonTreeMask)

        image = PhyloParser.bilateralFilter(image)
        if debug:
            print "bilateralFilter image"
            PhyloParser.displayImage(image)
                        
        image_data.image_preproc = image
        image_data.preprocessed = True

        return image_data
        
    ## static method for preprocessing ##
    
    @staticmethod
    def sobelFilter(image, k=5, sigma = 3):
        image  = PhyloParser.gaussianBlur(image, (k,k), sigma)
        sobelx = cv.Sobel(image, cv.CV_64F, 1, 0, ksize = k)
        sobely = cv.Sobel(image, cv.CV_64F, 0, 1, ksize = k)
        image = cv.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
        return image
    
    @staticmethod 
    def gaussianBlur(image, kernelSize = (3,3), sigma= 1.0):
        image = cv.GaussianBlur(image, kernelSize, sigma)
        return image
    @staticmethod 
    ## Old bolding, duplicate with threshold
    def binarize(image, thres=180, mode = 0):
        ret, image = cv.threshold(image, thres, 255, mode)
        return image
    
    @staticmethod
    def bilateralFilter(image, radius = 3, sigmaI = 30.0, sigmaS = 3.0):
        image = cv.bilateralFilter(image, radius, sigmaI, sigmaS)
        return image

    @staticmethod
    def downSample(image, parameter = 500.0):
        height, width = image.shape
        if height>700:
            if height > 1000:
                parameter = 700.0
            ratio  = (parameter + 0.0) / height
            image = cv.resize(image, (0,0), fx = ratio, fy=ratio)

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
    def displayImage(image):
        plt.imshow(image, cmap='Greys_r')
        plt.show()

    @staticmethod
    def sortByCntsLength(item):
        return -len(item)

    @staticmethod
    # return a mask of the tree, a mask of text and contours
    def findContours(var_mask1, var_mask2 = None):

        var_mask1 = 255 - var_mask1

        height, width = var_mask1.shape
        var_mask1 = cv.copyMakeBorder(var_mask1, 1, 1, 1, 1, cv.BORDER_CONSTANT, value = 0)
        _, contours, hierarchy= cv.findContours(var_mask1.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        
        lenghtList = []
        for cnts in contours:
            lenghtList.append(len(cnts))
            for index, points in enumerate(cnts):
                cnts[index] = points - 1 #shift back (because of padding)
           
        
        hierarchy = hierarchy[0].tolist()
        temp =  zip(contours, hierarchy)
        temp = sorted(temp, key = lambda x: -len(x[0]))
        contours = [x for x, y in temp]
        hierarchy = [y for x, y in temp]
        
        
        mask = np.zeros((height,width), dtype=np.uint8)
        cv.drawContours(mask, contours, 0, (255), thickness = -1)
        
#         PhyloParser.displayImage(var_mask1)
#         PhyloParser.displayImage(255-var_mask2)
#         PhyloParser.displayImage(mask)
        
        nonTreeMask = np.zeros((height, width), dtype = np.uint8)

        textContours = []
        for index in range(1, len(contours)):
#             print contours
#             nonTreeMask = np.zeros((height, width), dtype = np.uint8)
            # draw only contour in level 0
            if hierarchy[index][3] == -1:
                cv.drawContours(nonTreeMask, contours, index, (255), thickness = -1)
                textContours.append(contours[index])
#             print hierarchy[index][3] == -1
#             PhyloParser.displayImage(nonTreeMask)

#         PhyloParser.displayImage(nonTreeMask)
        return mask, nonTreeMask, textContours, hierarchy

    @staticmethod
    def removeLabels(image, mask):
        image[np.where(mask == 0)] = 255
        return image


    def erosion(self):
        # size = 3
        # radius = size/2
        kernel = self.erosionKernel()
        image = self.image
        image = cv.erode(image, kernel, anchor = (1,1), iterations=1)
        image = cv.dilate(image, kernel, anchor = (1,1), iterations=1)

        plt.imshow(image, cmap='Greys_r')
        plt.show()
    
    @staticmethod
    # remove color back ground
    def purifyBackGround(image, threshold_var = 0.008, threshold_var2 = 0.001, threshold_pixel = 60, threshold_hist = 10, kernel_size = (3,3), morph_kernel_size = (3,5)):

        dim = image.shape
        mask = np.zeros(dim, dtype=np.uint8)   # 1:keep 0:remove
        mask_2 = np.zeros(dim, dtype=np.uint8)
        var_map = np.zeros(dim, dtype=np.float)  ## for debugging
#         print image
         
        hist, bins = np.histogram(image.ravel(),256,[0,256])
#         print hist
#         print hist[-5:]
#         print sum(hist[-10:])
#         print sum(hist[-threshold_hist:]) / float(dim[0]*dim[1])
#         print (255-4*threshold_hist)/float(255)
#         print hist, bins
         
         
#         sort_order = hist.argsort()
#         sorted_hist = hist[sort_order[::-1]]
#         sorted_bins = bins[sort_order[::-1]]
         
#         print sorted_hist
#         print sorted_bins
         
#         plt.hist(image.ravel(),256,[0,256]) 
#         plt.show()

        hasColorBackGround = False;
        if sum(hist[-threshold_hist:]) / float(dim[0]*dim[1]) <= (255-4*threshold_hist)/float(255):
            hasColorBackGround = True;
        
        for i in range(0, dim[0] - kernel_size[0] + 1):
            for j in range(0, dim[1] - kernel_size[1] + 1):

                patch = image[i:i+kernel_size[0], j:j+kernel_size[1]].copy().astype("float")/255      
                patch_variance =  np.var(patch)
#                 patch_sum = np.sum(patch)*255
                patch_mean = np.mean(patch)*255
                
                var_map[i, j] = patch_variance ## for debugging
            
                
#                 print "%d, %d"%(i, j)
#                 print image[i:i+kernel_size[0], j:j+kernel_size[1]]
#                 print "variacne:", patch_variance
#                 print "mean:", patch_mean
#                 print "make white?", patch_variance < threshold_var and patch_mean > threshold_pixel
#                 PhyloParser.displayImage(image[i:i+kernel_size[0], j:j+kernel_size[1]])
                
                if patch_variance < threshold_var and patch_mean > threshold_pixel:                 
                    mask[i:i+kernel_size[0], j:j+kernel_size[1]] = 255
                    
                if patch_variance < threshold_var2:
                    mask_2[i:i+kernel_size[0], j:j+kernel_size[1]] = 255


        # return mask if the image has color background
        # return mask_1 if the image has no color background
        if hasColorBackGround:
            # remove background
            image[np.where(mask == 255)] = 255
        else:
            mask = mask_2
        
        # recover defect using morphology
        kernel = cv.getStructuringElement(cv.MORPH_RECT, morph_kernel_size)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        
        var_map= var_map * 255/(np.max(var_map) - np.min(var_map)) ## for debugging

        # var_map = var_map.astype('uint8')
        # varMap = np.zeros(dim, dtype = np.uint8)
        # plt.imshow(var_map, cmap='Greys_r')
        # plt.show()
        # _, contours, hierarchy= cv.findContours(var_map, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        # cv.drawContours(varMap, contours, -1, (255), thickness = -1)
        # plt.imshow(varMap, cmap='Greys_r')
        # plt.show()

        
#         PhyloParser.displayImage(var_map)
        
        
        return image, mask, var_map

    
    ## end static method for preprocessing ##
    

    @staticmethod
    def detectLines(image_data, debug = False):
        
        # use preprocessed image
        if image_data.preprocessed:      
            image = image_data.image_preproc
        else:
            image = image_data.originalImage
        
        # sub-preprocessing 
        image = PhyloParser.binarize(image, thres = 180, mode = 3)
        if debug:
            print "detecting lines ..."
            print "binerized image"
            PhyloParser.displayImage(image)
        
        # save the preprocessed image into image_data
        image_data.image_preproc_for_line_detection = image
        
        # remove text information
        if image_data.treeMask is not None:
            print "Found available tree mask! Applied the tree mask"
            image = PhyloParser.removeLabels(image, image_data.treeMask)

        image = PhyloParser.negateImage(image)
        
        # find vertical lines
        mode = 0
        height, width = image_data.image.shape
        if height>500:
            ratio = (height - 500) * 3 / 200
            minVerLine = 12 + ratio
            minHorLine = 7 + ratio
        else:
            minVerLine = 10 + height / 250
            minHorLine = 4 + height / 150
        print 'minVerLine:', minVerLine, ' minHorLine:', minHorLine
        image_data.verLines = PhyloParser.getLines(image, mode, minLength = minVerLine)

        # find horizontal lines
        image = PhyloParser.rotateImage(image)
        mode = 1
        image_data.horLines = PhyloParser.getLines(image, mode, minLength = minHorLine)

        # split horizontal lines that cut by vertical lines
        # to solve the problem of Poseidon trigeminal stick
        image_data.horLines = PhyloParser.cutLines(image_data.horLines, image_data.verLines)

        if debug:
            print "detectLines debugging"
            image_data.displayTargetLines('horLines')
            image_data.displayTargetLines('verLines')
        
        image_data.lineDetected = True # Update image_data status
        return image_data
        
        
    ## static method for detectLine ##remremoveTextoveText
    @staticmethod
    def negateImage(image, thres = 30):
        image = 255 - image
        rat, image = cv.threshold(image, thres, 255, 0)
        return image
    
    @staticmethod
    def rotateImage(image):
        return np.rot90(image)
    
    
    @staticmethod
    # the image is in the image data
    # fill up the list in image_data
    # return the image_data
    def getLines(image, mode, minLength = 10):
        
        tmp = cv.HoughLinesP(image, rho = 9, theta = np.pi, threshold = 10, minLineLength = minLength, maxLineGap = 0)
        image_height, image_width =  image.shape

        lineList = []
        if mode == 0:
            for line in tmp:
                x1, y1, x2, y2 = list(line[0])
                lineInfo = (x1, y2, x2, y1, abs(y2-y1))
                lineList.append(lineInfo)
        elif mode == 1:
            for line in tmp:
                x1, y1, x2, y2 = line[0]
                y1 = -y1 + image_height 
                y2 = -y2 + image_height
                lineInfo = (y1, x2, y2, x1, abs(y2-y1))
                lineList.append(lineInfo)
                
        return lineList
    
    @staticmethod
    def cutLines(horLines, verLines, length_threshold = 20):

        newList = []
        margin = 5
        for line in horLines:
            if line[4] < length_threshold:
                newList.append(line)
            else:
                x1, y1, x2, y2, length = line
                isnotcut = True
                for vx1, vy1, vx2, vy2, vlength in verLines:
                    if x1+margin < vx1 and vx1 < x2-margin and vy1 < y1 and y1 < vy2:
                        newline1 = [x1, y1, vx1, y2, vx1-x1]
                        newline2 = [vx1, y1, x2, y2, x2-vx1]
                        if (vx1-x1+0.0) / length > 0.3 and (vx1 - x1+0.0)/length < 0.7:
                            newList.append(tuple(newline1))
                            newList.append(tuple(newline2))
                            isnotcut = False
                            break
                if isnotcut:
                    newList.append(line)
                    
        return newList

    ## end static method for detectLine ##
    
    
    @staticmethod
    # corner data will be written in the image_data
    def getCorners(image_data, mask = None, debug = False):
        
        if image_data.preprocessed:      
            image = image_data.image_preproc
        else:
            image = image_data.originalImage
        
        if debug:
            print "Getting corners..."
            print "original image"
            PhyloParser.displayCorners(image)
         
        #sub preprocessing
        image = PhyloParser.binarize(image, thres = 180, mode = 0)
        if debug:
            print "binarized image"
            PhyloParser.displayCorners(image)

        # save the preprocessed image into image_data
        image_data.image_preproc_for_corner = image
        
        if mask is not None:
            image = PhyloParser.removeLabels(image, mask)
            if debug:
                print "use given mask"
            
        elif image_data.treeMask is not None:
            image = PhyloParser.removeLabels(image, image_data.treeMask)
            if debug:
                print "use mask generated from preprocessing"
        else:
            if debug:
                print "no mask"
                
        image_data.upCornerList = PhyloParser.detectCorners(image, 1)
        image_data.downCornerList = PhyloParser.detectCorners(image, -1)
        image_data.jointUpList = PhyloParser.detectCorners(image, 2)
        
#         image_data.jointDownList = PhyloParser.detectCorners(image, -2, mask = mask)
        
        if debug:                
            PhyloParser.displayCorners(image_data.image_preproc, [image_data.upCornerList, image_data.downCornerList, image_data.jointUpList, image_data.jointDownList])
        
        image_data.cornerDetected = True
        return image_data
    
    @staticmethod
    # NOT use
    def refinePoint_(pointList, tolerance = 5):
        
        print "refine"
        print pointList
        
        remove_index = []
        for i in range(0, len(pointList)-1):
            j = i + 1
            
            print "index=",i 
            while True and j < len(pointList):

                p = pointList[i]
                next_p = pointList[j]
                
                print i, p
                print j, next_p
                
                if abs(p[0] - next_p[0]) <= tolerance and abs(p[1] - next_p[1]) <= tolerance:
                    if next_p[0] > p[0]:
                        remove_index.append(i)
                        print "remove", i   
                    else:
                        remove_index.append(j)
                        print "remove", j

                
                j += 1
                if j < len(pointList) and abs(p[1] - pointList[j][1]) > tolerance:
                    break                
        
        remove_index = list(Set(remove_index))
        remove_index = sorted(remove_index, reverse=True)
        
        print "pointList"
        print "remove_index", remove_index
        
        for index in remove_index:
            del pointList[index]
            
        return pointList
        
    
#     @staticmethod
#     def refineCorners(upCornerList, downCornerList, jointDownList):
#         removeIndexUp = []
#         removeIndexDown = []
#         
#         for i, p in enumerate(upCornerList):
#             if (p[0]-1, p[1]) in jointDownList:
#                 removeIndexUp.append(i)
#                 
#         removeIndexUp = sorted(removeIndexUp, reverse=True)
#         for index in removeIndexUp:
#             del upCornerList[index]
#         
#         for i, p in enumerate(downCornerList):
#             if (p[0]+1, p[1]) in jointDownList:
#                 removeIndexDown.append(i)
#                 
#         removeIndexDown = sorted(removeIndexDown, reverse=True)
#         for index in removeIndexDown:
#             del downCornerList[index]
#             
#         return upCornerList, downCornerList
#     
#     @staticmethod
#     def refineJoints(jointDownList, upCornerList, downCornerList):
#         
#         removeIndexList = []
#         for i, p in enumerate(jointDownList):
#             if (p[0]+1, p[1]) in upCornerList:
#                 removeIndexList.append(i)
#                 
#         for i, p in enumerate(jointDownList):
#             if (p[0]-1, p[1]) in downCornerList:
#                 removeIndexList.append(i)
#                 
#         removeIndexList = sorted(removeIndexList, reverse=True)
#         for index in removeIndexList:
#             del jointDownList[index]
#             
#         return jointDownList
                
        
        
    @staticmethod
    def detectCorners(image, mode, kernelSize = 3):
        

        line_width = 1##
        (kernel, norPixelValue, mode) = PhyloParser.createKernel(mode, 3)
#         print kernel
        
        filteredImage = cv.filter2D(image, -1, kernel)
        
#         print "filteredImage"
#         print filteredImage
#         print "image"
#         print image
#         
        
        # First threshold to find possible corners from filtering
        mmin = np.amin(filteredImage)
        threshold= norPixelValue * pow(line_width,1.5) * 255 - 1 ####
#         print "threshold: ", threshold
        upperBound = mmin + threshold
                
        indices = np.where(filteredImage < upperBound)
            
        cornerList = zip(indices[0], indices[1])
                
        new_cornerList = []
        for corner in cornerList:
                        
            if(corner[0]-1 >= 0 and corner[0]+2 <= image.shape[0] and corner[1]-1 >=0 and corner[1]+2 <= image.shape[1]):
                patch = image[corner[0]-1:corner[0]+2, corner[1]-1:corner[1]+2].copy().astype("float")/255      
                patch_variance =  np.var(patch)
            
#                 print corner
#                 print "patch"
#                 print patch
#                 print filteredImage[corner[0]-1:corner[0]+2, corner[1]-1:corner[1]+2].copy().astype("float")
                
                patch_sum = max(np.amax(np.sum(patch, 0)), np.amax(np.sum(patch, 1)))
#                 print "sum, ", patch_sum
#                 print max(np.amax(np.sum(patch, 0)), np.amax(np.sum(patch, 1)))
#                 print "filter value,", filteredImage[corner[0], corner[1]]
#                 print "var,", patch_variance

                if abs(mode) == 1:
                    if patch_variance > 0.2 and patch_sum <= 2:
                        if mode == 1:
                            new_cornerList.append((corner[0]-1, corner[1]-1)) # shift back in line
                        else:
                            new_cornerList.append((corner[0]+1, corner[1]-1)) # shift back in line
                            
                if abs(mode) == 2:
                    if  0.17 < patch_variance < 0.23 and patch_sum <= 2:
                        if mode == 2:
                            new_cornerList.append((corner[0], corner[1]+1)) # shift back in line
                        else:
                            new_cornerList.append((corner[0], corner[1]-1)) # shift back in line
                
#         print "cornerList", len(cornerList)
#         print "new_cornerList", new_cornerList
        
#         print cornerList
        cornerList = new_cornerList
#         print cornerList
        cornerList = sorted(cornerList, key = lambda x: (int(x[1]), x[0]))
#         cornerList = PhyloParser.removeRepeatCorner(cornerList)

        return cornerList
        
    ## static method for detectCorners ##

    ## NOT USE
    @staticmethod
    def removeRepeatCorner(cornerList):
        i=0
        margin = 5
        xList = []
        yList = []
        while i<len(cornerList):
            x, y = cornerList[i]
            xList.append(x)
            yList.append(y)
            j = i
            while j+1<len(cornerList) and x + margin > cornerList[j+1][0] and x-margin < cornerList[j+1][0]:
                if y+margin > cornerList[j+1][1] and y-margin < cornerList[j+1][1]:
                    del cornerList[j+1]
                else:
                    j+=1
            i +=1
        return cornerList
    
    
    @staticmethod
    def displayCorners(image, list_pointList = []):
        
        displayImage = cv.cvtColor(image,cv.COLOR_GRAY2RGB)
        if len(list_pointList) > 0:

            rad = 2            
            colors = [(255, 0 , 0), (0, 255 , 0), (0, 0 , 255), (0, 255 , 255)]
            for i, pointList in enumerate(list_pointList):
                for y, x in pointList:
                    cv.rectangle(displayImage, (x-rad, y - rad), (x + rad, y +rad), color=colors[i], thickness=2)

        plt.imshow(displayImage)
        plt.show()
    
    
    @staticmethod
    def createKernel(mode, kernelSize):
        width = 1##
        kernel = np.zeros((kernelSize, kernelSize), np.float32)
        
        # top left corner
        if mode==1:
            for i in range(width):
                for x in range(kernelSize):
                    kernel[i][x] = 1
                    kernel[x][i] = 1
                    
        # join up kernel
        elif mode == 2:
            for i in range(width):
                for x in range(kernelSize):
                    kernel[x][kernelSize-1 - i] = 1
                    kernel[(kernelSize/2)+i][x] = 1
                    
        # join down kernel
        elif mode == -2:
            for i in range(width):
                for x in range(kernelSize):
                    kernel[x][0] = 1
                    kernel[(kernelSize/2)+i][x] = 1
            
        # bottom left corner
        elif mode == -1:
            for i in range(width):
                for x in range(kernelSize):
                    kernel[x][i] = 1
                    kernel[kernelSize-1 - i][x] = 1
                    
        summ = np.sum(kernel)
        kernel = kernel / summ
        
        if kernel[0][0] != 0:
            norPixelValue = kernel[0][0]
        else:
            norPixelValue = kernel[kernelSize-1][kernelSize-1]

        kernelPackage = (kernel, norPixelValue, mode)

        return kernelPackage
    
    ## end static method for detectCorners ##
    
    
    @staticmethod
    #return a list of dictionary
    #each element is set of point that stand in the same line
    #in each element
    #    key "corners" contains all corners in such line
    #    key "joints" contains the corner and it's corresponding joints, the first point in the list is the anchor corner
    def makeLinesFromCorner(image_data, margin = 5, debug = False):
        
        image = image_data.image_preproc_for_corner.copy()
        
        upCornerList_ver = list(image_data.upCornerList)
        upCornerList_hor = sorted(list(image_data.upCornerList),  key = lambda x: (int(x[0]), x[1]))
        upCornerIndex_ver = 0  
        upCornerIndex_hor = 0  
#         print "upCornerList", upCornerList_ver

        downCornerList_ver = list(image_data.downCornerList)
        downCornerList_hor = sorted(list(image_data.downCornerList),  key = lambda x: (int(x[0]), x[1]))
        downCornerIndex_hor = 0
#         print "downCornerList", downCornerList_ver
        
        jointUpList_ver = list(image_data.jointUpList)
        jointUpList_hor = sorted(list(image_data.jointUpList),  key = lambda x: (int(x[0]), x[1]))
#         print "jointUpList", jointUpList_ver
        
#         jointDownList = list(image_data.jointDownList)
#         print "jointDownList", jointDownList
        
        pointSet_ver = [] #vertical line between corners
        upPointSet_hor = [] #horizontal line between top left corners and corresponding joints
        downPointSet_hor = [] #horizontal line between bottom left corners and corresponding joints
        
        while upCornerIndex_ver < len(upCornerList_ver):
            upCorner = upCornerList_ver[upCornerIndex_ver]
             
            # vertical match
            cornerCandidate, downCornerList_ver = PhyloParser.matchPoints(upCorner, downCornerList_ver, image, 0, margin = margin)
            jointCandidate, jointUpList_ver = PhyloParser.matchPoints(upCorner, jointUpList_ver, image, 0, margin = margin)
                 
            ## find vertical line!
            if len(cornerCandidate) > 1:
                data = {}
                data["corners"] = cornerCandidate
                 
#                 if len(jointCandidate) > 1:
#                     del jointCandidate[0]
                     
                data["joints"] = jointCandidate
                 
                pointSet_ver.append(data)
                del upCornerList_ver[upCornerIndex_ver]
                 
            ## find on line, go next
            else:
                upCornerIndex_ver += 1
            
            
        # match horizontal line on up corner   
        while upCornerIndex_hor < len(upCornerList_hor):
            upCorner = upCornerList_hor[upCornerIndex_hor]
              
            # horizontal math
            jointCandidate, jointUpList_hor = PhyloParser.matchPoints(upCorner, jointUpList_hor, image, 1, margin = margin)
 
            ## find horizontal line!
            if len(jointCandidate) > 1:
                data = {}
                data["joints"] = jointCandidate
                 
                upPointSet_hor.append(data)
                del upCornerList_hor[upCornerIndex_hor]
#                 print "find joint candidate", jointCandidate
                 
            ## find no line, go next
            else:
                upCornerIndex_hor += 1

        
        # match horizontal line on down corner
        # keep using the same jointUpList_hor
        while downCornerIndex_hor < len(downCornerList_hor):
            downCorner = downCornerList_hor[downCornerIndex_hor]
              
            # horizontal math
            jointCandidate, jointUpList_hor = PhyloParser.matchPoints(downCorner, jointUpList_hor, image, 1, margin = margin)
 
            ## find horizontal line!
            if len(jointCandidate) > 1:
                data = {}
                data["joints"] = jointCandidate
                 
                downPointSet_hor.append(data)
                del downCornerList_hor[downCornerIndex_hor]
#                 print "find joint candidate", jointCandidate
                 
            ## find no line, go next
            else:
                downCornerIndex_hor += 1

        pointSet_ver = PhyloParser.removeDuplicatePoint(pointSet_ver, 0)
        upPointSet_hor = PhyloParser.removeDuplicatePoint(upPointSet_hor, 0)
        downPointSet_hor = PhyloParser.removeDuplicatePoint(downPointSet_hor, 0)

        if debug:
            ver_lines = PhyloParser.pointSetToLine(pointSet_ver, type="corners")
            hor_lines_up =  PhyloParser.pointSetToLine(upPointSet_hor, type="joints")
            hor_lines_down =  PhyloParser.pointSetToLine(downPointSet_hor, type="joints")
            
            PhyloParser.displayCornersAndLine(image, [upCornerList_hor, jointUpList_hor], [ver_lines, hor_lines_up, hor_lines_down])
        
                
        #         print "remain upCornerList horizontal", upCornerList_hor
        
        image_data.pointSet_ver = pointSet_ver
        image_data.upPointSet_hor = upPointSet_hor
        image_data.downPointSet_hor = downPointSet_hor
        
        image_data.lineDetectedFromCorners = True
        
        return image_data

    ## static method for makeLinesFromCorner ##

    @staticmethod
    # axis = 0 --> vertically match
    # axis = 1 --> horizontally match 
    def matchPoints(point, candidatePoints, image, axis, margin = 5):
        
        if axis == 0 or axis == 1 :
            
            index_for_margin_test = 1 - axis
            index_for_location_test = axis
            
            matchPoints = [point] ### 
            candidatePointsIndex = 0

            while True and candidatePointsIndex < len(candidatePoints):
                downCorner = candidatePoints[candidatePointsIndex]
#                 print "this downCornerIndex: ", candidatePointsIndex, downCorner
                
                if  (abs(downCorner[1-axis] - point[1-axis]) <= margin) and  (downCorner[axis] - point[axis] > 0) and PhyloParser.isInLine(point, downCorner, image):
                    # find match,  stay in the same index due to removal"
                    matchPoints.append(downCorner)
                    del candidatePoints[candidatePointsIndex]
                
                elif downCorner[1-axis] - point[1-axis] <= margin or downCorner[axis] - point[axis] > 0 or PhyloParser.isInLine(point, downCorner, image):
                    # not match, but close, keep searching next element
                    candidatePointsIndex += 1
                    
                else: 
                    # once margin test fail, the later elements will all fail, so stop iterating
                    break
            
            return matchPoints, candidatePoints
        
        else:
            print "axis must ether 1 or 0"
            return None, candidatePoints

    @staticmethod
    # determine if two points are in the same line
    def isInLine(corner1, corner2, image, threshold = 0.01):
        
        y_min = min(corner1[0], corner2[0])
        y_max = max(corner1[0], corner2[0])      
        x_min = min(corner1[1], corner2[1])
        x_max = max(corner1[1], corner2[1])
        
        subimage = image[y_min:y_max+1, x_min:x_max+1].copy().astype("float")/255  ## not count the later index
        variance =  np.var(subimage)
        
        return variance < threshold
    
    @staticmethod
    # remove duplicate point in each point set
    def removeDuplicatePoint(pointSet, axis, margin = 5):
        for s in pointSet:
            if "corners" in s and len(s["corners"]) > 2:
                s["corners"] = PhyloParser.refinePoint(s["corners"], margin)
            if "joints" in s and len(s["joints"]) > 2:
                s["joints"] = PhyloParser.refinePoint(s["joints"], margin)
                
        return pointSet
                
                
    @staticmethod
    # need sorted
    # called by removeDuplicatePoint
    # select the very bottom or very right point in the margin as the representative point 
    def refinePoint(pointList, margin = 5):
         
        remove_index = []
        for i in range(0, len(pointList)-1):
            j = i + 1

            p = pointList[i]
            next_p = pointList[j]

            if abs(p[0] - next_p[0]) <= margin and abs(p[1] - next_p[1]) <= margin:
                remove_index.append(i)
              
        remove_index = list(Set(remove_index))
        remove_index = sorted(remove_index, reverse=True)
         
        for index in remove_index:
            del pointList[index]
             
        return pointList
    
    @staticmethod
    # need sorted already
    def pointSetToLine(pointSetList, type = "corners"):
        lineList = []
        for pointSet in pointSetList:
            
            points = pointSet[type] # select the type of point set
            
            # must have at least two points to form a line
            if len(points) > 1:
                y1 = points[0][0]
                x1 = points[0][1]
                
                y2 =  points[-1][0]
                x2 =  points[-1][1]
            
                lineLength = max(abs(x2-x1),abs(y2-y1))
                lineList.append((x1, y1, x2, y2, lineLength))
        
        return lineList
    
    @staticmethod
    def displayCornersAndLine(image, list_pointList = [], list_lines = []):
        
        displayImage = cv.cvtColor(image,cv.COLOR_GRAY2RGB)
        if len(list_pointList) > 0:
            rad = 2            
            colors = [(255, 0 , 0), (0, 255 , 0), (0, 0 , 255), (0, 255 , 255)]
            for i, pointList in enumerate(list_pointList):
                for y, x in pointList:
                    cv.rectangle(displayImage, (x-rad, y - rad), (x + rad, y +rad), color=colors[i], thickness=2)


        if len(list_lines) > 0:
            colors = [(255, 150 , 0), (150, 255 , 0), (150, 0 , 255)]
            for i, lines in enumerate(list_lines):
                for line in lines:
                    x1, y1, x2, y2, length = line
                    cv.rectangle(displayImage, (x1, y1), (x2, y2), color=colors[i], thickness=2)
            
        plt.imshow(displayImage)
        plt.show()

    ## end static method for makeLinesFromCorner ##
    

    @staticmethod
    # remove duplicate lines
    # for vertical lines: select the very right line 
    # for horizontal lines: select the very bottom line
    def refineLines(image_data, debug = False):
        
        image = image_data.image_preproc_for_corner
                
        if debug:
            PhyloParser.displayLines(image, image_data.horLines)
        
        midPointOfHorLines = PhyloParser.getMidPoints(image_data.horLines)
        midPointOfVerLines = PhyloParser.getMidPoints(image_data.verLines)
        
        
        midPointOfHorLines = sorted(midPointOfHorLines, key = lambda x: (x[0], x[1]))
        midPointOfVerLines = sorted(midPointOfVerLines, key = lambda x: (x[1], x[0]), reverse=True) #take the very right lines
        

        image_data.horLines, image_data.horLineGroup = PhyloParser.getRepresentativeLines(image, midPointOfHorLines)
                    
        if debug:
            PhyloParser.displayLines(image, image_data.horLines)
            
        image_data.verLines, image_data.verLineGroup = PhyloParser.getRepresentativeLines(image, midPointOfVerLines)
                    
        if debug:
            PhyloParser.displayLines(image, image_data.verLines)
            
        image_data.lineRefined = True
        return image_data

    ## static method for refineLines ##

    @staticmethod
    def getMidPoints(lines):
        midPoints = []
        for l in lines:
            midPoints.append((int((l[0] + l[2])/2), int((l[1] + l[3])/2), l[4], l))
        return midPoints
        
    
    @staticmethod
    #axis = 0: vertical lines
    #axis = 1: horizontal lines
    def getRepresentativeLines(image, midPointOfLines, margin = 5):
                
        keep_lines = []
        group_line_indices = []
        group_lines = []
           
        index_head = 0
        line1 = midPointOfLines[index_head]
        max_length = line1[2]
        keep_index = 0
        index_head += 1
        temp_line_indices = [0]
        
        while index_head < len(midPointOfLines):
            
            line2 = midPointOfLines[index_head]
            
            x_in_margin = abs(line1[0] - line2[0]) <= margin
            y_in_margin = abs(line1[1] - line2[1]) <= margin
            
            print "line1", line1
            print "line2", line2, PhyloParser.checkLine(image, line2[3])
            
            if x_in_margin and y_in_margin:
                print "find a continuous line"
                #find a continuous line
                temp_line_indices.append(index_head)            
                
                # check if line2 is better to keep as the main line of the set
                if line2[2] >= max_length and PhyloParser.checkLine(image, line2[3]):
                    print "put line2 index into keep index"
                    max_length = line2[2]
                    keep_index = index_head
                    
                # moving forward
                line1 = line2
                index_head += 1
            
            elif (not x_in_margin and not y_in_margin) or index_head == len(midPointOfLines) - 1:
                print "break"
                #save set
                group_line_indices.append(temp_line_indices)####
                group_lines.append([midPointOfLines[x] for x in temp_line_indices])
                keep_lines.append(midPointOfLines[keep_index][3])
                
                print "this set:", group_lines[-1]
                print "keep lines:", len(keep_lines), keep_lines
                
                #remove found index from the lines
                temp_line_indices = sorted(temp_line_indices, reverse=True)
                for i in temp_line_indices:
                    del midPointOfLines[i]
                    
                print "remain lines:", midPointOfLines
                #start over
                index_head = 0 
                line1 = midPointOfLines[index_head]
                max_length = line1[2]
                keep_index = 0
                index_head += 1
                temp_line_indices = [0]
                
            else:
                #keep moving foward
                print "not matched, keep searching next"
                index_head += 1
            
        
        #pick up last sest
        if midPointOfLines > 0:
            group_lines.append(midPointOfLines)
            keep_lines.append(midPointOfLines[keep_index][3])
            
            
        print "group_lines", len(group_lines), group_lines
        print "keep lines:", len(keep_lines), keep_lines
        
        return keep_lines, group_lines

    @staticmethod
    # determine if the given line is a truly BLACK line
    def checkLine(image, line, var_threshold = 0.01, mean_threshold = 3):

        if line[0] == line[2]:
            array = image[line[1]:line[3], line[0]:line[0]+1]
        else:
            array = image[line[1]:line[1]+1, line[0]:line[2]]
            
            
        variance = np.var(array.astype("float")/255)
        mean = np.mean(array)

        return variance < var_threshold and mean < mean_threshold
    
    @staticmethod
    #for debug
    def displayLines(image, lines):
                
        if len(image.shape) == 2:
            whatever = cv.cvtColor(image, cv.COLOR_GRAY2RGB)

        for line in lines:
            x1, y1, x2, y2, length = line
            cv.rectangle(whatever, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
        plt.imshow(whatever)
        plt.show()
                
    ## end static method for refineLines ##
    
    
    @staticmethod
    # merge the lines created from corners with lines created from line detection
    def includeLinesFromCorners(image_data):
        if image_data.lineDetectedFromCorners and image_data.lineDetected:
            ver_lines = PhyloParser.pointSetToLine(image_data.pointSet_ver, type="corners")
            hor_lines_up =  PhyloParser.pointSetToLine(image_data.upPointSet_hor, type="joints")
            hor_lines_down =  PhyloParser.pointSetToLine(image_data.downPointSet_hor, type="joints")
            
            image_data.horLines += hor_lines_up
            image_data.horLines += hor_lines_down
            image_data.verLines += ver_lines
        
        else:
            print "Found no lines created from corner detection."
            
        return image_data
    
    ## static method for includeLinesFromCorners ##
    
    ## end static method for includeLinesFromCorners ##
    
    
    
    
    def matchLines(self, image_data, debug = False):
                
        if image_data.lineDetected:
            image_data = self.matchParent(image_data)
            image_data = self.matchChildren(image_data)
            # image_data = self.removeText(image_data)
            
            if debug:
                image_data.displayTargetLines('parent')
                image_data.displayTargetLines('children')
                image_data.displayTargetLines('interLines')
                image_data.displayTargetLines('anchorLines')
            
            image_data.lineMatched = True
            
        else:
            print "Error! Please do detectLines before this method"
        
        
        return image_data

    
    @staticmethod
    # your display() method
    def displayTree(root):
        return
    
    
    def matchParent(self, image_data):
        horLines = sorted(image_data.horLines, key = image_data.sortByXEndFromLeft)
        verLines = sorted(image_data.verLines, key = image_data.sortByXEndFromLeft)
        margin = 5
        parent = []
        interLines = []
        jointPoints = []

        for line in horLines:
            x1, y1, x2, y2, length = line
            probParent = []
            for vline in verLines:
                vx1, vy1, vx2, vy2, vlength= vline
                if y1 > vy1 and y1 < vy2 and x2 > vx1 - margin and x2 < vx2 + margin:                   
                    probParent.append(vline)
            # print probParent
            minDist = None
            target = None
            if len(probParent) >0:

                for vline in probParent:
                    vx1, vy1, vx2, vy2, vlength= vline
                    dist = abs(vx2-x2)
                    if dist == 0:
                        score = vlength
                    else:
                        score = vlength * (1 - int(dist/margin)*0.1)
                    if not target or score > minDist:
                        target = vline
                        minDist = score
                tupleLine = tuple(line)
                interLines.append(tupleLine)
                parent.append(((tupleLine, target), minDist))
                jointPoints.append((x2-1,y2))
        parent = sorted(parent, key = image_data.sortByDist)

        parent = self.removeRepeatLines(parent)
        interLines = self.removeRepeatLinesBasic(interLines)

        image_data.parent = parent
        image_data.interLines = interLines



        return image_data
    
    
    def matchChildren(self, image_data):
        verLines = sorted(image_data.verLines, key = image_data.sortByXEndFromLeft)
        horLines = sorted(image_data.horLines, key = image_data.sortByXHead)
        margin = 5
        children= []
        anchorLines = []
        for line in verLines:
            x1, y1, x2, y2, length = line
            upperLeave = []
            lowerLeave = []
            interLeave = []
            for hline in horLines:
                hx1, hy1, hx2, hy2, hlength = hline
                isUpperLeave = False
                isLowerLeave = False
                if x1 > hx1 - margin and x1 < hx1 + margin and y1 > hy1 - margin and y1 < hy1 + margin:
                    upperLeave.append(hline)
                    isUpperLeave = True
                if x1 > hx1 - margin and x1 < hx1 + margin and y2 > hy1 - margin and y2 < hy1 + margin:
                    lowerLeave.append(hline)
                    isLowerLeave = True
                if x1 -margin < hx1  and x1 + margin > hx1  and y1 + margin < hy1 and y2 - margin > hy1:
                    if not (isUpperLeave or isLowerLeave):
                        interLeave.append(hline)
            if len(upperLeave) > 0 or len(lowerLeave) > 0 or len(interLeave) > 0:
                totalDist = 0
                minDist = None
                upTarget = None
                for upLine in upperLeave:
                    upx1, upy1, upx2, upy2, uplength = upLine
                    dist = abs(upx1-x1) + abs(upy1 - y1)
                    if dist == 0:
                        score = uplength
                    else:
                        score = uplength * (1 - int(dist/margin)*0.1)
                    if not minDist or score > minDist:
                        minDist = score
                        upTarget = upLine
                if minDist:
                    totalDist +=minDist
                minDist = None
                downTarget = None
                for downLine in lowerLeave:
                    downx1, downy1, downx2, downy2, downlength = downLine
                    dist = abs(y2-downy1) + abs(downx1 - x1)
                    if dist == 0:
                        score = downlength
                    else:
                        score = downlength * (1 - int(dist/margin)*0.1)
                    if not minDist or score > minDist:
                        minDist = score
                        downTarget = downLine
                if minDist:
                    totalDist +=minDist 

                if len(interLeave) > 0:
                    image_data.isBinary = False
                    interLeave = self.removeRepeatLinesBasic(interLeave)
                    for subline in interLeave:
                        subline = tuple(subline)
                        isAnchor = True
                        for interline in image_data.interLines:
                            if self.isSameLine(subline, interline):
                                isAnchor = False
                                break
                        if isAnchor:
                            anchorLines.append(subline)
                if upTarget:
                    upTarget = tuple(upTarget)
                    isAnchor = True
                    for interline in image_data.interLines:
                        if self.isSameLine(upTarget, interline):
                            isAnchor = False
                            break
                    if isAnchor:
                        anchorLines.append(upTarget)
                if downTarget:
                    downTarget = tuple(downTarget)
                    isAnchor = True
                    for interline in image_data.interLines:
                        if self.isSameLine(downTarget, interline):
                            isAnchor = False
                            break
                    if isAnchor:
                        anchorLines.append(downTarget)
                if not (upTarget or downTarget):
                    if image_data.isBinary:
                        children.append(((line, (upTarget, downTarget)),totalDist - 100))
                    else:
                        tmpLineList = [upTarget, downTarget]
                        for subline in interLeave:
                            tmpLineList.append(subline)
                        children.append(((line, tuple(tmpLineList)), totalDist-100))
                else:
                    if image_data.isBinary:
                        children.append(((line, (upTarget, downTarget)), totalDist))
                    else:
                        tmpLineList = [upTarget, downTarget]
                        for subline in interLeave:
                            tmpLineList.append(subline)
                        children.append(((line, tuple(tmpLineList)), totalDist))
        children = sorted(children, key = image_data.sortByDist)
        children = self.removeRepeatLines(children)
        anchorLines = self.removeRepeatLinesBasic(anchorLines)
        image_data.children = children
        image_data.anchorLines = anchorLines



        return image_data


    @staticmethod     
    def removeText(image_data):
        return image_data
    

    def removeRepeatLinesBasic(self, lineList):
        margin = 5
        i=0 
        while i<len(lineList):
            x1, y1, x2, y2, length= lineList[i]

            for j in xrange(len(lineList)-1, i, -1):
                if x1 - margin < lineList[j][0] and x2 + margin > lineList[j][2]:
                    if y2+margin > lineList[j][3] and y1-margin < lineList[j][1]:
                        del lineList[j]
            i +=1
        return lineList

    def removeRepeatLines(self, lineList):
        margin = 5
        i=0
        while i<len(lineList):
            lines, dist = lineList[i]
            x1, y1, x2, y2, length =lines[0]

            for j in xrange(len(lineList)-1, i, -1):

                if x1 - margin < lineList[j][0][0][0] and x2 + margin > lineList[j][0][0][2]:
                    if y2+margin > lineList[j][0][0][3] and y1-margin < lineList[j][0][0][1]:
                        if lineList[j][0][0][4] <= length+ (margin-2):
                            del lineList[j]

            i +=1

        return lineList


    ## end static method for matchLines ##


    # 1. check contours
    # 2. check the right 5 pixel in the mask
    # 1T2F: combines contours and set the very left as the line right
    # 1T2T: cut, recontour
    #    if invovle h>w contour, drop, it not, include the new contour
    # 1F2T: cut, recontour
    # 1F2F: drop 
    @staticmethod
    def getSpecies(image_data, debug = False):

        image = image_data.image
        nonTreeMask = image_data.nonTreeMask
        treeMask = image_data.treeMask
        anchorLines = image_data.anchorLines
        contours = image_data.contours       
        varianceMask = image_data.varianceMask
        hierarchy = image_data.hierarchy
        textMask = image_data.textMask
        
        # cut out anchor line
        dim = varianceMask.shape
        cut_height = 5
        cut_width = 2
        
        # cut out anchorlines from text
        for line in anchorLines:        
            varianceMask[max(0, line[1] - cut_height) : min(dim[0], line[1] + cut_height), max((line[2] - cut_width), 0) : min((line[2] + cut_width), dim[1])] = 255

        # find contours        
        treeMask, nonTreeMask, contours, hierarchy = PhyloParser.findContours(varianceMask)

        # transform contours to bonding boxes
        contourBoxes = []
        for cnt in contours:
            contourBoxes.append(PhyloParser.getContourInfo(cnt))
                        
        # sort boxes and anchorlines from top to bot for further matching       
        contourBoxes = sorted(contourBoxes, key = lambda x: (x[0], x[2])) #top, left
        anchorLines = sorted(anchorLines, key = lambda x: (x[3], x[2])) #top, right        
        
        # DEBUG
        if debug:
            print "show variance mask with anchor lines"
            img = varianceMask.copy()
            for b in contourBoxes:
                cv.rectangle(img,(b[2],b[0]),(b[3],b[1]),(0,125,0),2)
            PhyloParser.displayLines(img, anchorLines)
        
        # group entangled boxes together
        height_group, contour_group, textBoxes = PhyloParser.clusterBoxes(contourBoxes)
        

    
        textGroup = []    
        for i in range(0, len(contour_group)):
            contours = contour_group[i]
            heights = height_group[i]
            group = zip(heights, contours)
            group = sorted(group, key = lambda x: (x[0]))
            
            # split boxes in the group vertically
            boundaries, boxes = PhyloParser.splitBoxGroup(group)
            
            # modified boundaries of boxes to attach each other
            boundaries, boxes = PhyloParser.stitchBoundries(boundaries, boxes)


#             print "show boundaries"
#             img = image.copy()
#             for b in boundaries:
#                 cv.rectangle(img,(b[2],b[0]),(b[3],b[1]),(0,125,0),2)
#             PhyloParser.displayImage(img)
#             
#             for box in boxes:
#                 for b in box:
#                     cv.rectangle(img,(b[2],b[0]),(b[3],b[1]),(0,125,0),2)
#             PhyloParser.displayImage(img)

            
            # split boxes in the group horizontally
            new_boundaries, boxes = PhyloParser.splitBoundaries(boundaries, boxes)

            for i in range(0, len(boundaries)):
                textGroup.append({"text_row":boundaries[i], "boxes":new_boundaries[i]})
                #DUBUG
#                 img = image.copy()
#                 for b in new_boundaries[i]:
#                     cv.rectangle(img,(b[2],b[0]),(b[3],b[1]),(0,125,0),2)
#                 PhyloParser.displayImage(img)
                
                
                
            
        if debug:
            print "show bonding boxes"
            img = image.copy()
            for dict in textGroup:
#                 b = dict["text_row"]
                for b in dict["boxes"]:
                    cv.rectangle(img,(b[2],b[0]),(b[3],b[1]),(0,125,0),2)
            PhyloParser.displayImage(img)
            
        ### match anchorlines and textboxes
        lineIndex2BoxIndex, boxIndex2LineIndex, noTextLineIndex, robbedLineIndex = PhyloParser.matchAnchorLineAndTextBox(anchorLines, textGroup)
        
        if debug:
            ###debug
            print "show matched line and textbox"
            img = image.copy()
            lines = []
            for key in boxIndex2LineIndex:
                index = key[0]
                box_index = key[1]
                line_indices = boxIndex2LineIndex[key]            
                b = textGroup[index]["boxes"][box_index]
                cv.rectangle(img,(b[2],b[0]),(b[3],b[1]),(0,125,0),2)
                for line_index in line_indices:
                    lines.append(anchorLines[line_index])
                    
            PhyloParser.displayLines(img, lines)

            print "show noTextLines"
            img = image.copy()
            lines= []
            for line_index in noTextLineIndex:
                lines.append(anchorLines[line_index])
            PhyloParser.displayLines(img, lines)
             
            print "show robbedLines"
            img = image.copy()
            lines= []
            for line_index in robbedLineIndex:
                lines.append(anchorLines[line_index])
            PhyloParser.displayLines(img, lines)
            
            PhyloParser.extractText(image, anchorLines, textGroup, lineIndex2BoxIndex, boxIndex2LineIndex, noTextLineIndex, robbedLineIndex)
        
        
        
    @staticmethod
    def extractText(image, anchorLines, textGroup, lineIndex2BoxIndex, boxIndex2LineIndex, noTextLineIndex, robbedLineIndex, padding = 2):
        
        dim = image.shape
        line2species = {}
        
        for line_index in lineIndex2BoxIndex:
            print "line_index", line_index, "line", anchorLines[line_index]
            boxIncides = lineIndex2BoxIndex[line_index]
            print boxIncides
            
            result = []
            for element in boxIncides:
                index = element[0]
                box_index = element[1]
                box = textGroup[index]["boxes"][box_index]
                print "box", box
            
                aspect_ratio = float((box[1]-box[0])) / (box[3]-box[2]+0.00001)
                
            
                subImage = image[max(0,box[0]-padding):min(dim[0],box[1]+padding), max(0,box[2]-padding):min(dim[1],box[3]+padding)]
                
                #increase size to increase accuracy of ocr
                subImage = cv.resize(subImage, None, fx=2, fy=2, interpolation = cv.INTER_CUBIC)
                subImage = cv.copyMakeBorder(subImage, padding, padding, padding, padding, cv.BORDER_CONSTANT, value = 255)
                
                if aspect_ratio <= 1:
                    cv.imwrite("tmp.tiff", subImage)
                    text = pytesseract.image_to_string(Image.open('tmp.tiff'))
                    if text != "":
                        result.append(text)
                    print text
                    PhyloParser.displayImage(subImage)
                else:
                    subImage = PhyloParser.rotateImage(subImage)
                    cv.imwrite("tmp.tiff", subImage)
                    text = pytesseract.image_to_string(Image.open('tmp.tiff'))
#                     if text != "":
#                         result.append(text)
                    print text
                    PhyloParser.displayImage(subImage)
                        
                    
            if len(result) > 0:
                text = " ".join(result)    
            else:
                text = ""
                
            line2species[anchorLines[line_index]] = {"text": text, "status": "from_box"}
            
        for line_index in noTextLineIndex:
            line2species[anchorLines[line_index]] = {"text": "", "status": "no_box"}
           
        for line_index in robbedLineIndex:
            line2species[anchorLines[line_index]] = {"text": "", "status": "robbed"}       
            
        print "final result"
        for key in line2species:
            print key, line2species[key]
                
        return
    
#     labelBox = image[labelSpot[0]-margin:labelSpot[1]+margin, labelSpot[2]-margin:labelSpot[3]+margin]
#             
#             cv.imwrite("tmp.tiff", labelBox)
#             label = pytesseract.image_to_string(Image.open('tmp.tiff'))
#             anchorLabelList[line] = label

        
        
    
    @staticmethod
    def matchAnchorLineAndTextBox(anchorLines, textGroup, horizontal_anchor_margin = 5, verticle_anchor_margin = 2):

        index = 0        
        boxIndex2LineIndex = {}
        lineIndex2BoxIndex={}
        noTextLineIndex = []
        
        for line_index, line in enumerate(anchorLines):
            y = line[1]
            right = line[2]
            target_box_index = []
            target_boxes = []
            while index < len(textGroup):
                textBox = textGroup[index]
                text_row = textBox["text_row"]
    
                if  text_row[0] <= y - verticle_anchor_margin <= text_row[1] \
                    or text_row[0] <= y <= text_row[1] \
                    or text_row[0] <= y + verticle_anchor_margin <= text_row[1]:
                    
                    start_index = index
                    for i, box in enumerate(textBox["boxes"]):
                        
                        # find corresponding box
                        # (give a small margin to tolerate the overlapping between line and box)
                        if right - horizontal_anchor_margin <= box[2]:

                            target_boxes.append(box)
                            target_box_index.append((index, i))
                            
                            #push into boxIndex2LineIndex
                            if (index, i) in boxIndex2LineIndex:
                                boxIndex2LineIndex[(index, i)].append(line_index)
                            else:
                                boxIndex2LineIndex[(index, i)] = [line_index]
                            
                    index += 1
                elif text_row[0] < y + verticle_anchor_margin:
                    index += 1
                else:
                    index = start_index ## go back to the start_index, since the multiple lines can match with the same box
                    break

            if len(target_box_index) > 0:
                lineIndex2BoxIndex[line_index] = target_box_index
            else:
                noTextLineIndex.append(line_index)
        
     
        ## select the best line among multiple matched lines
        robbedLineIndex = Set()
        for key in boxIndex2LineIndex:
            index = key[0]
            box_index = key[1]
            line_indices = boxIndex2LineIndex[key]
            
            # multiple lines take the same text box
            if len(line_indices) > 1:

                closest_distance = 999999999
#                 selected_index = 0
                for line_index in line_indices:
                    line = anchorLines[line_index]
                    box = textGroup[index]["boxes"][box_index]
                    
                    distance = box[2] - (line[2] - horizontal_anchor_margin)    
                    if distance < closest_distance:
                        closest_distance = distance
                        selected_line_index = line_index

                boxIndex2LineIndex[key] = [selected_line_index]
                
                for line_index in line_indices:
                    if line_index != selected_line_index:
                        if line_index in lineIndex2BoxIndex:
                            del lineIndex2BoxIndex[line_index]
                            robbedLineIndex.add(line_index)
        
        return lineIndex2BoxIndex, boxIndex2LineIndex, noTextLineIndex, robbedLineIndex
            
        
            
    @staticmethod
    def clusterBoxes(contourBoxes): 
        contour_group = [] # a list of groups of contour boxes 
        height_group = [] # a list of groups of box height corresponding to contour_group

        textBoxes = [] # a list of outer bonding box corresponding to contour_group
            
        ## group contour boxes
        box = contourBoxes[0]
        box_height = box[1] - box[0]
        top = box[0]
        bot = box[1]
        left = box[2]
        right = box[3]
        tmp = [box] # temporary list of box
        tmp_h = [box_height] # temporary list of box height
        
        index = 1
        while index < len(contourBoxes):
            box =  contourBoxes[index]
            # current overlaps the previous box or entirely locates in the previous box
            if box[0] <= bot:
                tmp.append(box)
                tmp_h.append(box[1] - box[0])
                top = min(top, box[0])
                bot = max(bot, box[1])
                left = min(left, box[2])
                right = max(right, box[3])
                
                index += 1
            else:
                contour_group.append(tmp)
                height_group.append(tmp_h)
                textBoxes.append([top, bot, left, right])
                box_height = box[1] - box[0]
                top = box[0]
                bot = box[1]
                left = box[2]
                right = box[3]
                tmp = [box]
                tmp_h = [box_height]
                index += 1
                
            # append the last group
            if index == len(contourBoxes):
                contour_group.append(tmp)
                height_group.append(tmp_h)
                textBoxes.append([top, bot, left, right]) 
                     
        return height_group, contour_group, textBoxes
    
    @staticmethod
    # split boxes in the group vertically
    def splitBoxGroup(boxGroup, text_height_threshold = 6, text_length_threshold = 5):
        
        #data structure: sub_group = [boundary, [original boxes]]
        
        if len(boxGroup) == 1:
            if boxGroup[0][1][1] - boxGroup[0][1][0] < text_height_threshold:
                # only one element that does not pass the threshold in this group
                sub_group = []
            else:
                # only on element in this group and it passes the threshold in this group
                box = boxGroup[0][1]
                sub_group = [{"boundary": box, "boxes" : [box]}]
                
        else:
            index = 0
            while index < len(boxGroup) and boxGroup[index][1][1] - boxGroup[index][1][0] < text_height_threshold:
                index += 1
        
            # all boxes can't pass threshold, merge them all
            if index == len(boxGroup):
                
                top = min(boxGroup[0][1][0], boxGroup[-1][1][0])
                bot = max(boxGroup[0][1][1], boxGroup[-1][1][1])
                left = min(boxGroup[0][1][2], boxGroup[-1][1][2])
                right = max(boxGroup[0][1][3], boxGroup[-1][1][3])
                return [[top, bot, left, right]], [[x[1] for x in boxGroup]]###
            ###########################################

            # not all boxes can't pass threshold
            # drop disqualified boxes and start from the first qualified box
            box = boxGroup[index][1]
            sub_group = [{"boundary": box, "boxes" : [box]}] # push the first boxes
            
            for i in range(index+1, len(boxGroup)):

                box = boxGroup[i][1]
                to_sg_index = [] #target group to merge
                count = 0
                for j, dict in enumerate(sub_group):

                    boundary = dict["boundary"]
#                     print "scan sub_group j=",j, boundary, PhyloParser.isOverLap(boundary, box)
                    if PhyloParser.isOverLap(boundary, box):
                        to_sg_index.append(j)
                        
#                     print "to_sg_index", to_sg_index

                # belong to only one sub_group
                if len(to_sg_index) == 1:
                    #update sub_group
                    index = to_sg_index[0]
                    boundary = sub_group[index]["boundary"]
                    top = min(boundary[0], box[0])
                    bot = max(boundary[1], box[1])
                    left = min(boundary[2], box[2])
                    right = max(boundary[3], box[3])
                    sub_group[index]["boundary"] = [top, bot, left, right] # update boundaries
                    sub_group[index]["boxes"].append(box) # add the box into the box set
                    
                # belong to no sub_group
                # add a new sub_group
                elif len(to_sg_index) == 0:
                    sub_group.append({"boundary": box, "boxes" : [box]})
#                     print "add new elemenet in sub group"
                    # re-sort from top to bot
                    sub_group = sorted(sub_group, key = lambda x: (x["boundary"][0])) # sorted by boundary's top
                    
                # overlap move then 2 sub groups, split the current box
                else:
                    for j, index in enumerate(to_sg_index):
                        # top one
                        dict = sub_group[index]
                        boundary = dict["boundary"] # the boundary of assigned box
                        if index == 0:
                            top = min(boundary[0], box[0])
                            box_top = top
                        else:
                            top = boundary[0]
                            ## preserve mutli-row box's top
                            ## which will align with the new boundary created in stitchBoundaries
                            previous_boundary = sub_group[index-1]["boundary"]
                            box_top = (boundary[0]+previous_boundary[1])/2
                            
                        # bot one
                        if index == len(sub_group) - 1:
                            bot = max(boundary[1], box[1])
                            box_bot = bot 
                        else:
                            bot = boundary[1]
                            ## preserve mutli-row box's bot
                            ## which will align with the new boundary created in stitchBoundaries
                            next_boundary = sub_group[index+1]["boundary"]
                            box_bot = (boundary[1]+next_boundary[0])/2 
                            
                        left = min(boundary[2], box[2])
                        right = max(boundary[3], box[3])
                    
                        sub_group[index]["boundary"] = [top, bot, left, right] #update boundary
                        sub_group[index]["boxes"].append([box_top, box_bot, box[2], box[3]]) #push splitted box, use new top and bot
        
        boundries = [x["boundary"] for x in sub_group]
        boxes = [x["boxes"] for x in sub_group]
        return boundries, boxes
  
    
    @staticmethod
    # modified boxes to attached each other 
    def stitchBoundries(boundaries, boxes, text_height_threshold = 5):
     
        current_index = 0
        while current_index < len(boundaries) - 1:

            next_index = current_index + 1

            while next_index < len(boundaries) - 1 and (boundaries[next_index][1] - boundaries[next_index][0]) < text_height_threshold:
                next_index += 1
                
            # current box does not pass threshold
            # jump to the first qualified box 
            # update the qualified box
            if (boundaries[current_index][1] - boundaries[current_index][0]) < text_height_threshold:
                boundaries[next_index][0] = boundaries[current_index][0]
                # remove previous disqualified elements
                for i in range(next_index-1, current_index-1, -1):
                    del boundaries[i]
                    del boxes[i]
            
            # next box is not qualified
            elif (boundaries[next_index][1] - boundaries[next_index][0]) < text_height_threshold:
                boundaries[current_index][1] = boundaries[next_index][1]
                for i in range(next_index, current_index, -1):
                    del boundaries[i]
                    del boxes[i]
                    
                
            # next box is a qualified
            else:
                # update current box bot
                boundary = (boundaries[current_index][1] + boundaries[next_index][0])/2
                boundaries[current_index][1] = boundary
                # update next box top
                boundaries[next_index][0] = boundary
                
                # remove previous disqualified elements
                for i in range(next_index-1, current_index, -1):
                    del boundaries[i]
                    del boxes[i]
                                
                current_index += 1 
                
        return boundaries, boxes
 
    @staticmethod
    # split boxes in the group horizontally
    def splitBoundaries(boundaries, boxes, split_threshold = 25):
        new_boundaries = []
        for i in range(0, len(boundaries)):
            boundary = boundaries[i]         
            box_set = sorted(boxes[i], key = lambda x: (x[2])) # by boundary's top
            index = 0
            box = box_set[index]
            
#             sub_boundaries = [[boundary[0], boundary[1], box[2], box[3]]]
            sub_boundaries = [box]
            
            index += 1
            while index < len(box_set):
                box = box_set[index]
                # merge this box with previous box
                # update the previous box right, top and bot
                if box[2] - sub_boundaries[-1][3] <= split_threshold: # can be a negative value
                    sub_boundaries[-1][3] = max(box[3], sub_boundaries[-1][3]) # right 
                    sub_boundaries[-1][0] = min(box[0], sub_boundaries[-1][0]) # top
                    sub_boundaries[-1][1] = max(box[1], sub_boundaries[-1][1]) # bot
                else:
                    sub_boundaries.append(box)
        
                index += 1
                
            new_boundaries.append(sub_boundaries)

        return new_boundaries, boxes
    
    @staticmethod
    def isOverLap(box1, box2):
        box1_top = box1[0]
        box1_bot = box1[1]
        
        box2_top = box2[0]
        box2_bot = box2[1]
        
        return (box1_top < box2_bot <= box1_bot) or (box1_top <= box2_top < box1_bot) or (box2_top < box1_bot <= box2_bot) or (box2_top <= box1_top < box2_bot)
        

    # TODO: not implemented yet, if you have, put it here
    @staticmethod
    def getSpecies_(image_data, debug = False):

        image = image_data.image
        # displayImage = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        # displayImage1 = displayImage.copy()
        nonTreeMask = image_data.nonTreeMask
        treeMask = image_data.treeMask
        anchorLines = image_data.anchorLines
        contours = image_data.contours
        margin = 3 #margin for tesseract
        contourInfo = PhyloParser.getCntsInfo(contours)

        anchorLabelList = {}
        for line in anchorLines:
            x1, y1, x2, y2, length = line
            potentialCnts = []
            if not isinstance(line, tuple):
                line = tuple(line)
            anchorLabelList[line] = None
            for cntsInfo in contourInfo:
                cnts, index, (top, btm, left, right) = cntsInfo
                if y2 > top and y2 < btm and x2 < left:
                    potentialCnts.append(cntsInfo)

            potentialCnts = sorted(potentialCnts, key=PhyloParser.sortCntsByX)

            # for cntsInfo in potentialCnts:
            #     cv.rectangle(displayImage, (cntsInfo[2][2], cntsInfo[2][0]), (cntsInfo[2][3], cntsInfo[2][1]), color=(255,0,0), thickness=1)
            # plt.imshow(displayImage)
            # plt.show() 
 
            labelSpot = PhyloParser.getLabelSpot(potentialCnts)
            # cv.rectangle(displayImage1, (labelSpot[2], labelSpot[0]), (labelSpot[3], labelSpot[1]), color=(255,0,0), thickness=1)
            
            labelBox = image[labelSpot[0]-margin:labelSpot[1]+margin, labelSpot[2]-margin:labelSpot[3]+margin]
            
            cv.imwrite("tmp.tiff", labelBox)
            label = pytesseract.image_to_string(Image.open('tmp.tiff'))
            anchorLabelList[line] = label

        image_data.species = anchorLabelList
        image_data.speciesNameReady = True 

        return image_data

    #The following static methods (getLabelSpot, getCntsInfo, sortCntsByX) are for getSpecies

    @staticmethod
    def sortCntsByX(item):
        return item[2][2]
    @staticmethod
    def getCntsInfo(contours):
        #return the four tip points of each contour and the index in the contours list
        contourInfo = []
        for index, cnts in enumerate(contours):
            tupleCnts = []
            top = None
            btm = None
            right = None
            left  = None
            for point in cnts:
                x, y = point[0]
                if not top or y<top:
                    top = y
                if not btm or y>btm:
                    btm = y
                if not left or x<left:
                    left = x
                if not right or x>right:
                    right = x
                tupleCnts.append((x,y))
            contourInfo.append([tupleCnts, index, (top, btm, left, right)])
        return contourInfo
    
    
    @staticmethod
    # return the top, bottom, left, right value of the contour
    def getContourInfo(contour):
        max_location = np.max(np.max(contour, axis=1), axis = 0)
        min_location = np.min(np.min(contour, axis=1), axis = 0)
        
        box = [min_location[1], max_location[1], min_location[0], max_location[0]]
        return box
    
    @staticmethod
    def getLabelSpot(potentialCnts):
        #input: sorted potentialSpots, and return the label location
        threshold = 50 #threshold for deciding if it's connected
        textLeft = None
        textRight = None
        textTop = None
        textBtm = None
        for index, cntsInfo in enumerate(potentialCnts):
            cnts, _, (top, btm, left, right) = cntsInfo
            if index==0:
                textTop = top
                textBtm = btm
                textLeft = left
                textRight = right
            else:
                if left - textRight < threshold:
                    if textTop > top:
                        textTop = top
                    if btm > textBtm:
                        textBtm = btm
                    if right > textRight:
                        textRight = right
        return (textTop, textBtm, textLeft, textRight)    

    # End static methos for getSpecies
    
    def makeTree(self, image_data, debug = False):
        
        if image_data.lineDetected and image_data.lineMatched:
        

            # Detect Label and Create Label List

#             image_data = self.getSpecies_(image_data)

            # image_data = self.getSpecies(image_data)


            # Create node from matched lines
            image_data = self.createNodes(image_data)
            if debug:
                # for node in image_data.nodeList:
                #     node.getNodeInfo()
                image_data.displayNodes()
            pass

            # Gather trees from nodes
            image_data = self.createRootList(image_data)
            if debug:
                image_data.displayTrees('regular')
            
            # Check if it's perfectly recovered
            image_data = self.checkDone(image_data)
            
            
            if not image_data.treeReady:
                ## Fix false-positive sub-trees and mandatorily connect sub-trees
                image_data = self.fixTrees(image_data)
                image_data = self.checkDone(image_data)
                
            # fixTrees fixed everything
            if image_data.treeReady:
                image_data.defineTreeHead()
                if debug:
                    print "TODO: draw something here"
                    image_data.displayTrees('final')
            
            # something bad happens
            else:
                print "unknown bad happens"
                image_data.defineTreeHead()## For now, it still defines the tree head. However, we need something else returned to notice it's not perfect
                print self.treeRecover(image_data)
                image_data.treeHead.getNodeInfo()
                if debug:
                    print "TODO: draw something here"
                    image_data.displayTrees('final')
        
        else:
            print "Error! Please do detectLine and matchLine before this method"

        return image_data
    
    
    
    @staticmethod
    def isDotWithinLine(dot, line):
        margin = 5
        x, y = dot
        x1, y1, x2, y2, length = line
        if x1-margin < x and x < x2+margin and y1 - margin < y and y < y2 + margin:
            return True
        else:
            return False
    @staticmethod
    def isLefter(branch, ref):
        x1 = branch[0]
        x2 = ref[0]

        if x2 < x1:
            return True
        return False

    @staticmethod
    def sortNodeByLeftEnd(item):
        return item.branch[0]

    def getNodeBranchOnTheRight(self, dot, nodeList):
        x, y = dot
        potentialNodes = []
        for node in nodeList:
            x1, y1, x2, y2, length = node.branch
            if y1 < y and y2 > y and x1>x:
                potentialNodes.append(node)
        if len(potentialNodes) == 0:
            return False
        else:
            potentialNodes = sorted(potentialNodes, key = self.sortNodeByLeftEnd)
            return potentialNodes[0]

    @staticmethod
    def treeRecover(image_data, mode = 'structure'):
        if mode == 'structure':
            return image_data.treeHead.getTreeString()
        elif mode == 'species':
            return image_data.treeHead.getTreeSpecies(image_data.species)


    def fixTrees(self,image_data):
        rootList = image_data.rootList
        parent = image_data.parent
        children = image_data.children
        horLines = image_data.horLines
        verLines = image_data.verLines
        breakNodeList = []
        tmpList = rootList[:]

        for node in rootList:
            if node in tmpList:
                if not node.isComplete:
                    for breakNode in node.breakSpot[:]:

                        isFixed = False
                        isUpper = True
                        if not ((breakNode.to[0] or breakNode.upperLeave) or (breakNode.to[1] or breakNode.lowerLeave)):
                            pass
                        elif (breakNode.to[0] or breakNode.upperLeave) and not (breakNode.to[1] or breakNode.lowerLeave):
                            print "lower"
                            x1, y1, x2, y2, length = breakNode.branch
                            result = self.getNodeBranchOnTheRight((x2,y2), rootList)
                            if result:  
                                to = list(breakNode.to)
                                to[1] = result
                                breakNode.to = tuple(to)
                                result.whereFrom = breakNode
                                result.origin = node
                                if result.isComplete:
                                    print "remove"
                                    if result in tmpList:
                                        tmpList.remove(result)                                
                                node.breakSpot.remove(breakNode)
                                result.whereFrom = breakNode
                                isFixed = True
                            else:
                                isUpper = False

                        elif not (breakNode.to[0] or breakNode.upperLeave) and (breakNode.to[1] or breakNode.lowerLeave):
                            x1, y1, x2, y2, length = breakNode.branch
                            result = self.getNodeBranchOnTheRight((x1,y1), rootList)
                            if result:
                                to = list(breakNode.to)
                                to[0] = result
                                breakNode.to = tuple(to)
                                node.breakSpot.remove(breakNode)
                                result.origin = node
                                if result.isComplete:
                                    print "remove"
                                    if result in tmpList:
                                        tmpList.remove(result)
                                result.whereFrom = breakNode
                                isFixed = True
                        if isUpper:
                            breakSpot = 'upper'
                        else:
                            breakSpot = 'lower'

                else:
                    pass
        rootList = tmpList[:]
        for node in rootList:
            if node in tmpList:
                if len(node.breakSpot) == 0 and node.whereFrom != None:

                    tmpList.remove(node)
        rootList = tmpList[:]
        if len(rootList) == 1:
            rootList[0].isComplete = True

        image_data.rootList = rootList
        image_data.parent = parent
        image_data.children = children
        return image_data


    def checkDone(self, image_data):
        rootList = image_data.rootList
        isDone = True
        rootNode = rootList[0]

        if not rootNode.isComplete:
            isDone =  False

        if rootNode.root:
            x1, y1, x2, y2, length = rootNode.root
            for node in rootList:
                if node != rootNode:
                    if self.isDotWithinLine((x1, y1), node.branch) or self.isLefter(rootNode.branch, node.branch):
                        isDone = False
        else:
            for node in rootList:
                if node != rootNode:
                    if self.isLefter(rootNode.branch, node.branch):
                        isDone = False

        image_data.treeReady = isDone
        
        return image_data


    @staticmethod
    def countArea(lineList,image_data):

        lineList = sorted(lineList, key = image_data.sortByLeftTop)

        leftTop = lineList[0]
        x1 = leftTop[0]
        y1 = leftTop[1]
        lineList = sorted(lineList, key = image_data.sortByBtmRight)
        btmRight = lineList[0]
        x2 = btmRight[2]
        y2 = btmRight[3]
        area = abs(y2-y1) *abs(x2-x1) 
        if area==0:
            x1 = leftTop[2]
            y1 = leftTop[3]
            x2 = btmRight[2]
            y2 = btmRight[3]
            area = abs(y2-y1) *abs(x2-x1) 


        return area

    def checkError(self, node, mode , image_data):
     
        anchorLines = image_data.anchorLines
        parent = image_data.parent
        if mode == 'upper':
            if node.upperLeave:
                if node.isUpperAnchor:
                    return True
                else:
                    for package in parent:
                        lines, dist = package
                        if self.isSameLine(lines[0], node.upperLeave):
                            newNode = Node(node.upperLeave, lines[1])
                            nodeTo = list(node.to)
                            nodeTo[0] = newNode
                            node.to = tuple(nodeTo)

                    return False
            else:

                if node.isRoot:
                    node.breakSpot.append(node)
                else:
                    rootNode = node.origin
                    rootNode.breakSpot.append(node)
                return False

        elif mode == 'lower':
            if node.lowerLeave:
                if node.isLowerAnchor:
                    return True
                else:
                    for package in parent:
                        lines, dist = package
                        if self.isSameLine(lines[0], node.lowerLeave):
                            newNode = Node(node.lowerLeave, lines[1])
                            nodeTo = list(node.to)
                            nodeTo[1] = newNode
                            node.to = tuple(nodeTo)
                            return False
            else:
                if node.isRoot:
                    node.breakSpot.append(node)
                else:
                    rootNode = node.origin
                    rootNode.breakSpot.append(node)
                return False
        elif mode[:5] == 'inter':

            index = int(mode[5:])
            if index < len(node.interLeave) and node.interLeave[index]:

                if node.isInterAnchor[index]:
                    return True
                else:
                    for package in parent:
                        lines, dist = package
                        if self.isSameLine(lines[0], node.interLeave[index]):
                            newNode = Node(node.interLeave[index], lines[1])
                            node.otherTo[index] = newNode
                            return False
            else:
                if node.isRoot:
                    node.breakSpot.append(node)
                else:
                    rootNode = node.origin
                    rootNode.breakSpot.append(node)
                return False


    def createRootList(self, image_data):
        nodeList = image_data.nodeList
        anchorLines = image_data.anchorLines
        seen = []
        
        rootList = []
        count = 0
        for node in nodeList:
            if node not in seen:
                stack = []
                count +=1
                # print count
                rootNode = None
                stack.append(node)
                foundRoot = False
                while stack:
                    print stack
                    subnode = stack.pop()
                    if subnode in seen:
                        break               
                    else:
                        seen.append(subnode)
                        if subnode.whereFrom:                     
                            stack.append(subnode.whereFrom)
                        else:
                            foundRoot = True
                            subnode.isRoot = True
                            rootNode = subnode
                            rootList.append(subnode)
                if foundRoot:

                    (seen, loop) = self.groupNodes(rootNode, seen, image_data)

                    if loop[0]:
                        rootList.remove(rootNode)


        rootList = sorted(rootList, key = lambda x: -x.area)

        image_data.rootList = rootList
        return image_data

    @staticmethod    
    def drawNode(whatever, node):
        color = (255, 0, 0)
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
        return whatever


    def groupNodes(self, rootNode, seen, image_data):

        # whatever = cv.cvtColor(imge_data.image.copy(), cv.COLOR_GRAY2RGB)


        anchorLines = image_data.anchorLines
        stack = []
        visit = []
        lineList = []
        visit.append(rootNode)
        lineList.append(rootNode.branch)
        rootNode.origin = rootNode
        isComplete = True


        if rootNode.to[0]:
            if rootNode.branch != rootNode.to[0].branch:
                stack.append(rootNode.to[0])
            else:
                tmpNode = list(rootNode.to)
                tmpNode[0] = None
                rootNode.to = tuple(tmpNode)            
        else:
            isAnchorLine = self.checkError(rootNode, 'upper', image_data)
            if isAnchorLine:
                lineList.append(rootNode.upperLeave)
            else:
                if rootNode.to[0]:
                    stack.append(rootNode.to[0])
                else:
                    isComplete = False
                    lineList.append(rootNode.branch)
        if rootNode.to[1]:
            if rootNode.branch != rootNode.to[1].branch:
                stack.append(rootNode.to[1])
            else:
                tmpNode = list(rootNode.to)
                tmpNode[1] = None
                rootNode.to = tuple(tmpNode)
        else:
            isAnchorLine = self.checkError(rootNode, 'lower', image_data)
            if isAnchorLine:
                lineList.append(rootNode.lowerLeave)
            else:
                if rootNode.to[1]:
                    stack.append(rootNode.to[1])
                else:
                    isComplete = False
                    lineList.append(rootNode.branch)
        if not rootNode.isBinary:

            for index, to in enumerate(rootNode.otherTo):
                if to:
                    if rootNode.branch != to.branch:
                        stack.append(to)
                    else:
                        rootNode.otherTo[index] = None
                else:
                    isAnchorLine = self.checkError(rootNode, 'inter%s' %str(index), image_data)
                    if isAnchorLine:
                        anchorLines.append(rootNode.interLeave[index])
                        lineList.append(rootNode.interLeave[index])
                    else:
                        if rootNode.otherTo[index]:
                            stack.append(rootNode.otherTo[index])
                        else:
                            if isComplete:
                                isComplete = False
                                lineList.append(rootNode.branch)
        numNodes = 1

        while stack:
            numNodes +=1
            node = stack.pop()
            # node.getNodeInfo()
            # self.drawNode(whatever, node)
            # plt.imshow(whatever)
            # plt.show()
            visit.append(node)
            node.origin = rootNode
            if node.to[0] :

                if node.to[0] not in seen:
                    seen.append(node.to[0])
                if node.to[0] not in visit and node.branch != node.to[0].branch:
                    stack.append(node.to[0])
                else:
                    loop = [True, node]
                    tmpNode = list(node.to)
                    tmpNode[0] = None
                    node.to = tuple(tmpNode)
                    return (seen, loop)
            else:
                if not self.checkError(node,'upper',image_data):
                    isComplete = False
                    if node.to[0]:
                        stack.append(node.to[0])
                    else:
                        lineList.append(node.branch)
                else:
                    lineList.append(node.upperLeave)
                

            if node.to[1]:

                if node.to[1] not in seen:
                    seen.append(node.to[1])
                if node.to[1] not in visit and node.branch != node.to[1].branch:

                    stack.append(node.to[1])
                else:


                    loop = [True, node]
                    tmpNode = list(node.to)
                    tmpNode[1] = None
                    node.to = tuple(tmpNode)

                    return (seen, loop)
            else:

                if not self.checkError(node, 'lower', image_data):
                    isComplete = False
                    if node.to[1]:
                        stack.append(node.to[1])
                    else:
                        lineList.append(node.branch)
                else:
                    lineList.append(node.lowerLeave)

            if not node.isBinary:

                for index, to in enumerate(node.otherTo):

                    if to:

                        if to not in seen:
                            seen.append(to)
                        if to not in visit and node.branch != to.branch:
                            stack.append(to)
                        else:
                            node.otherTo[index] = None
                    else:

                        isAnchorLine = self.checkError(node, 'inter%s' %str(index), image_data)
                        if isAnchorLine:
                            anchorLines.append(node.interLeave[index])
                            lineList.append(node.interLeave[index])
                        else:
                            if node.otherTo[index]:
                                stack.append(node.otherTo[index])
                            else:
                                if isComplete:
                                    isComplete = False
                                    lineList.append(node.branch)




        rootNode.numNodes = numNodes

        area = self.countArea(lineList, image_data)

        rootNode.area = area


        if isComplete:
            rootNode.isComplete = True
        loop = False, None
        return (seen, loop)    


    # @staticmethod
    # def createNodes(image_data):
    #     pass

    def createNodes(self,image_data):
        
        if not image_data.speciesNameReady:
            print "Species names not found! Use dummy names."

        parent = image_data.parent
        children = image_data.children
        anchorLines = image_data.anchorLines
        nodeList = []
        for item in children:
            lines, dist = item
            (branch, hlines) = lines
            match = False
            isAnchor = []
            hlines = list(hlines)
            for index, line in enumerate(hlines):
                isAnchor.append(False)
                if line:
                    for anchorLine in anchorLines:
                        if self.isSameLine(line, anchorLine):
                            hlines[index] = anchorLine
                            isAnchor[index] = True
            hlines = tuple(hlines)
            for pitem in parent:
                ((root, pbranch), pdist) = pitem
                if self.isSameLine(branch, pbranch):
                    match = True
                    if len(hlines) <= 2:
                        upperLeave, lowerLeave = hlines
                        a = Node(root, branch, upperLeave, lowerLeave) 
                        a.isUpperAnchor = isAnchor[0]
                        a.isLowerAnchor = isAnchor[1]           
                    else:
                        hlines = list(hlines)
                        interLine = []
                        for index, line in enumerate(hlines):
                            
                            if index== 0:
                                upperLeave = line
                            elif index == 1:
                                lowerLeave = line
                            else:
                                interLine.append(line)
                        a = Node(root,branch,upperLeave,lowerLeave)
                        a.isUpperAnchor = isAnchor[0]
                        a.isLowerAnchor = isAnchor[1]
                        a.interLeave = interLine
                        numInterLeave = len(hlines) - 2
                        for index in range(numInterLeave):
                            a.otherTo.append(None)
                            a.isInterAnchor.append(isAnchor[index+2])
                            a.interLabel.append(None)
                        a.isBinary = False
                    nodeList.append(a)

            if not match:
                if len(hlines) <=2:
                    upperLeave, lowerLeave = hlines
                    a = Node(None, branch ,upperLeave, lowerLeave)
                    a.isUpperAnchor = isAnchor[0]
                    a.isLowerAnchor = isAnchor[1]
                else:
                    hlines = list(hlines)
                    for index, line in enumerate(hlines):
                        interLine = []
                        if index == 0:
                            upperLeave = line
                        elif index == 1:
                            lowerLeave = line
                        else:
                            interLine.append(line)
                    a = Node(None, branch, upperLeave, lowerLeave)
                    a.isUpperAnchor = isAnchor[0]
                    a.isLowerAnchor = isAnchor[1]
                    a.interLeave = interLine
                    numInterLeave = len(hlines) - 2
                    for index in range(numInterLeave):
                        a.interLabel.append(None)
                        a.otherTo.append(None)
                        a.isInterAnchor.append(isAnchor[index+2])
                        # a.isInterAnchor.append(False)

                    a.isBinary = False

                nodeList.append(a)
        nodeList = sorted(nodeList, key = lambda x: -x.branch[0])

        for node in nodeList:
            if not node.isConnected:
                potentialUpperLeaves = []
                potentialLowerLeaves = []
                potentialInterLeaves = []
                if not node.isBinary:
                    for line in node.interLeave:
                        potentialInterLeaves.append([])
                if not (node.isUpperAnchor and node.isLowerAnchor and node.isBinary):
                    for subNode in nodeList:
                        if subNode!=node and subNode.branch != node.branch and subNode.root:
                            if node.upperLeave and not node.isUpperAnchor:
                                if self.isSameLine(subNode.root, node.upperLeave):
                                    score = self.evaluateNode(subNode)
                                    potentialUpperLeaves.append((subNode, score))
                            if node.lowerLeave and not node.isLowerAnchor:
                                if self.isSameLine(subNode.root, node.lowerLeave):
                                    score = self.evaluateNode(subNode)
                                    potentialLowerLeaves.append((subNode, score))
                            if not node.isBinary:
                                for index, interLine in enumerate(node.interLeave):
                                    if self.isSameLine(interLine, subNode.root):
                                        score = self.evaluateNode(subNode)
                                        potentialInterLeaves[index].append((subNode, score))

            potentialUpperLeaves = sorted(potentialUpperLeaves, key = lambda x: -x[1])
            potentialLowerLeaves = sorted(potentialLowerLeaves, key = lambda x: -x[1])

            
            tmpTo = list(node.to)
            if len(potentialUpperLeaves) != 0:
                targetNode = potentialUpperLeaves[0][0]
                if not targetNode.isConnected:
                    tmpTo[0] = targetNode
                    targetNode.isConnected = True
                    targetNode.whereFrom = node
                else:
                    refNode = targetNode.whereFrom
                    if PhyloParser.betterNode(node, refNode):
                        tmpTo[0] = targetNode
                        targetNode.whereFrom = node

            if len(potentialLowerLeaves) !=0:
                targetNode = potentialLowerLeaves[0][0]
                if not targetNode.isConnected:
                    tmpTo[1] = targetNode
                    targetNode.isConnected = True
                    targetNode.whereFrom = node
                else:
                    refNode = targetNode.whereFrom
                    if PhyloParser.betterNode(node, refNode):
                        tmpTo[1] = targetNode
                        targetNode.whereFrom = node
            node.to = tuple(tmpTo)
            if not node.isBinary:               
                for index, inter in enumerate(potentialInterLeaves):
                    if len(inter)!=0:
                        # print interLeave
                        tmpList = sorted(inter, key = lambda x: -x[1])
                        targetNode = tmpList[0][0]
                        if not targetNode.isConnected:
                            node.otherTo[index] = targetNode
                            targetNode.isConnected = True
                            targetNode.whereFrom = node
                        else:
                            refNode = targetNode.whereFrom
                            if PhyloParser.betterNode(node, refNode):
                                node.otherTo[index] = targetNode
                                targetNode.whereFrom = node

        image_data.nodeList = nodeList
        return image_data

    @staticmethod
    def betterNode(node1, node2):
        score1 = 0
        score2 = 0
        if node1.score:
            score1 = node1.score
        else:
            score1 = PhyloParser.evaluateNode(node1)
        if node2.score:
            score2 = node2.score
        else:
            score1 = PhyloParser.evaluateNode(node2)
        score1 += math.sqrt(int(PhyloParser.countNodeArea(node1)))
        score2 += math.sqrt(int(PhyloParser.countNodeArea(node2)))

        if score1> score2:
            return True
        else:
            return False


    @staticmethod
    def countNodeArea(node):
        if node.branch:
            if node.lowerLeave:
                return abs((node.lowerLeave[2] - node.branch[0]) * (node.branch[1] - node.lowerLeave[1]))
            if node.upperLeave:
                return abs((node.branch[0] - node.upperLeave[2]) * (node.branch[4]))
            return 0
        else:
            return 0

    @staticmethod
    def evaluateNode(node):
        score = 0
        if node.upperLeave:
            score +=1
            if node.to[0]:
                score +=1
                if node.to[0].score:
                    score += node.to[0].score
            elif node.isUpperAnchor:
                score +=1
        if node.lowerLeave:
            score +=1
            if node.to[1]:
                if node.to[1].score:
                    score+= node.to[1].score
            elif node.isLowerAnchor:
                score +=1
        if not node.isBinary:
            for index, line in enumerate(node.interLeave):
                if line:
                    score+=1
                    if node.otherTo[index]:
                        score+=1
                        if node.otherTo[index].score:
                            score+=node.otherTo[index].score
                    elif node.isInterAnchor[index]:
                        score +=1
        node.score = score
        return score



    @staticmethod
    def isSameLine(aline, bline, margin = 5):
        ax1, ay1, ax2, ay2, alength = aline
        bx1, by1, bx2, by2, blength = bline

        if ay1 - margin < by1 and ay2 + margin> by2 and ax1 - margin < bx1 and ax2 + margin > bx2 and alength + margin > blength:
            return True
        elif by1 - margin < ay1 and by2 + margin > ay2 and bx1 - margin < ax1 and bx2 + margin > ax2 and blength + margin > alength:
            return True
        else:
            return False        
        
#     @staticmethod
#     def splitBoxGroup_(boxGroup, text_height_threshold = 5):
#         
#         print boxGroup
#         index = 0
#         while index < len(boxGroup) and boxGroup[index][1][1] - boxGroup[index][1][0] < text_height_threshold:
#             index += 1
#         
#         # all boxes can't pass threshold
#         if index == len(boxGroup):
#             top = min(boxGroup[0][1][0], boxGroup[-1][1][0])
#             bot = max(boxGroup[0][1][1], boxGroup[-1][1][1])
#             left = min(boxGroup[0][1][2], boxGroup[-1][1][2])
#             right = max(boxGroup[0][1][3], boxGroup[-1][1][3])
#             return [[top, bot, left, right]]
#         
#         print "start from index ", index
#         if len(boxGroup) > 1:
#             sub_group = [boxGroup[index][1]]
#             print sub_group
#             for i in range(index+1, len(boxGroup)):
# 
#                 box = boxGroup[i][1]
#                 print "i=,", i, box
#                 print "sub_group=", sub_group
#                 
#                 to_sg_index = []
#                 count = 0
#                 for j, sg in enumerate(sub_group):
#                     print "scan sub_group j=",j, sg, PhyloParser.isOverLap(sg, box)
#                     if PhyloParser.isOverLap(sg, box):
#                         to_sg_index.append(j)
#                         
#                     print "to_sg_index", to_sg_index
#                 # belong to only one sub_group
#                 if len(to_sg_index) == 1:
#                     print "update sub_group"
#                     #update sub_group
#                     index = to_sg_index[0]
#                     top = min(sub_group[index][0], box[0])
#                     bot = max(sub_group[index][1], box[1])
#                     left = min(sub_group[index][2], box[2])
#                     right = max(sub_group[index][3], box[3])
#                     print [top, bot, left, right]
#                     sub_group[index] = [top, bot, left, right]
#                 # belong to no sub_group
#                 # add a new sub_group
#                 elif len(to_sg_index) == 0:
#                     sub_group.append(box)
#                     print "add new elemenet in sub group"
#                     # sort from top to bot
#                     sub_group = sorted(sub_group, key = lambda x: (x[0]))
#                     
#                 # overlap move then 2 sub groups, split the current box
#                 else:
#                     print "split"
# #                     print to_sg_index
#                     for index in to_sg_index:
#                         # top one
#                         if index == 0:
#                             top = min(sub_group[index][0], box[0])
#                         else:
#                             top = sub_group[index][0]
#                             
#                         # bot one
#                         if index == len(sub_group) - 1:
#                             bot = max(sub_group[index][1], box[1])
#                         else:
#                             bot = sub_group[index][1]
#                             
#                         left = min(sub_group[index][2], box[2])
#                         right = max(sub_group[index][3], box[3])
#                     
#                         sub_group[index] = [top, bot, left, right]
#             
#         else:
#             if boxGroup[0][1][1] - boxGroup[0][1][0] < text_height_threshold:
#                 sub_group = []
#             else:
#                 sub_group = [boxGroup[0][1]]
#                 
#         # modify boundaries
# #         sub_group = PhyloParser.stitchBoxes(sub_group)
#         
#         return sub_group