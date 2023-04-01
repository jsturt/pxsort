from itertools import islice
import numpy as np
import cv2 
import sys


def srt(unsorted,sign=1):
    # calculate luma 
    key = lambda rgb: np.array( [ luma(elem) for elem in rgb ] )
    # sort by luma
    indices = key(unsorted).argsort()
    # sign determines what direction the sort is returned in
    return unsorted[indices[::sign]]

def luma(px):
    return (float(px[0]) + float(px[0]) + float(px[1]) + float(px[1]) + float(px[1]) + float(px[2]))/6.0

def contrastMask(x):
    # Return numpy array of b/w mask by luma threshold
    upper = 0.95
    lower = 0.35
    # calculate luma for each pixel
    for ix,iy in np.ndindex(x.shape[:2]):
        x[ix,iy] = luma(x[ix,iy])

    # mask by threshold
    x[x>upper*255]=0
    x[x<lower*255]=0
    x[x>lower*255]=255
    return x

def horizSpanMask(src):
    cmask = contrastMask(np.copy(src))
    buffer = np.empty(cmask.shape)
    imItr = np.ndindex(cmask.shape[:2])
    for iy,ix in imItr:
        # skip if finds black
        if(not bool(sum(cmask[iy,ix]))):
            continue
        n=0
        for n in range(0,(cmask.shape[:2][1] - ix)):
            if(not bool(sum(cmask[iy,ix+n]))):
                break
        buffer[iy,ix] = np.array([n,0,0])
        [next(imItr,None) for i in range(n)]
    return buffer

def pixelSortHoriz(img):
    mask = horizSpanMask(img)
    imItr = np.ndindex(mask.shape[:2])
    for iy,ix in imItr:
        # skip if finds black
        if(not bool(mask[iy,ix][0])):
            continue 
       
        # find end of span
        offset = ix + int(mask[iy,ix][0])
        # find x indices of span
        cols = [ix + n for n in range(int(mask[iy,ix][0]))]
        # set span to sorted values
        img[iy,cols] = srt(img[iy,cols])
        #img[iy,cols] = 0
        
        [next(imItr,None) for i in range(int(mask[iy,ix][0]))]
    return img


# load img
if(not bool(len(sys.argv)-1)):
   print("no file provided")
   quit()

filepath = str(sys.argv[1])
im = cv2.imread(filepath)

cv2.imshow('contrast mask',contrastMask(np.copy(im)))
cv2.waitKey(0)
cv2.destroyAllWindows()

if(bool(input("continue?"))):
    quit()

# switch to RGB 
cv2.imshow('sorted image', pixelSortHoriz(im))
cv2.imwrite('output.jpg',pixelSortHoriz(im))
cv2.waitKey(0)
cv2.destroyAllWindows()
