# -*- coding: utf-8 -*-

###############################################################################
# Information
###############################################################################
# Created by Linwood Creekmore 
# Input from Vikram Mittal

# In partial fulfillment of the requirements for the Georgetown University Data Analytics Graduate Certificate Program

# useful links
# http://stackoverflow.com/questions/27504870/sklearn-kmeans-get-class-centroid-labels-and-reference-to-a-dataset  -> returns the class of the cluster; can see where each driver's trip fell

# May 1, 2015

# https://plus.google.com/+LinwoodCreekmoreIII/

###############################################################################
# Imports
###############################################################################

###############################################################################
# Helper Functions
###############################################################################

# This is a generic dot product or Euclidean distance function


def dotproduct(x,y):
    return math.sqrt(x**2 + y**2)


# This is used to convert the meters per second velocity into a miles per hour value.  
def tomph(velocity):
    return velocity * 2.24


'''
This function is used to convert the units for increment traveled from meters to feet; because we measure over each second, 
 a more human readable unit is needed.
'''
def meterstofeet(args):
    return args * 3.28    


def getDriver(dirName):
    return (basename(dirName))


def getFileBaseName(baseName):
    return fileName.split('.')[0]


def getTrip(baseName):
    return int(getFileBaseName(baseName))


def isFloat(str):
    try:
        float(str)
        return True
    except:
        return False

