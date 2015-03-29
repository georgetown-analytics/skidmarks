###############################################################################
# Information
###############################################################################
# Created by Linwood Creekmore

# In partial fulfillment of Georgetown University Data Analytics Graduate Certificate

# March 25, 2015

# https://plus.google.com/+LinwoodCreekmoreIII/

###############################################################################
# Imports
###############################################################################

import math
import unicodecsv
import pandas as pd
import matplotlib.pyplot as plt
from unum.units import * 
import numpy as np
from pylab import plot
from os import walk
from os.path import basename
import unicodecsv as csv
import logging as log

##########################################################################
## Module Constants
##########################################################################

# expects models to be in same directory
  = os.path.dirname(__file__)
.. = os.path.join(MODEL_DIR, "heating.pickle")
.. = os.path.join(MODEL_DIR, "cooling.pickle")

###############################################################################
# Helper Functions 2
###############################################################################

def dotproduct(x,y):
    return math.sqrt(x**2 + y**2)

def tomph(velocity):
    return velocity * 2.24
    

def getDriver(dirName):
    return int(basename(dirName))


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
###############################################################################
# Helper Functions
###############################################################################
'''
def getDisplacement(x,last_x,y,last_y):
    increment_traveled = math.sqrt( (x - last_x)**2 + (y - last_y)**2 ) 
    return increment_traveled
        
        # increment total distance by displacement
'''
'''       
def get_total_distance(distance,increment_traveled):
    distance += increment_traveled
    return distance
'''
'''
We are dealing with a positon vector; therefore, we must calculate the x and y components of the position vector (i.e. car's position at point of time) for the car as it moves in space.
Velocity is measured by dividing the change in position by the change in time.
The magnitude of the velocity vector can be found using the scalar dot product, which is equivalent to Pythagorean theorem.
Acceleratioin is measured by dividing the change in velocity by the change in time
'''

        # Calculate the average horizontal and vertical component of velocity; we ignore time because it's always equal to 1 second
        

def getvelocity(x,y,last_x,last_y):
    x_avg_vel = abs(abs(x) - abs(last_x)) 
    y_avg_vel = abs(abs(y) - abs(last_y))  
    return x_avg_vel, y_avg_vel
      

        # Calculate the horizontal and vertical acceleration components; this will be critical to determining turns, braking, etc.

def getacceleration(x_avg_vel,last_x_avg_vel,y_avg_vel,last_y_avg_vel):
    x_avg_acl = x_avg_vel - last_x_avg_vel
    y_avg_acl = y_avg_vel - last_y_avg_vel
    return x_avg_acl,y_avg_acl


def getincrement(x,last_x,y,last_y):
    # calculate displacement
    increment_traveled = math.sqrt( (x - last_x)**2 + (y - last_y)**2 )
    return increment_traveled

    '''
    if increment_traveled > max_velocity:
        max_velocity = increment_traveled # we have a new record!

    print "\nseconds: %d incremement: %f \nx velocity: %f, y velocity: %f \nx accel: %f,y accel: %f" % (seconds, increment_traveled, x_avg_vel, y_avg_vel, x_avg_acl, y_avg_acl)
    '''    
    
# Calculate the direction

def getDirection(y,x):
    direction_rad = math.atan2(y,x)
    direction = math.degrees(direction_rad) 
    if direction < 0:
        direction += 360
    return direction 

# Calculate cardinal direction

def getcardinal_direct(direction):
    carddir = ''
    if direction >= 0 and direction <= 22.5:
        carddir = 'East'
    elif direction > 337.5 and direction <=360:
        carddir = 'East' 
    elif direction > 22.5 and direction <= 67.5:
        carddir = 'Northeast'
    elif direction > 67.5 and direction <= 112.5:
        carddir = 'North'
    elif direction > 112.5 and direction <= 157.5:
        carddir  = 'Northwest'
    elif direction > 157.5 and direction <= 202.5:
        carddir  = 'West'
    elif direction > 202.5 and direction <= 247.5:
        carddir = 'Southwest'
    elif direction > 247.5 and direction <= 292.5:
        carddir  = 'South'
    elif direction > 292.5 and direction <= 337.5:
        carddir  = 'Southeast'

    return carddir

# counting braking or deceleration events
'''
def countdeceleration_and_stops(x_avg_acl,y_avg_acl,x_avg_vel,y_avg_vel):

    if x_avg_acl < 0 and y_avg_acl < 0:
        braking_event += 1
        return braking_event

    if last_x_avg_vel == 0 and y_avg_vel == 0:
        stops += 1
        return stops


# Keeping track of highest velocity
def maxvelocity():
    if increment_traveled > max_velocity:
        max_velocity = increment_traveled # we have a new record!
        return max_velocity
'''





###############################################################################
# Main Functions
###############################################################################

def createfile():
    #with open('/Users/linwoodc3/Google Drive/Python/projects/test.csv', 'rU') as infile:

    #open the 
    with open('/home/captainobvious/Downloads/110.csv', 'rU') as infile:
        reader = csv.DictReader(infile, delimiter=',', quotechar='"')

        with open('/home/captainobvious/Downloads/110metrics.csv', 'wb') as outfile:
            writer = csv.writer(outfile)
            
            writer.writerow(['Velocity (mph)', 'Acceleration (m/s^2)','Time (s)', 'Increment Traveled (m)','Direction (deg)','Direction(card)'])

            infile.next() #skip first line with headings
            last_x, last_y, = 0.0, 0.0
            seconds = 0 
            distance = 0
            stops = 0
            braking_event = 0
            last_x_avg_vel= 0
            last_y_avg_vel= 0
            max_velocity = 0
            accelerations = 0
            decelerations = 0
            x_avg_vel = 0
            y_avg_vel = 0
            x_avg_acl = 0
            y_avg_acl = 0

            metrics = []

            for l in infile:
            
                x, y = l.split(',')
                x, y = float(x), float(y)

                x_avg_vel,y_avg_vel = getvelocity(x,y,last_x,last_y)
                x_avg_acl,y_avg_acl = getacceleration(x_avg_vel,last_x_avg_vel,y_avg_vel,last_y_avg_vel)

                metrics.append(tomph(dotproduct(x_avg_vel, y_avg_vel)))
                metrics.append(dotproduct(x_avg_acl,y_avg_acl))
                metrics.append(seconds)
                metrics.append(getincrement(x,last_x,y,last_y))
                metrics.append(getDirection(y,x))
                metrics.append(getcardinal_direct(getDirection(y,x)))

                
                writer.writerow(metrics)

                metrics = []

                seconds += 1

                last_x, last_y = x, y
                last_x_avg_vel, last_y_avg_vel = x_avg_vel, y_avg_vel

                
                

###############################################################################
# 'Main' Function
############################################################################### 

if __name__ == '__main__':
    createfile()