# -*- coding: utf-8 -*-

###############################################################################
# Information
###############################################################################
# Created by Linwood Creekmore 
# Starter file by Danny Holloway of HumanGeo
# Significant input from Vikram Mittal

# In partial fulfillment of the requirements for the Georgetown University Data Analytics Graduate Certificate Program

# March 5, 2015

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
import os
import math
import statistics
from os import walk
from os.path import basename
import unicodecsv as csv
import logging as log
import time


###############################################################################
# File Paths
###############################################################################

path = os.path.abspath(os.getcwd())
LOG_DIR    = os.path.normpath(os.path.join(os.path.dirname(path),os.path.basename(path),"log"))
INPUT_DIR  = os.path.normpath(os.path.join(os.path.dirname(path),os.path.basename(path),"input","test"))
OUTPUT_DIR = os.path.normpath(os.path.join(os.path.dirname(path),os.path.basename(path),"output","trip"))
OUTPUT_DIR2 = os.path.normpath(os.path.join(os.path.dirname(path),os.path.basename(path),"output","aggregate"))

###############################################################################
# Create Logger, File Handlers, Formatters, and Filters (as needed)
###############################################################################

class SingleLevelFilter(log.Filter):
    def __init__(self, passlevel, reject):
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        if self.reject:
            return (record.levelno != self.passlevel)
        else:
            return (record.levelno == self.passlevel)

logger = log.getLogger("DataQuality")
logger.setLevel(log.DEBUG)

infoHandler = log.FileHandler(os.path.join(LOG_DIR,"Displacement_Summary.log"), mode='w')
infoFilter = SingleLevelFilter(log.INFO, False)
infoHandler.addFilter(infoFilter)

warnHandler = log.FileHandler(os.path.join(LOG_DIR,"Warnings.log"), mode='w')
warnFilter = SingleLevelFilter(log.WARNING, False)
warnHandler.addFilter(warnFilter)

logger.addHandler(infoHandler)
logger.addHandler(warnHandler)

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
# Key Metric Functions
###############################################################################



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

# Calculate displacement
def getincrement(x,last_x,y,last_y):

    increment_traveled = math.sqrt( (x - last_x)**2 + (y - last_y)**2 )
    return increment_traveled  
    
# Calculate the direction

def getDirection(y,x):
    direction_rad = math.atan2(y,x)
    direction = math.degrees(direction_rad) 
    if direction < 0:
        direction += 360
    return direction 

###
#This calculation splits cardinal directions into buffer zones of 45 degrees each. We get the directional heading from 
#the x,y location and output a string based on the buffer zone conditional statement below
###

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


###############################################################################
# Main Functions
###############################################################################



def createfile(dirName, fileName):
    driver = getDriver(dirName)         # obtain driver id from directory name
    # obtain trip id from file name
    

    # This will open the driver directory/folder in a directory with all the driver files.  This should go through all 200 trip files for one driver.
    
    with open(os.path.join(os.path.normpath(path), dirName, fileName), 'rU') as infile:
        reader = csv.DictReader(infile, delimiter=',', quotechar='"')

        # This creates the output csv file that will hold all the calculated metrics

        with open(os.path.join(OUTPUT_DIR,fileName), 'wb') as outfile:
            writer = csv.writer(outfile)

            # This writes the header row for our output file using trip/driver IDs from Vik's IDify.py file
            
            for idx, row in enumerate(reader):
                if idx == 0:
                    trip_id = 1

                    writer.writerow(['driver_id', 'trip_id', 'Velocity (mph)', 'Acceleration (mph per s)','Time (s)', 'Increment Traveled (feet)','Change in Direction per s','Direction (deg)','Direction(card)'])

                    if not row == { 'driver_id':'driver_id', 'trip_id':'trip_id', 'x':'x', 'y':'y'}:


                        logger.warning("Missing or invalid header for driver %s and trip %s" % (driver, trip_id))

                    else:

                        if len(row) != 4:

                            logger.warning("Too few/many values in row %s for driver %s and trip %s" % (idx, driver, trip_id))
            

                    # skip first line with headings.

                    infile.next() 

                    # Here we establish all the zero values for the key metrics we iterate.  These are defined in the key metrics documentation.

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
                    last_heading = 0


                    # Creating an list to append all the calculated key metric values
                    metrics = []

                    # Establish the trip id value to iterate over for our database
                    #trip_id = 1

                    # We loop through the row values of x,y and calculate the key metric values, and then append the value to the metrics list above.

                    for l in infile:
                    
                        driver_id, trip_id, x, y = l.split(',')
                        driver_id, trip_id, x, y = int(driver_id), int(trip_id), float(x), float(y)
            

                        x_avg_vel,y_avg_vel = getvelocity(x,y,last_x,last_y)
                        x_avg_acl,y_avg_acl = getacceleration(x_avg_vel,last_x_avg_vel,y_avg_vel,last_y_avg_vel)

                        metrics.append(driver)
                        metrics.append(trip_id)
                        metrics.append(tomph(dotproduct(x_avg_vel, y_avg_vel)))
                        metrics.append(dotproduct(x_avg_acl,y_avg_acl))
                        metrics.append(seconds)
                        metrics.append(getincrement(x,last_x,y,last_y))
                        metrics.append(getDirection(y,x) - getDirection(last_y,last_x))
                        metrics.append(getDirection(y,x))
                        metrics.append(getcardinal_direct(getDirection(y,x)))

                        # We write the identifying and key metrics values to our csv

                        writer.writerow(metrics)

                        # Next, we clear our metrics list for the next loop, iterate our time and trip ID values, and store the previous locations and velocity.

                        metrics = []
                        seconds += 1
                        trip_id += 1
                        last_x, last_y = x, y

                        last_x_avg_vel, last_y_avg_vel = x_avg_vel, y_avg_vel
                        direction = last_heading

    
    with open(os.path.join(OUTPUT_DIR, fileName), 'rU') as infile:
        reader = csv.DictReader(infile, delimiter=',', quotechar='"')

        with open(os.path.join(OUTPUT_DIR2,fileName), 'wb') as outfile:
            writer = csv.writer(outfile)

            df = pd.read_csv(os.path.join(OUTPUT_DIR,fileName))

            writer.writerow(['driver_id', 'trip_id', 'Average Velocity (mph)', 'Max Velocity', 'Velocity Stdev','Average Acceleration (mph per s)', 'Max Acceleration (mph per s)', ' Acceleration Stdev','Max Direction Change per sec', ' Direction Stdev','Time (s)'])

            agvalues = []
            
            

            agvalues.append(driver)
            agvalues.append(df.loc[1][1])
            agvalues.append(df.loc[2:]['Velocity (mph)'].mean())
            agvalues.append(df.loc[2:]['Velocity (mph)'].max())
            agvalues.append(statistics.stdev(df.loc[1:]['Velocity (mph)']))
            agvalues.append(df.loc[2:]['Acceleration (mph per s)'].mean())
            agvalues.append(df.loc[2:]['Acceleration (mph per s)'].max())
            agvalues.append(statistics.stdev(df.loc[1:]['Acceleration (mph per s)']))
            agvalues.append(df.loc[2:]['Change in Direction per s'].max())
            agvalues.append(statistics.stdev(df.loc[2:]['Direction (deg)']))
            agvalues.append(df.loc[1:]['Time (s)'].max())

            
            writer.writerow(agvalues)
                    
            agvalues = []
                        

###############################################################################
# 'Main' Function
############################################################################### 

if __name__ == '__main__':
    for dirName, subdirList, fileList in walk(INPUT_DIR):
        if basename(dirName).isdigit():                         # directory likely represents a driver
            for fileName in fileList:
                if fileName.endswith('.csv') :       # trip data is in .csv files and a numeric file name is likely a trip file, and getFileBaseName(fileName).isdigit()
                    createfile(dirName, fileName)
