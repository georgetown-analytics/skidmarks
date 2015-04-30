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
import stop
import turn
import deceleration
import bigdeceleration
import AggressiveTurn
import maxdecel


###############################################################################
# File Paths
###############################################################################

path        = os.path.abspath(os.getcwd())
LOG_DIR     = os.path.normpath(os.path.join(os.path.dirname(path),os.path.basename(path),"log"))
INPUT_DIR   = os.path.normpath(os.path.join(os.path.dirname(path),os.path.basename(path),"output","test"))
OUTPUT_DIR  = os.path.normpath(os.path.join(os.path.dirname(path),os.path.basename(path),"output","trip"))
OUTPUT_DIR2 = os.path.normpath(os.path.join(os.path.dirname(path),os.path.basename(path),"output","aggregate"))

#print "Home: %s, \nAbsolute: %s \nInput:%s \nOutput: %s." % (
    #path, INPUT_DIR, OUTPUT_DIR2, OUTPUT_DIR)


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


###############################################################################
# Key Metric Functions
###############################################################################


'''
We are dealing with a positon vector; therefore, we must calculate the x and y components of the position vector (i.e. car's position at point of time) for the car as it moves in space.
Velocity is measured by dividing the change in position by the change in time.
The magnitude of the velocity vector can be found using the scalar dot product, which is equivalent to using the Pythagorean theorem.
Acceleration is measured by dividing the change in velocity by the change in time
'''

# Calculate total Displacement; since we always start from 0,0, it's just the dot product of the last point

# Calculate the average horizontal and vertical component of velocity; we ignore time because it's always equal to 1 second
def getVelocity(x,y,last_x,last_y):
    x_avg_vel = abs(abs(x) - abs(last_x)) 
    y_avg_vel = abs(abs(y) - abs(last_y))  
    return x_avg_vel, y_avg_vel
      

# Calculate the horizontal and vertical acceleration components; this will be critical to determining turns, braking, etc.
def getAcceleration(x_avg_vel,last_x_avg_vel,y_avg_vel,last_y_avg_vel):
    return x_avg_acl,y_avg_acl

# Calculates the NEW heading of the vehicle
def heading(y,x,last_y,last_x):
    return getDirection(y-last_y,x-last_x)

# Calculate the direction
def getDirection(y,x):
    direction_rad = math.atan2(y,x)
    direction = math.degrees(direction_rad) 
    if direction < 0:
        direction += 360
    return direction 

# Calculate displacement
def getIncrement(x,last_x,y,last_y):
    increment_traveled = math.sqrt( (x - last_x)**2 + (y - last_y)**2 )
    return increment_traveled




###
#This calculation splits cardinal directions into buffer zones of 45 degrees each. We get the directional heading from 
#the x,y location and output a string based on the buffer zone conditional statement below
###

def getCardinalDirection(direction):
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


def createFile(dirName, fileName):
    driver = getDriver(dirName)         # obtain driver id from directory name
    #df2 = pd.read_csv(os.path.join(OUTPUT_DIR,fileName))

    trip = fileName.split('_')[1]

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

                    writer.writerow(['driver_id', 'trip_id', 'Velocity (mph)', 'Acceleration (mph per s)','Absolute Acceleration','Time (s)', 'Increment Traveled (feet)','Change in Direction per s','Direction (deg)','Direction(card)'])

                    if not row == { 'driver_id':'driver_id', 'trip_id':'trip_id', 'x':'x', 'y':'y'}:

                        logger.warning("Missing or invalid header for driver %s and trip %s" % ("driver", trip_id))

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
                    last_heading = 0.0
                    cur_heading = 0.0
                    lastvel = 0
                    lastaccel = 0
                    absoluteaccel = 0


                    # Creating an list to append all the calculated key metric values
                    metrics = []

                    # Establish the trip id value to iterate over for our database
                    #trip_id = 1

                    # We loop through the row values of x,y and calculate the key metric values, and then append the value to the metrics list above.

                    for l in infile:
                    
                        driver_id, trip_id, x, y, = l.split(',') 
                        driver_id, trip_id, x, y = int(driver_id), int(trip_id), float(x), float(y)
                        x_avg_vel,y_avg_vel = getVelocity(x,y,last_x,last_y)


                        metrics.append(fileName.split('_')[0]) #append driver #
                        metrics.append(trip_id) #append trip id
                        velocity = tomph(dotproduct(x_avg_vel, y_avg_vel))
                        if velocity > 200:
                            velocity = lastvel

                        metrics.append(velocity)

                        holdingvel = velocity - lastvel
                        absoluteaccel = velocity - lastvel
                        if abs(holdingvel) > 200:
                            holdingvel = lastaccel
                        

                        metrics.append(holdingvel) #acceleration
                        if absoluteaccel < 0:
                            absoluteaccel = abs(absoluteaccel)

                        metrics.append(absoluteaccel) #absolute acceleration
                        metrics.append(seconds) #time
                        metrics.append(getIncrement(x,last_x,y,last_y)) #distance traveled

                        last_heading = cur_heading
                        cur_heading = heading(y,x, last_y, last_x)
                        holding = abs(cur_heading - last_heading)
                        if holding >= 45 and getIncrement(x,last_x,y,last_y) <= 2:
                            holding = 0
                        elif holding >= 180:
                            holding = 360 - holding

                        metrics.append(holding) # enters change in direction
                        metrics.append(heading(y,x, last_y, last_x)) # calculates the heading in degrees
                        metrics.append(getCardinalDirection(heading(y,x,last_y,last_x))) # calculates cardinal position

                        # We write the identifying and key metrics values to our csv

                        writer.writerow(metrics)

                        # Next, we clear our metrics list for the next loop, iterate our time and trip ID values, and store the previous locations and velocity.

                        metrics = []
                        seconds += 1
                        trip_id += 1
                        last_x, last_y = x, y
                        lastvel = velocity
                        holdingvel = lastaccel
                        last_x_avg_vel, last_y_avg_vel = x_avg_vel, y_avg_vel
                        direction = last_heading

    
    with open(os.path.join(OUTPUT_DIR, fileName), 'rU') as infile:
        reader = csv.DictReader(infile, delimiter=',', quotechar='"')

        with open(os.path.join(OUTPUT_DIR2,fileName), 'wb') as outfile:
            writer = csv.writer(outfile, lineterminator = '\n')

            df = pd.read_csv(os.path.join(OUTPUT_DIR,fileName))
            writer.writerow(['driver_id', 'trip_id', 'Average Velocity (mph)', 'Max Velocity', 'Velocity Stdev','Average Acceleration (mph per s)', 'Max Acceleration (mph per s)', ' Acceleration Stdev','Displacement','Total Distance Traveled','Max Direction Change per sec', ' Direction Stdev','Time (s)', 'Turns', 'Aggressive Turns', 'Stops', 'Large Deceleration Events', 'Deceleration Events', 'Max Deceleration Event'])
            df2 = pd.read_csv(os.path.join(INPUT_DIR,str(driver),fileName))
            
            print "Driver is", driver
            print "Trip is", trip
            agvalues = []        

            agvalues.append(fileName.split('_')[0])
            agvalues.append(df.loc[1][1])
            agvalues.append(df.loc[2:]['Velocity (mph)'].mean()) #Average velocity
            agvalues.append(df.loc[2:]['Velocity (mph)'].max()) #maximum velocity, excluded hyperspace jumps
            agvalues.append(statistics.stdev(df.loc[1:]['Velocity (mph)'])) # standard deviation of velocity
            agvalues.append(df.loc[2:]['Absolute Acceleration'].mean()) # Average Acceleration of absolute value
            agvalues.append(df.loc[2:]['Absolute Acceleration'].max()) # Maximum value of positive and negative acceleration
            agvalues.append(statistics.stdev(df.loc[1:]['Acceleration (mph per s)'])) # Accleration standard deviation
            displace = dotproduct(df2.loc[(len(df2)-1)]['x'],df2.loc[(len(df2)-1)]['y'])  # Total displacement of the trip calculation, different from total distance traveled, converted to miles
            agvalues.append(displace) # adding displacement to the csv
            totaldist = (df.loc[2:]['Increment Traveled (feet)'].sum())  #converting feet to miles
            agvalues.append(totaldist) # Total distance traveled in miles
            agvalues.append(df.loc[2:]['Change in Direction per s'].max()) # maxiumum change in direction
            agvalues.append(statistics.stdev(df.loc[2:]['Direction (deg)'])) # Standard deviation of driving direction changes 
            agvalues.append(df.loc[1:]['Time (s)'].max()) # total driving time
            driver = df.iloc[1]['driver_id']
            trip= df.iloc[1]['trip_id']
            agvalues.append(turn.TurnCount(driver,trip)) # turns
            agvalues.append(AggressiveTurn.AggressiveTurn(driver,trip))  # aggresive turns
            agvalues.append(stop.StopCount(driver,trip)) # counting stops
            agvalues.append(bigdeceleration.BigBraking(driver,trip)) # counting big braking events
            agvalues.append(deceleration.Braking(driver,trip)) # counting braking events over 3 second windows
            agvalues.append(maxdecel.MaxDecel(driver,trip)) # return the maximum deceleration

            
            writer.writerow(agvalues)
                    
            agvalues = []
                       

###############################################################################
# 'Main' Function
############################################################################### 

if __name__ == '__main__':
    for dirName, subdirList, fileList in walk(INPUT_DIR):
        for fileName in fileList:
            if fileName.endswith('.csv'):
                '''
                for dirName, subdirList, fileList in walk(INPUT_DIR):
                    if basename(dirName).isdigit():                         # directory likely represents a driver
                        for fileName in fileList:
                            if fileName.endswith('.csv') :       # trip data is in .csv files and a numeric file name is likely a trip file, and getFileBaseName(fileName).isdigit()
                '''
                createFile(dirName, fileName)
    os.system('python lin.py')