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


###############################################################################
# Helper Functions
###############################################################################



def getDisplacement(x,last_x,y,last_y):
    ### calculate displacement ###
    increment_traveled = math.sqrt( (x - last_x)**2 + (y - last_y)**2 )
    return increment_traveled


def gettotaldistance(distance,increment_traveled):
    ### increment total distance by displacement ###
    distance += increment_traveled
    return distance


def xyvelocity(x,y,last_x,last_y):
    # Calculate the average horizontal and vertical component of velocity; we ignore time because it's always equal to 1 second
    x_avg_vel = abs(abs(x) - abs(last_x))
    y_avg_vel = abs(abs(y) - abs(last_y))
    return "Horizontal Velocity: %f meters per second\nVertical Velocity: %f meters per second" % (x_avg_vel, y_avg_vel)


def xyacceleration(x_avg_vel,last_x_avg_vel,y_avg_vel,last_y_avg_vel):
    ### Calculate the horizontal and vertical acceleration components; this will be critical to determining turns, braking, etc. ###
    x_avg_acl = x_avg_vel - last_x_avg_vel
    y_avg_acl = y_avg_vel - last_y_avg_vel
    return "Horizontal direction accel: %f meters per sec^2\nVertical direction accel: %f meters per sec^2" % (x_avg_acl, y_avg_acl)


def getDirection(y,x):
     ### Calculate the direction ###
    direction_rad = math.atan2(y,x)
    direction = math.degrees(direction_rad)
    return "Driver heading: %f degrees" % (direction)

    # Calculate cardinal direction
    if direction == 0:
        return 'East'
    elif direction > 0 and direction < 90:
        return 'Northeast'
    elif direction == 90:
        return 'North'
    elif direction > 90 and direction< 180:
        return  'Northwest'
    elif direction == 180:
        return  'West'
    elif direction > 180 and direction < 270:
        return 'Southwest'
    elif direction == 270:
        return  'South'
    elif direction > 270 and direction < 360:
        return  'Southeast'

###############################################################################
# Main Functions
###############################################################################

#with open('/Users/linwoodc3/Google Drive/Python/projects/test.csv', 'rU') as infile:
with open('/home/captainobvious/Downloads/110.csv', 'rU') as infile:
    reader = csv.DictReader(infile, fieldnames=dimensions, delimiter=',', quotechar='"')
        
        with open('/home/captainobvious/Downloads/110.csv', 'wb') as outfile:
            writer = csv.writer(outfile)
            
            for idx, row in enumerate(reader):
                if idx == 0:
                    writer.writerow(['x', 'y', 'X Velocity', 'Y Velocity', ])

    infile.next() #skip first line with headings
    last_x, last_y, = 0.0, 0.0
    seconds = 0 
    distance = 0
    stops = 0
    braking_event = 0
    last_x_avg_vel= 0
    last_y_avg_vel= 0
    last_x_avg_acl = 0
    last_y_avg_acl=0
    max_velocity = 0
    accelerations = 0
    decelerations = 0

    for l in infile:
        
        x, y = l.split(',')
        x, y = float(x), float(y)
        

        
        
        # increment total distance by displacement
        distance += increment_traveled

        """
        We are dealing with a positon vector; therefore, we must calculate the x and y components of the position vector (i.e. car's position at point of time) of the car
        Velocity is measured by dividing the change in position by the change in time.
        The magnitude of the velocity vector can be found using the scalar dot product.
        Acceleratioin is measured by dividing the change in velocity by the change in time

        """

        # Calculate the average x and y component of velocity; we ignore time because it's always equal to 1 second
        x_avg_vel = abs(abs(x) - abs(last_x))

        y_avg_vel = abs(abs(y) - abs(last_y))

        # Calculate the average x and y direction accelerations; this will be critical to determining turns, braking, etc.

        x_avg_acl = x_avg_vel - last_x_avg_vel

        y_avg_acl = y_avg_vel - last_y_avg_vel

    
       

        
            

        


        if increment_traveled > max_velocity:
            max_velocity = increment_traveled # we have a new record!

        print "\nseconds: %d incremement: %f \nx velocity: %f, y velocity: %f \nx accel: %f,y accel: %f" % (seconds, increment_traveled, x_avg_vel, y_avg_vel, x_avg_acl, y_avg_acl)
        
        # counting braking or deceleration events

        if x_avg_acl < 0 and y_avg_acl < 0:
            decelerations += 1
            print "\nbraking event at %d" %(seconds)

        if last_x_avg_vel == 0 and y_avg_vel == 0:
            stops += 1
            print "Stop event count is %d" %(stops)

        

        seconds += 1

        last_x, last_y = x, y
        last_x_avg_vel, last_y_avg_vel = x_avg_vel, y_avg_vel
        
        

    print "seconds: %d \ndistance: %f\navg. velocity(units/s): %f\nmax velocity(units/s): %f\ndecelerations: %f\nstops: %f\n" % (seconds, distance, distance/seconds, max_velocity, decelerations, stops)

plt.ion()

#df = pd.read_csv('//Users/linwoodc3/Google Drive/Python/projects/test.csv')
df = pd.read_csv('/home/captainobvious/Downloads/110.csv')
df.plot(kind = 'scatter', x='x', y='y')
print len(df)
plt.pause(5)