# -*- coding: utf-8 -*-

###############################################################################
# Information
###############################################################################
# Created by Linwood Creekmore 
# Input from Vikram Mittal

# In partial fulfillment of the requirements for the Georgetown University Data Analytics Graduate Certificate Program

# May 1, 2015

# https://plus.google.com/+LinwoodCreekmoreIII/

###############################################################################
# Imports
###############################################################################

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

