
# -*- coding: utf-8 -*-

###############################################################################
# Information
###############################################################################
# Created by Linwood Creekmore 


# In partial fulfillment of the requirements for the Georgetown University Data Analytics Graduate Certificate Program

# April 19, 2015

# https://plus.google.com/+LinwoodCreekmoreIII/

###############################################################################
# Imports
###############################################################################

import pandas as pd
import os


###############################################################################
# File Paths
###############################################################################

path = os.path.abspath(os.getcwd())

###############################################################################
# Helper Functions
###############################################################################

###############################################################################
# Main Functions
###############################################################################

driver = str(raw_input("Enter a driver number: \n "))
# get the change in direction  from the file

# get the sum of the change in 3 seconds in a column

df = pd.read_csv(os.path.join(path,"output","trip", "1_" + driver + ".csv"))

numbers = df.loc[1:][['Change in Direction per s', 'Velocity (mph)']]


val = pd.rolling_sum(numbers, window = 5)
print len(val)
print val[20:35]
print "Seconds where high maneuvers are going on  \n", val.loc[val['Change in Direction per s'] >= 60].index



'''
for l in val:
	if l >= 45:
		print l
	else: 
		"nothing"

** Calculating turns and stops: Use the rolling window.  Divide "Change in Direction per s" by "Velocity (mph)" or vice versa to get a figure.
These are windows of 5 seconds, so values that are closer to 1 indicate a driver that conducts turning-like maneuvers at high speeds.  We will 
count the numbers of times a driver is in a high speed, high turn state. 
'''