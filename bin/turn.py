
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

driver = str(raw_input("Enter a driver number: \n >"))
# get the change in direction  from the file

# get the sum of the change in 3 seconds in a column

df = pd.read_csv(os.path.join(path,"output","trip", "1_" + driver + ".csv"))

df['start'] = 0
df['end'] = 0

numbers = df.loc[1:][['Change in Direction per s', 'Velocity (mph)']]


val = pd.rolling_sum(numbers, window = 5)
print len(val)
print val[720:750]

turns = val.loc[val['Change in Direction per s'] >= 60].index

if len(turns) < 1:
	print "No measurable maneuvers"

else: 

	print "Seconds where high maneuvers are going on  \n", turns

	#iterate through turns and flag start and end of turns in the dataframe

	df['start'][turns[0]] = 1	#the first index must be the start of the first turn

	for i in range(1, len(turns)-1):
		#print i
		if turns[i] - 1 > turns[i-1]:
			df['start'][turns[i]] = 1	#the current index and last index are more than 1 second apart so this must be the start of a new turn
			df['end'][turns[i-1]] = 1	#the last index must also be the end of the last turn since we are beginning a new turn

	df['end'][turns[len(turns)-1]] = 1	#the last index must be the end of the last turn


	for index, row in df.iterrows():
		if row['start'] == 1 or row['end'] == 1:
			print index, row



'''
Calculating turns and stops: Use the rolling window.  Divide "Change in Direction per s" by "Velocity (mph)" or vice versa to get a figure.
These are windows of 5 seconds, so values that are closer to 1 indicate a driver that conducts turning-like maneuvers at high speeds.  We will 
count the numbers of times a driver is in a high speed, high turn state. 
'''