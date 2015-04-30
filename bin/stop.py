
# -*- coding: utf-8 -*-

###############################################################################
# Information
###############################################################################
# Created by Linwood Creekmore 
# Input by Vikkram Mittal


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

def StopCount(driver,trip):



	path = os.path.abspath(os.getcwd())
	#pathtocsv = os.path.normpath(os.path.join(path,"output","test",str(driver),str(driver)+"_"+str(trip)+".csv"))
	pathtocsv = os.path.normpath(os.path.join(path,"output","trip",str(driver)+"_"+str(trip)+".csv"))

	print pathtocsv

	df = pd.read_csv(pathtocsv)

	df['start'] = 0
	df['end'] = 0

	print df[250:280][['Velocity (mph)','Change in Direction per s']]

	numbers = df.loc[1:][['Velocity (mph)', 'Increment Traveled (feet)']] 


	val = pd.rolling_sum(numbers, window = 3)
	print len(val)
	print val[700:750]

	still = val.loc[val['Increment Traveled (feet)'] <= 3].index

	if len(still) < 1:
		print "No measured stops"

	else: 

		print "Seconds of consistent stops  \n", still

		#iterate through still and flag start and end of still in the dataframe

		stops = 0

		df['start'][still[0]] = 1	#the first index must be the start of the first turn

		for i in range(1, len(still)-1):
			#print i
			if still[i] - 1 > still[i-1]:
				df['start'][still[i]] = 1	#the current index and last index are more than 1 second apart so this must be the start of a new turn
				df['end'][still[i-1]] = 1	#the last index must also be the end of the last turn since we are beginning a new turn

		df['end'][still[len(still)-1]] = 1	#the last index must be the end of the last turn


		for index, row in df.iterrows():
			if row['start'] == 1 or row['end'] == 1:
				print index, row['Velocity (mph)']
			if row['end'] == 1:
				stops += 1
		print "Driver # %s made %d stop(s) in this trip" % (driver,stops)

		return stops



	'''
	Calculating still and still: Use the rolling window.  Divide "Change in Direction per s" by "Velocity (mph)" or vice versa to get a figure.
	These are windows of 5 seconds, so values that are closer to 1 indicate a driver that conducts turning-like maneuvers at high speeds.  We will 
	count the numbers of times a driver is in a high speed, high turn state. 
	'''

###############################################################################
# 'Main' Function
############################################################################### 

if __name__ == '__main__':
	driver = raw_input('Pick a driver. Enter a number between 1-3612:\n')
	trip = raw_input('Pick a trip. Enter a number between 1-200:\n')
	StopCount(driver,trip)