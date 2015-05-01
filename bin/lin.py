
# -*- coding: utf-8 -*-

###############################################################################
# Information
###############################################################################
# Created by Linwood Creekmore 
# Input by Vikram Mittal


# In partial fulfillment of the requirements for the Georgetown University Data Analytics Graduate Certificate Program

# April 19, 2015

# https://plus.google.com/+LinwoodCreekmoreIII/

###############################################################################
# Imports
###############################################################################

import os
import csv

###############################################################################
# Main Functions
###############################################################################

#def AgTables(driver,trip)

path = os.path.abspath(os.getcwd())
rootdir = os.path.normpath(os.path.join(os.path.dirname(path),os.path.basename(path),"output","aggregate"))

agvalues = []

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
    	if file.endswith('.csv'):
            with open(os.path.join(subdir, file),'rb') as infile:
                fieldnames = ['driver_id', 'trip_id', 'Average Velocity (mph)', 'Max Velocity', 'Velocity Stdev','Average Acceleration (mph per s)', 'Max Acceleration (mph per s)', ' Acceleration Stdev','Displacement','Total Distance Traveled','Max Direction Change per sec', ' Direction Stdev','Time (s)', 'Turns', 'Aggressive Turns', 'Stops', 'Large Deceleration Events', 'Deceleration Events', 'Max Deceleration Event']
                reader = csv.reader(infile)
                reader.next()
                for row in reader:
                    agvalues.append(row)



with open(os.path.normpath(os.path.join(os.path.dirname(path),os.path.basename(path),"lin.csv")),'wb') as outfile:
	writer = csv.writer(outfile, delimiter=',', quotechar='"', lineterminator = '\n')				
	writer.writerow(['driver_id', 'trip_id', 'Average Velocity (mph)', 'Max Velocity', 'Velocity Stdev','Average Acceleration (mph per s)', 'Max Acceleration (mph per s)', ' Acceleration Stdev','Displacement','Total Distance Traveled','Max Direction Change per sec', ' Direction Stdev','Time (s)', 'Turns', 'Aggressive Turns', 'Stops', 'Large Deceleration Events', 'Deceleration Events', 'Max Deceleration Event'])
    		
	for l in agvalues:
		writer.writerow(l)
		

###############################################################################
# 'Main' Function
############################################################################### 
'''
if __name__ == '__main__':
    driver = raw_input('Pick a driver. Enter a number between 1-3612:\n')
    trip = raw_input('Pick a trip. Enter a number between 1-200:\n')
    AgTables(driver,trip)
'''
    

