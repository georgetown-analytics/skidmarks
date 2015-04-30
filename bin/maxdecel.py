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

def MaxDecel(driver,trip):


	path = os.path.abspath(os.getcwd())
	#pathtocsv = os.path.normpath(os.path.join(path,"output","test",str(driver),str(driver)+"_"+str(trip)+".csv"))
	pathtocsv = os.path.normpath(os.path.join(path,"output","trip",str(driver)+"_"+str(trip)+".csv"))

	print pathtocsv

	df = pd.read_csv(pathtocsv)

	

	return df['Acceleration (mph per s)'].min()
	 
	

	

###############################################################################
# 'Main' Function
############################################################################### 

if __name__ == '__main__':
	driver = raw_input('Pick a driver. Enter a number between 1-3612:\n')
	trip = raw_input('Pick a trip. Enter a number between 1-200:\n')
	MaxDecel(driver,trip)