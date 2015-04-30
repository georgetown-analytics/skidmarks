# -*- coding: utf-8 -*-

###############################################################################
# Information
###############################################################################
# Created by Linwood Creekmore 


# Skeleton from http://stackoverflow.com/questions/20011494/plot-normal-distribution-with-matplotlib


# In partial fulfillment of the requirements for the Georgetown University Data Analytics Graduate Certificate Program

# April 29, 2015

# https://plus.google.com/+LinwoodCreekmoreIII/

###############################################################################
# Imports
###############################################################################


import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

def DisPlt(driver,trip):

	path = os.path.abspath(os.getcwd())
	pathtocsv = os.path.normpath(os.path.join(path,"output","trip",str(driver)+"_"+str(trip)+".csv"))
	df = pd.read_csv(pathtocsv)

	investigation = str(raw_input("Enter a variable \n>"))

	h = sorted([df[investigation]]) #sorted




	fit = stats.norm.pdf(h, np.mean(h), np.std(h))  #this is a fitting indeed

	plt.plot(h,fit,'-o')

	plt.hist(h,normed=True)      #use this to draw histogram of your data

	plt.show()                   #use may also need add this

###############################################################################
# 'Main' Function
############################################################################### 

if __name__ == '__main__':
	driver = raw_input('Pick a driver. Enter a number between 1-3612:\n')
	trip = raw_input('Pick a trip. Enter a number between 1-200:\n')
	DisPlt(driver,trip)