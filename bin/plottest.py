
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def PlotTrip(driver,trip):

	plt.ion()

	path = os.path.abspath(os.getcwd())
	pathtocsv = os.path.normpath(os.path.join(path,"input","test",str(driver),str(trip)+".csv"))
	#pathtocsv = os.path.normpath(os.path.join(path,"output","trip",str(driver)+"_"+str(trip)+".csv"))


	#pathtocsv = os.path.normpath(os.path.join(path,"input","test","1","1"+"_"+target+".csv"))

	df = pd.read_csv(pathtocsv)

	#df.plot(kind = 'scatter', x='x', y='y')
	#plt.pause(15)
	plt.scatter(df["x"], df['y'])
	plt.show()
	raw_input('hit enter to quit')
	plt.pause(15)



###############################################################################
# 'Main' Function
############################################################################### 
if __name__ == '__main__':
	driver = raw_input('Pick a driver. Enter a number between 1-3612:\n')
	trip = raw_input('Pick a trip. Enter a number between 1-200:\n')
	PlotTrip(driver,trip)