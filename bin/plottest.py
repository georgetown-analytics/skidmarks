
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


plt.ion()

target = raw_input('Enter a number between 1-200:\n')
path = os.path.abspath(os.getcwd())
pathtocsv = os.path.normpath(os.path.join(path,"input","test","1","1"+"_"+target+".csv"))

df = pd.read_csv(pathtocsv)

#df.plot(kind = 'scatter', x='x', y='y')
#plt.pause(15)
plt.scatter(df["x"], df['y'])
plt.show()
raw_input('hit enter to quit')
plt.pause(15)

def main():
    csv = raw_input('Enter a word:')
    new_word = convert_to_pig_latin(original)

###############################################################################
# 'Main' Function
############################################################################### 
if __name__ == '__main__':
	main()