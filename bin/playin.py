import csv
import pandas as pd 



df = pd.read_csv('./output/trip/1_1.csv')

'''
#This is the pandas library code to index and return the values of a column; the first number is rows, second is columns.
The ':' is used to represent "through these values".  For example, 1:10 symbolizes numbers 1 through values 10.  
'''

df.loc[1:]['Direction (deg)']



# Code to get columnar aggregate values
df2.loc[1:]['Velocity (mph)'].describe()

from itertools import islice
numbers = iter(range(10))
for i in numbers:
    print i 
    next(islice(numbers, 3, 3), None)


from itertools import islice
numbers = df.loc[1:100]['Change in Direction per s']
windows = []

for i in numbers:
    print i 
    window = next(islice(numbers, 3, 7), None)
    windows.append(window)

# takes a sliced up list of numbers from a window and selects the fourth index (i.e [3]).  
print list(islice(numbers,start,None,1))[3]


# Rolling windows to calculate turns; need to use negative turns for 
numbers = df.loc[700:750]['Change in Direction per s']
print pd.rolling_sum(numbers, window = 3)



# takes a sstandard deviation of a slice of degrees.
statistics.stdev(df.loc[10:15]['Direction (deg)'])


for s in list(islice(numbers,0,None,3)):
	print s
  


crazy = pd.rolling_sum(numbers, windows = 3)
for l in crazy:

    if l >= 45:

        print "turn"
    else: 
        print "nothing"


