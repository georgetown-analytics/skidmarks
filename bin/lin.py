import os
import csv

path = os.path.abspath(os.getcwd())
rootdir = os.path.normpath(os.path.join(os.path.dirname(path),os.path.basename(path),"output","aggregate"))

agvalues = []

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
    	if file.endswith('.csv'):
            with open(os.path.join(subdir, file),'rb') as infile:
                fieldnames = ['driver_id', 'trip_id', 'Average Velocity (mph)', 'Max Velocity', 'Velocity Stdev','Average Acceleration (mph per s)', 'Max Acceleration (mph per s)', ' Acceleration Stdev','Max Direction Change per sec', ' Direction Stdev','Time (s)']
                reader = csv.reader(infile)
                reader.next()
                for row in reader:
                    agvalues.append(row)



with open(os.path.normpath(os.path.join(os.path.dirname(path),os.path.basename(path),"lin.csv")),'wb') as outfile:
	writer = csv.writer(outfile, delimiter=',', quotechar='"', lineterminator = '\n')				
	writer.writerow(['driver_id', 'trip_id', 'Average Velocity (mph)', 'Max Velocity', 'Velocity Stdev','Average Acceleration (mph per s)', 'Max Acceleration (mph per s)', ' Acceleration Stdev','Max Direction Change per sec', ' Direction Stdev','Time (s)'])
    		
	for l in agvalues:
		writer.writerow(l)
		





    

