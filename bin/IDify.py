
###############################################################################
# Imports
###############################################################################

from os import walk
from os.path import basename
import unicodecsv as csv
import logging as log


###############################################################################
# Constants
###############################################################################

DIR_CHAR = '\\'


###############################################################################
# File Paths
###############################################################################

LOG_DIR    = 'C:\\Users\\Vikram\\Documents\\Georgetown\\Projects\\MyProjects\\SkidMarks\\logs'
INPUT_DIR  = 'C:\\Users\\Vikram\\Documents\\Georgetown\\Projects\\MyProjects\\SkidMarks\\input\\test'
OUTPUT_DIR = 'C:\\Users\\Vikram\\Documents\\Georgetown\\Projects\\MyProjects\\SkidMarks\\output\\trips'


###############################################################################
# Create Logger, File Handlers, Formatters, and Filters (as needed)
###############################################################################

class SingleLevelFilter(log.Filter):
    def __init__(self, passlevel, reject):
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        if self.reject:
            return (record.levelno != self.passlevel)
        else:
            return (record.levelno == self.passlevel)

logger = log.getLogger("DataQuality")
logger.setLevel(log.DEBUG)

infoHandler = log.FileHandler(LOG_DIR + DIR_CHAR + "Displacement_Summary.log", mode='w')
infoFilter = SingleLevelFilter(log.INFO, False)
infoHandler.addFilter(infoFilter)

warnHandler = log.FileHandler(LOG_DIR + DIR_CHAR + "Warnings.log", mode='w')
warnFilter = SingleLevelFilter(log.WARNING, False)
warnHandler.addFilter(warnFilter)

logger.addHandler(infoHandler)
logger.addHandler(warnHandler)


###############################################################################
# Helper Functions
###############################################################################

def getDriver(dirName):
	return int(basename(dirName))


def getFileBaseName(baseName):
	return fileName.split('.')[0]


def getTrip(baseName):
	return int(getFileBaseName(baseName))


def isFloat(str):
	try:
		float(str)
		return True
	except:
		return False


###############################################################################
# Main Functions
###############################################################################

def parseFile(dirName, fileName):
	driver = getDriver(dirName)			# obtain driver id from directory name
	trip = getTrip(fileName)			# obtain trip id from file name
	dimensions = ('x', 'y')
	displacement = {'x': 0, 'y': 0}
	
	with open(dirName + DIR_CHAR + fileName, 'rb') as infile:
		reader = csv.DictReader(infile, fieldnames=dimensions, delimiter=',', quotechar='"')
		
		with open(OUTPUT_DIR + DIR_CHAR + str(driver) + "_" + fileName, 'wb') as outfile:
			writer = csv.writer(outfile)
			
			for idx, row in enumerate(reader):
				if idx == 0:
					writer.writerow(['driver_id', 'trip_id', 'x', 'y'])
					
					if not row == {'x':'x', 'y':'y'}:
						logger.warning("Missing or invalid header for driver %s and trip %s" % (driver, trip))
				else:
					if len(row) != 2:
						logger.warning("Too few/many values in row %s for driver %s and trip %s" % (idx, driver, trip))
					else:
						for dim in dimensions:
							if isFloat(row[dim]):
								row[dim] = float(row[dim])
								displacement[dim] += row[dim]
							else:
								logger.warning("Invalid value in row %s for driver %s and trip %s" % (idx, driver, trip))
						
						writer.writerow([driver, trip, row['x'], row['y']])
			
			logger.info("%s\t%s\t%s\t%s" % (driver, trip, displacement['x'], displacement['y']))


###############################################################################
# 'Main' Function
###############################################################################	

if __name__ == '__main__':
	for dirName, subdirList, fileList in walk(INPUT_DIR):
		if basename(dirName).isdigit():							# directory likely represents a driver
			for fileName in fileList:
				if fileName.endswith('.csv') and getFileBaseName(fileName).isdigit():		# trip data is in .csv files and a numeric file name is likely a trip file
					parseFile(dirName, fileName)