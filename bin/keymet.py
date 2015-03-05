import math
import unicodecsv

with open('/Users/linwoodc3/Google Drive/Python/projects/test.csv', 'rU') as infile:
    infile.next() #skip first line with headings
    last_x, last_y = 0.0, 0.0
    seconds = 0
    distance = 0
    max_velocity = 0
    for l in infile:
        
        x, y = l.split(',')
        x, y = float(x), float(y)

        # calculate displacement
        increment_traveled = math.sqrt( (x - last_x)**2 + (y - last_y)**2 ) 
        
        # increment total distance by displacement
        distance += increment_traveled

        if increment_traveled > max_velocity:
            max_velocity = increment_traveled # we have a new record!

        print "seconds: %d, incremement: %f" % (seconds, increment_traveled)
        seconds += 1
        last_x, last_y = x, y

    print "seconds: %d, distance: %f, avg. velocity(units/s): %f, max velocity(units/s): %f" % (seconds, distance, distance/seconds,  max_velocity)