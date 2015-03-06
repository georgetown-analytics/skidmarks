import math
import unicodecsv

with open('/Users/linwoodc3/Google Drive/Python/projects/test.csv', 'rU') as infile:
    infile.next() #skip first line with headings
    last_x, last_y, = 0.0, 0.0
    seconds = 0
    distance = 0
    stops = 0
    last_v = 0
    last_a = 0
    max_velocity = 0
    for l in infile:
        
        x, y = l.split(',')
        x, y = float(x), float(y)
        print x, y, last_x,last_y

        # calculate displacement
        increment_traveled = math.sqrt( (x - last_x)**2 + (y - last_y)**2 ) 
        
        # increment total distance by displacement
        distance += increment_traveled

        """
        We are dealing with a positon vector; therefore, we must calculate the x and y components of the position vector (i.e. car's position at point of time) of the car
        Velocity is measured by dividing the change in position by the change in time
        Acceleratioin is measured by dividing the change in velocity by the change in time
        """

        # slope equals instantaneous velocity at a point in time; 
        xvol = 
        # counting stops
        if acceleration == 0 and x > 0 and y > 0:
            stop += 1

        # counting braking or deceleration events


        if increment_traveled > max_velocity:
            max_velocity = increment_traveled # we have a new record!

        print "seconds: %d, incremement: %f" % (seconds, increment_traveled)
        seconds += 1

        last_x, last_y= x, y
        velocity = last_v
        

    print "seconds: %d, distance: %f, avg. velocity(units/s): %f,  max velocity(units/s): %f" % (seconds, distance, distance/seconds, max_velocity)