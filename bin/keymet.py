import math
import unicodecsv

with open('/Users/linwoodc3/Google Drive/Python/projects/test.csv', 'rU') as infile:
    infile.next() #skip first line with headings
    last_x, last_y, = 0.0, 0.0
    seconds = 0
    distance = 0
    stops = 0
    last_x_avg_vel= 0
    last_y_avg_vel= 0
    last_x_avg_acl = 0
    last_y_avg_acl=0
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
        Velocity is measured by dividing the change in position by the change in time.
        The magnitude of the velocity vector can be found using the scalar dot product, which is equivalent to Pythagorean theorem.
        Acceleratioin is measured by dividing the change in velocity by the change in time

        """

        # Calculate the average x and y component of velocity; we ignore time because it's always equal to 1 second
        x_avg_vel = abs(abs(x) - abs(last_x))

        y_avg_vel = abs(abs(y) - abs(last_y))

        # Calculate the average x and y direction accelerations; this will be critical to determining turns, braking, etc.

        x_avg_acl = abs(x) - abs(last_x)

        y_avg_acl = abs(y) - abs(last_y)

    

        # counting braking or deceleration events

        if x_avg_acl - last_x_avg_acl < 0 :
            #x_dcl =


        if increment_traveled > max_velocity:
            max_velocity = increment_traveled # we have a new record!

        print "seconds: %d, incremement: %f, x velocity: %f, y velocity: %f, x accel: %f, y accel: %f" % (seconds, increment_traveled, x_avg_vel, y_avg_vel, x_avg_acl, y_avg_acl)
        seconds += 1

        last_x, last_y = x, y
        
        

    print "seconds: %d, distance: %f, avg. velocity(units/s): %f,  max velocity(units/s): %f" % (seconds, distance, distance/seconds, max_velocity)