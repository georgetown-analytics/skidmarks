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

        # Calculate the average x component of velocity; we ignore time because it's always equal to 1 second
        x_avg_vel = abs(abs(x) - abs(last_x))

        # Calculate the average y compoent of velocity; we ignore time because it's always equal to 1 second
        y_avg_vel = abs(abs(y) - abs(last_y))

        # Calculating the magnitude of the velocity vector with x and y components 
        v_xy= math.sqrt((x_avg_vel**2) + (y_avg_vel**2))

        # Calculate average acceleration

    

        # counting braking or deceleration events


        if increment_traveled > max_velocity:
            max_velocity = increment_traveled # we have a new record!

        print "seconds: %d, incremement: %f, x velocity: %f, y velocity: %f, magnitude of velocity: %f " % (seconds, increment_traveled, x_avg_vel, y_avg_vel, v_xy)
        seconds += 1

        last_x, last_y= x, y
        
        

    print "seconds: %d, distance: %f, avg. velocity(units/s): %f,  max velocity(units/s): %f" % (seconds, distance, distance/seconds, max_velocity)