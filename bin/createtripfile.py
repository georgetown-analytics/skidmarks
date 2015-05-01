# -*- coding: utf-8 -*-

###############################################################################
# Information
###############################################################################
# Created by Linwood Creekmore 
# Input from Vikram Mittal

# In partial fulfillment of the requirements for the Georgetown University Data Analytics Graduate Certificate Program

# May 1, 2015

# https://plus.google.com/+LinwoodCreekmoreIII/

###############################################################################
# Imports
###############################################################################


###############################################################################
# Main Functions
###############################################################################


def createFile(dirName, fileName):
    driver = getDriver(dirName)         # obtain driver id from directory name
    #df2 = pd.read_csv(os.path.join(OUTPUT_DIR,fileName))

    trip = fileName.split('_')[1]

    # This will open the driver directory/folder in a directory with all the driver files.  This should go through all 200 trip files for one driver.
    
    with open(os.path.join(os.path.normpath(path), dirName, fileName), 'rU') as infile:
        reader = csv.DictReader(infile, delimiter=',', quotechar='"')

        # This creates the output csv file that will hold all the calculated metrics

        with open(os.path.join(OUTPUT_DIR,fileName), 'wb') as outfile:
            writer = csv.writer(outfile)

            

            # This writes the header row for our output file using trip/driver IDs from Vik's IDify.py file
            
            for idx, row in enumerate(reader):
                if idx == 0:
                    trip_id = 1

                    writer.writerow(['driver_id', 'trip_id', 'Velocity (mph)', 'Acceleration (mph per s)','Absolute Acceleration','Time (s)', 'Increment Traveled (feet)','Change in Direction per s','Direction (deg)','Direction(card)'])

                    if not row == { 'driver_id':'driver_id', 'trip_id':'trip_id', 'x':'x', 'y':'y'}:

                        logger.warning("Missing or invalid header for driver %s and trip %s" % ("driver", trip_id))

                    else:

                        if len(row) != 4:

                            logger.warning("Too few/many values in row %s for driver %s and trip %s" % (idx, driver, trip_id))
            

                    # skip first line with headings.

                    infile.next() 

                    # Here we establish all the zero values for the key metrics we iterate.  These are defined in the key metrics documentation.

                    last_x, last_y, = 0.0, 0.0
                    seconds = 0 
                    distance = 0
                    stops = 0
                    braking_event = 0
                    last_x_avg_vel= 0
                    last_y_avg_vel= 0
                    max_velocity = 0
                    accelerations = 0
                    decelerations = 0
                    x_avg_vel = 0
                    y_avg_vel = 0
                    x_avg_acl = 0
                    y_avg_acl = 0
                    last_heading = 0.0
                    cur_heading = 0.0
                    lastvel = 0
                    lastaccel = 0
                    absoluteaccel = 0


                    # Creating an list to append all the calculated key metric values
                    metrics = []

                    # Establish the trip id value to iterate over for our database
                    #trip_id = 1

                    # We loop through the row values of x,y and calculate the key metric values, and then append the value to the metrics list above.

                    for l in infile:
                    
                        driver_id, trip_id, x, y, = l.split(',') 
                        driver_id, trip_id, x, y = int(driver_id), int(trip_id), float(x), float(y)
                        x_avg_vel,y_avg_vel = getVelocity(x,y,last_x,last_y)


                        metrics.append(fileName.split('_')[0]) #append driver #
                        metrics.append(trip_id) #append trip id
                        velocity = tomph(dotproduct(x_avg_vel, y_avg_vel))
                        if velocity > 200:
                            velocity = lastvel

                        metrics.append(velocity)

                        holdingvel = velocity - lastvel
                        absoluteaccel = velocity - lastvel
                        if abs(holdingvel) > 200:
                            holdingvel = lastaccel
                        

                        metrics.append(holdingvel) #acceleration
                        if absoluteaccel < 0:
                            absoluteaccel = abs(absoluteaccel)

                        metrics.append(absoluteaccel) #absolute acceleration
                        metrics.append(seconds) #time
                        metrics.append(getIncrement(x,last_x,y,last_y)) #distance traveled

                        last_heading = cur_heading
                        cur_heading = heading(y,x, last_y, last_x)
                        holding = abs(cur_heading - last_heading)
                        if holding >= 45 and getIncrement(x,last_x,y,last_y) <= 2:
                            holding = 0
                        elif holding >= 180:
                            holding = 360 - holding

                        metrics.append(holding) # enters change in direction
                        metrics.append(heading(y,x, last_y, last_x)) # calculates the heading in degrees
                        metrics.append(getCardinalDirection(heading(y,x,last_y,last_x))) # calculates cardinal position

                        # We write the identifying and key metrics values to our csv

                        writer.writerow(metrics)

                        # Next, we clear our metrics list for the next loop, iterate our time and trip ID values, and store the previous locations and velocity.

                        metrics = []
                        seconds += 1
                        trip_id += 1
                        last_x, last_y = x, y
                        lastvel = velocity
                        holdingvel = lastaccel
                        last_x_avg_vel, last_y_avg_vel = x_avg_vel, y_avg_vel
                        direction = last_heading
