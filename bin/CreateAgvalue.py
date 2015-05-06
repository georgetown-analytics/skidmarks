


###############################################################################
# Main Functions
###############################################################################


with open(os.path.join(OUTPUT_DIR, fileName), 'rU') as infile:
        reader = csv.DictReader(infile, delimiter=',', quotechar='"')

        with open(os.path.join(OUTPUT_DIR2,fileName), 'wb') as outfile:
            writer = csv.writer(outfile, lineterminator = '\n')

            df = pd.read_csv(os.path.join(OUTPUT_DIR,fileName))
            writer.writerow(['driver_id', 'trip_id', 'Average Velocity (mph)', 'Max Velocity', 'Velocity Stdev','Average Acceleration (mph per s)', 'Max Acceleration (mph per s)', ' Acceleration Stdev','Displacement','Total Distance Traveled','Max Direction Change per sec', ' Direction Stdev','Time (s)', 'Turns', 'Aggressive Turns', 'Stops', 'Large Deceleration Events', 'Deceleration Events', 'Max Deceleration Event'])
            df2 = pd.read_csv(os.path.join(INPUT_DIR,str(driver),fileName))
            
            print "Driver is", driver
            print "Trip is", trip
            agvalues = []        

            agvalues.append(fileName.split('_')[0])
            agvalues.append(df.loc[1][1])
            agvalues.append(df.loc[2:]['Velocity (mph)'].mean()) #Average velocity
            agvalues.append(df.loc[2:]['Velocity (mph)'].max()) #maximum velocity, excluded hyperspace jumps
            agvalues.append(statistics.stdev(df.loc[1:]['Velocity (mph)'])) # standard deviation of velocity
            agvalues.append(df.loc[2:]['Absolute Acceleration'].mean()) # Average Acceleration of absolute value
            agvalues.append(df.loc[2:]['Absolute Acceleration'].max()) # Maximum value of positive and negative acceleration
            agvalues.append(statistics.stdev(df.loc[1:]['Acceleration (mph per s)'])) # Accleration standard deviation
            displace = dotproduct(df2.loc[(len(df2)-1)]['x'],df2.loc[(len(df2)-1)]['y'])  # Total displacement of the trip calculation, different from total distance traveled, converted to miles
            agvalues.append(displace) # adding displacement to the csv
            totaldist = (df.loc[2:]['Increment Traveled (feet)'].sum())  #converting feet to miles
            agvalues.append(totaldist) # Total distance traveled in miles
            agvalues.append(df.loc[2:]['Change in Direction per s'].max()) # maxiumum change in direction
            agvalues.append(statistics.stdev(df.loc[2:]['Direction (deg)'])) # Standard deviation of driving direction changes 
            agvalues.append(df.loc[1:]['Time (s)'].max()) # total driving time
            driver = df.iloc[1]['driver_id']
            trip= df.iloc[1]['trip_id']
            agvalues.append(turn.TurnCount(driver,trip)) # turns
            agvalues.append(AggressiveTurn.AggressiveTurn(driver,trip))  # aggresive turns
            agvalues.append(stop.StopCount(driver,trip)) # counting stops
            agvalues.append(bigdeceleration.BigBraking(driver,trip)) # counting big braking events
            agvalues.append(deceleration.Braking(driver,trip)) # counting braking events over 3 second windows
            agvalues.append(maxdecel.MaxDecel(driver,trip)) # return the maximum deceleration

            
            writer.writerow(agvalues)
                    
            agvalues = []