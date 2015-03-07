#!/usr/bin/python

#Modified by Linwood Creekmore
#Sourced from psycopg2 documentation

###############################################################################
# Imports
###############################################################################

import psycopg2
import sys
 
###############################################################################
# Main Functions
###############################################################################

def main():

	#Enter username and password
	username = raw_input("AWS username:")
	password = raw_input("password:")


	#Define our connection string
	conn_string = "skidmarks.cssaygjswuzm.us-west-2.rds.amazonaws.com' dbname='skidmarks' user=%s password=%s port= 5432" % (username, password)
 
	# print the connection string we will use to connect
	print "Connecting to database\n	->%s" % (conn_string)
 
	# get a connection, if a connect cannot be made an exception will be raised here
	conn = psycopg2.connect(conn_string)
 
	# conn.cursor will return a cursor object, you can use this cursor to perform queries
	cursor = conn.cursor()
	print "Connected!\n"

 
if __name__ == "__main__":
	main()