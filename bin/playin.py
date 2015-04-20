for s in list(islice(numbers,0,None,3)):
	print s
  


crazy = pd.rolling_sum(numbers, windows = 3)
for l in crazy:

    if l >= 45:

        print "turn"
    else: 
        print "nothing"


