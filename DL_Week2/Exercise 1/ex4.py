# Ex4: print all Possible Combinations from the three Digits
# data4 = [1, 2, 3]
data4 = [1, 2, 3]
for i in range (0, len(data4)):
    for j in range(0, len(data4)):
        for k in range(0, len(data4)):
            if((i != j) & (j != k) & (i != k)):
                print(data4[i],data4[j],data4[k])