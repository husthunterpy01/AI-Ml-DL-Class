# Ex3: find the strongest neighbour. Given an array of N positive integers.
# The task is to find the maximum for every adjacent pair in the array.
#data3 = [4, 5, 6, 7, 3, 9, 11, 2, 10]

data3 = [4, 5, 6, 7, 3, 9, 11, 2, 10]
arr = []
for i in range(1, len(data3)):
    a = max(data3[i],data3[i-1])
    arr.append(a)
for b in arr:
    print(b,end=" ")