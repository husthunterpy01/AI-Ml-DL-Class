# Ex2: Given a list, extract all elements whose frequency is greater than k.
#data2 = [4, 6, 4, 3, 3, 4, 3, 4, 3, 8]
#k = 3
data2 = [4, 6, 4, 3, 3, 4, 3, 4, 3, 8]
k = 3

a = [i for i in set(data2) if data2.count(i) > k]
print(a)