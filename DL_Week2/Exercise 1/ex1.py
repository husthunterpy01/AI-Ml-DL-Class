# Ex1: Write a program to count positive and negative numbers in a list
# data1 = [-10, -21, -4, -45, -66, 93, 11, -4, -6, 12, 11, 4]
data1 = [-10, -21, -4, -45, -66, 93, 11, -4, -6, 12, 11, 4]
pos_num, neg_num = 0, 0
for num in data1:
    if (num > 0):
        pos_num += 1
    if (num < 0):
        neg_num += 1

print ("The number of positive numbers is : ", pos_num)
print ("The number of negative numbers is : ",neg_num)
