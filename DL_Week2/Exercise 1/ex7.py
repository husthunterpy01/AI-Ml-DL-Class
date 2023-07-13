# Ex7: Write a program, which will find all such numbers between 1000 and 3000 (both included) such that each digit of the number is an even number.
# The numbers obtained should be printed in a comma-separated sequence on a single line.

result = []
for i in range (1000,3001):
    if all((int(re) % 2 == 0) for re in str(i)):
        result.append(str(i))

    print(",".join(result))