# https://edabit.com/challenge/Mb8KmicGqpP3zDcQ5
# The Josephus Problem
# This classic problem dates back to Roman times. There are 41 soldiers arranged in a circle. Every third soldier is to be killed by their captors, continuing around the circle until only one soldier remains. He is to be freed. Assuming you would like to stay alive, at what position in the circle would you stand?
# 
# Generalize this problem by creating a function that accepts the number of soldiers n and the interval at which they are killed i, and returns the position of the fortunate survivor.
# 
# Examples
#   josephus(41, 3) ➞ 31
#   josephus(35, 11) ➞ 18
#   josephus(11, 1) ➞ 11
#   josephus(2, 2) ➞ 1
# Notes
# Assume the positions are numbered 1 to n going clockwise around the circle.
# If the interval is 3, the first soldiers to die are at positions 3, 6, and 9.


def josephus(n, i):
    soldiers = list([1]*n)
    countKill = 1
    countSoldier = 0
    while sum(soldiers) > 1:
        if soldiers[countSoldier] == 1:
            if countKill == i:
                soldiers[countSoldier] = 0
                countKill = 0
            countKill += 1
        if countSoldier == (n-1):
            countSoldier = 0
        else:
            countSoldier += 1
    
    countSoldier = 1
    for x in soldiers:
        if x==1:
            return countSoldier
        countSoldier += 1

if __name__ == "__main__":
    print(josephus(41, 3))