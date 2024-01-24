# Simplified Problem: 
# create solution(M, F) where M and F represent the number of Mach and Facula bombs required for success
# You start with 1 mach bomb and 1 facula bomb
# Each mach bomb can generate a facula bomb
# Each facula bomb can generate a mach bomb
# Determine the number of generations required to get from 1 Mach and 1 Facula to the required number
# See second point in the full word problem for details on generation
# It seems like you can switch between generating faculas or machs for each generation
# BUT you cant do both at the same time

# ANSWER MUST MATCH M AND F EXACTLY, cant have extra bombs, cant have too few bombs, of either type

# Known test cases:
# Solution(4, 7) = 4
# Solution(2, 1) = 1
# Solution(2, 4) = IMPOSSIBLE

# The tree can be traversed backwards from any
# point to the head (1,1) by doing the following:
#   1. Get the min and max of M and F,
#   2. Subtract min from max until value of max is less than min
#      - this will also indicate the number of steps taken
#   3. Min is now less than max, swap the variables and repeat
# Once the min value is less than or equal to 1, we are done
def solutionV11(M, F):
    M, F = int(M), int(F)
    hi, lo = max(M, F), min(M, F)
    count = 0

    while lo > 1:
        count += hi // lo
        hi = hi % lo
        hi, lo = lo, hi

    return str(count + hi - 1) if lo == 1 else 'impossible'
print(solutionV11(1,1))
print(solutionV11(1,2))
print(solutionV11(3,2))
print(solutionV11(3,5))
print(solutionV11(8,5))
print(solutionV11(2,4))
print(solutionV11(7,79))

def solutionV10(M, F):
    M, F = int(M), int(F)
    hi, lo = max(M, F), min(M, F)
    count = 0

    while lo > 0:
        if lo == 1:
            count += hi - 1
            break

        count += hi // lo
        hi = hi % lo
        hi, lo = lo, hi

    return str(count) if lo == 1 else 'impossible'
# print(solutionV10(1,1))
# print(solutionV10(1,2))
# print(solutionV10(3,2))
# print(solutionV10(3,5))
# print(solutionV10(8,5))
# print(solutionV10(2,4))
# print(solutionV10(7,79))


def solutionV9(M, F):
    # M = int(M)
    # F = int(F)
    M, F = int(M), int(F)
    count = 0
    while M > 0 < F:
        if M == 1 or F == 1:
            count += max(M, F) - 1
            break

        hi = max(M, F)
        lo = min(M, F)
        length = hi - (hi % lo)
        count += length // lo

        if hi == M:
            M -= length
        else:
            F -= length

    return str(count) if M > 0 < F else 'impossible'
# print(solutionV9(1,1))
# print(solutionV9(2,1))
# print(solutionV9(2,3))
# print(solutionV9(5,3))
# print(solutionV9(5,8))
# print(solutionV9(2,8))
# print(solutionV9(7,79))
# print('##########')

def solutionV8(M, F):
    count = 0
    while M > 0 < F:
        # print('iterating')
        # print(M, F)
        # if M == 1 and F == 1:
        #     break

        if M == 1 or F == 1:
            count += max(M, F)
            break

        if M > F:
            # greatest = M
            # least = F
            # legLength = greatest - greatest % least
            # stepCount = legLength / least
            # count += stepCount
            # M = greatest - legLength
            length = M - (M % F)
            count += length / F
            M = M - length
        else:
            greatest = F
            least = M
            legLength = greatest - greatest % least
            stepCount = legLength / least
            count += stepCount
            F = greatest - legLength

    # print('FINAL RESULT')
    # print(M, F)
    # print(count - 1)
    return count - 1 if M > 0 < F else 'IMPOSSIBLE'
# print(solutionV8(1, 1))
# print(solutionV8(2, 1))
# print(solutionV8(2, 3))
# print(solutionV8(2, 5))
# print(solutionV8(8, 3))
# print(solutionV8(7, 5))
# print(solutionV8(2, 4))
# print(solutionV8(10*50, 1000))

# Dad's solution
def solutionV7(M, F):
    currentM = M
    currentF = F
    count = 0
    # print(currentM, currentF)
    while currentM > 0 < currentF:
        if currentM == 1 and currentF == 1:
            break

        if currentM > currentF:
            currentM = currentM - currentF
        else:
            currentF = currentF - currentM
        count += 1

    result = count if currentM == 1 and currentF == 1 else 'IMPOSSIBLE'
    # print(result)
    return result

# solutionV7(32, 16)
# solutionV7(3, 5)
# solutionV7(10**30,1000000000000)

# This equation can only make one turn
def solutionV5(M, F):
    greatest = M if M > F else F
    least = M if M < F else F
    lastLeg = (greatest - 1) / least
    # print(firstLeg, lastLeg)
    # print(firstLeg + lastLeg - 1)
    # print(lastLeg.is_integer())
    if lastLeg.is_integer():
        return least + lastLeg - 1

    return 'IMPOSSIBLE'
# print(solutionV5(1, 5))
# print(solutionV5(4, 7))
# print(solutionV5(7, 3))
# print(solutionV5(2, 7))
# print(solutionV5(1, 4))
# print(solutionV5(2, 2))
# print(solutionV5(2, 4))
# print()


def solutionV4(M, F):
    if M < 3 and F < 3:
        return 0 if M == 1 and F == 1 else 1
    # curPoss = [[1, 2]]
    curPoss = [[1, 1]]
    # start = [1, 1]
    end = [M, F]
    end2 = [F, M]
    highest = M if M > F else F
    genCount = 0
    while not end in curPoss and not end2 in curPoss and curPoss:
        newPoss = []
        for poss in curPoss:
            newCount = poss[0] + poss[1]
            if (newCount <= highest):
                newPoss.append([newCount, poss[1]])
                newPoss.append([poss[0], newCount])
        curPoss = newPoss
        genCount += 1
    return genCount if end in curPoss else 'IMPOSSIBLE'
# print(solutionV4(1000,1000))
# print(solutionV4(2000,2000))
# print(solutionV4(3000,3000))
# print(solutionV4(4000,4000))
# print(solutionV4(5000,5000))
# print(solutionV4(1,1))
# print(solutionV4(1,2))
# print(solutionV4(4,8))
# print(solutionV4(4,5))
# # print(solutionV4(1,7))
# # print(solutionV4(2,4))
# print("##########")

class Bombs:
    def __init__(self, machCount, faculaCount, genCount):
        # self.bombCounts = {
        #     'machCount': machCount,
        #     'faculaCount': faculaCount,
        # }
        # self.bombCounts = BomCounts(machCount, faculaCount)
        self.machCount = machCount
        self.faculaCount = faculaCount
        self.genCount = genCount
# print(Bombs(7,9,1).bombCounts)
def solutionV3(M, F):
    # print(M, F)
    if M < 3 and F < 3:
        return 0 if M == 1 and F == 1 else 1
    queue = [ Bombs(1, 2, 1) ]
    i = 0
    # check = [ queue[0].machCount, queue[0].faculaCount ]
    # check = [ 0, 0 ] 
    check = [ M, F ] 
    while i < len(queue) and not (queue[i].machCount in check and queue[i].faculaCount in check):
        # print('Iterating')
        # machCount = queue[i].machCount
        # faculaCount = queue[i].faculaCount
        # newBombCount = machCount + faculaCount
        newBombCount = queue[i].machCount + queue[i].faculaCount
        newGenCount = queue[i].genCount + 1
        # print('mach count')
        # print(machCount)
        # print('facula count')
        # print(faculaCount)
        if newBombCount <= M:
            queue.append(Bombs(
                newBombCount,
                queue[i].faculaCount,
                newGenCount,
            ))
        if newBombCount <= F:
            queue.append(Bombs(
                queue[i].machCount,
                newBombCount,
                newGenCount,
            ))
        i += 1
        # check = [ queue[i].machCount, queue[i].faculaCount ]

    # print(len(queue))
    return queue[i].genCount if i < len(queue) else 'impossible'
# print(solutionV3(1,1))
# print(solutionV3(1,2))
# print(solutionV3(4,7))
# print(solutionV3(2,4))
# # If numbers are factors of each other, it will be impossible
# print(solutionV3(4,8))
# print(solutionV3(3,9))
# print(solutionV3(12,36))
# print(solutionV3(10,100))
# print(solutionV3(1,7))

# Builds the entire tree, checking to see if current node is the answer for each iteration
def solution(M, F):
    # print(M, F)
    queue = [ Bombs(1, 1, 0) ]
    i = 0
    while i < len(queue) and not (queue[i].machCount == M and queue[i].faculaCount == F):
        # print('Iterating')
        newBombCount = queue[i].machCount + queue[i].faculaCount
        newGenCount = queue[i].genCount + 1
        if newBombCount <= M:
            queue.append(Bombs(
                newBombCount,
                queue[i].faculaCount,
                newGenCount,
            ))
        if newBombCount <= F:
            queue.append(Bombs(
                queue[i].machCount,
                newBombCount,
                newGenCount,
            ))
        i += 1

    # print(len(queue))
    return queue[i].genCount if i < len(queue) else 'impossible'
# print(solution(1, 1))
# print(solution(2, 1))
# print(solution(4, 7))
# print(solution(2, 4))
# print(solution(10, 10))
# print(solution(100, 100))
# print(solution(1000, 1000))

import time
def test(func, min, max):
    start = time.time()
    mCount = min
    while mCount <= max:
        fCount = min
        while fCount <= max:
            # print(mCount, fCount)
            # print(func(mCount, fCount))
            func(mCount, fCount)
            fCount += 1
        mCount += 1
    end = time.time()
    print(end - start)
# test(solutionV5, 10**50, 10**50)
# test(solutionV4, 0, 500)
# test(solutionV7, 99000, 100000)
# test(solutionV9, 10**50 - 1000, 10**50)
# 
# test(solutionV9, 1, 1)
# test(solutionV9, 10, 10)
# test(solutionV9, 100, 100)
# test(solutionV9, 1000, 1000)
# test(solutionV9, 10000, 10000)
# test(solutionV9, 100000, 100000)

def testV2(func, min, max):
    start = time.time()

    i = min
    m = 1
    f = 1
    while i <= max:
        # print(m, f)
        # print(func(m, f))
        if m < f:
            m += f
        else:
            f += m
        i += 1
    end = time.time()
    print(end - start)
testV2(solutionV9, 1, 100)
testV2(solutionV9, 1, 1000)
testV2(solutionV9, 1, 10000)
testV2(solutionV9, 1, 100000)
testV2(solutionV9, 1, 1000000)
# testV2(solutionV9, 1, 10000000)
# testV2(solutionV9, 1, 100000000)

print('##########')
testV2(solutionV10, 1, 100)
testV2(solutionV10, 1, 1000)
testV2(solutionV10, 1, 10000)
testV2(solutionV10, 1, 100000)
testV2(solutionV10, 1, 1000000)
    
print('##########')
testV2(solutionV11, 1, 100)
testV2(solutionV11, 1, 1000)
testV2(solutionV11, 1, 10000)
testV2(solutionV11, 1, 100000)
testV2(solutionV11, 1, 1000000)

# given F and M
# F = first number of steps, we can get to [F, 1] in F steps every time
# Each step of M can be determined by F 
# i.e. lets say F is 7, 
# we know it takes 7 steps to get to 7,1
# the next element in the chain will be 7,8
# then 7, 15
# If we pass the number higher than F, return IMPOSSIBLE

# You will have to increment the equation for each iteration 
# until either M or F is greater than their corresponding value

# To calculate any random Leg
# (8, 3)
# stepLength = greatest - greatest % least 
# WRONG legLength = least * stepCount
# stepCount = steplength / least
# greatest = greatest - legLength
 
# ONLY CALCULATE THE STEP COUNT UNTIL 
# greatest would be less than least, 
# then make the turn
# diff = (greatest - least) / least, then round up
# THIS IS THE LENGTH OF THE LEG^^^

# ITS JUST A BREADTH FIRST TREE TRAVERSAL
# each generation only has 2 options, duplicate Mach or Facula 
# Use a queue, track all possibilities, return as soon as we find a match
# If we surpass the count, dont append new cases, if we reach the end of the queue, it is impossible



# First three branches
# print(solutionV5(1, 1))
# print(solutionV5(1, 2))
# print(solutionV5(2, 1))
# print(solutionV5(1, 3))
# print(solutionV5(3, 2))
# print(solutionV5(2, 3))
# print(solutionV5(3, 1))

# Fourth branch
# print(solutionV5(1,4))
# print(solutionV5(4,3))
# print(solutionV5(3,5))
# print(solutionV5(5,2))
# print(solutionV5(2,5))
# print(solutionV5(5,3))
# print(solutionV5(3,4))
# print(solutionV5(4,1))

# Fifth branch
# print('1,5: ' + str(solutionV5(1,5)))
# print('5,4: ' + str(solutionV5(5,4)))
# print('4,7: ' + str(solutionV5(4,7)))
# print('7,3: ' + str(solutionV5(7,3)))
# print('3,8: ' + str(solutionV5(3,8)))
# print('8,5: ' + str(solutionV5(8,5)))
# print('5,7: ' + str(solutionV5(5,7)))
# print('7,2: ' + str(solutionV5(7,2)))
# print('2,7: ' + str(solutionV5(2,7)))
# print('7,5: ' + str(solutionV5(7,5)))
# print('5,8: ' + str(solutionV5(5,8)))
# print('8,3: ' + str(solutionV5(8,3)))
# print('3,7: ' + str(solutionV5(3,7)))
# print('7,4: ' + str(solutionV5(7,4)))
# print('4,5: ' + str(solutionV5(4,5)))
# print('5,1: ' + str(solutionV5(5,1)))



#                     [1,5]
#                [1,4]
#                     [5,4]
#           [1,3]
#                     [4,7]
#                [4,3]
#                     [7,3]
#      [1,2]
#                     [3,8]
#               *[3,5]
#                     [8,5]
#           [3,2]
#                     [5,7]
#                [5,2]
#                     [7,2]
# [1,1]
#                     [2,7]
#                [2,5]
#                     [7,5]
#           [2,3]
#                     [5,8]
#               *[5,3]
#                     [8,3]
#      [2,1]
#                     [3,7]
#                [3,4]
#                     [7,4]
#           [3,1]
#                     [4,5]
#                [4,1]
#                     [5,1]

# ANSWER: (4, 7)
# START: (1, 1)
# 1st Gen: (1, 2)
# 2nd Gen: (1, 3)
# 3rd Gen: (4, 3)
# 4th Gen: (4, 7)

# ANSWER: (2, 1)
# START: (1, 1)
# 1st Gen: (2, 1)

#                          [1,6]
#                     [1,5]
#                          [6,5]
#                [1,4]
#                          [5,9]
#                     [5,4]
#                          [9,4]
#           [1,3]
#                          [4,11]
#                     [4,7]
#                          [11,7]
#                [4,3]
#                          [7,10]
#                     [7,3]
#                          [10,3]
#      [1,2]
#                          [3,11]
#                     [3,8]
#                          [11,8]
#                [3,5]
#                          [8,13]
#                     [8,5]
#                          [13,5]
#           [3,2]
#                          [5,12]
#                     [5,7]
#                          [12,7]
#                [5,2]
#                          [7,9]
#                     [7,2]
#                          [9,2]
# [1,1]
#                          [2,9]
#                     [2,7]
#                          [9,7]
#                [2,5]
#                          [7,12]
#                     [7,5]
#                          [12,5]
#           [2,3]
#                          [5,13]
#                     [5,8]
#                          [13,8]
#                [5,3]
#                          [8,11]
#                     [8,3]
#                          [11,3]
#      [2,1]
#                          [3,10]
#                     [3,7]
#                          [10,7]
#                [3,4]
#                          [7,11]
#                     [7,4]
#                          [11,4]
#           [3,1]
#                          [4,9]
#                     [4,5]
#                          [9,5]
#                [4,1]
#                          [5,6]
#                     [5,1]
#                          [6,1]

# Bomb, Baby!
# ===========
# 
# You're so close to destroying the LAMBCHOP doomsday device you can taste it! But 
# in order to do so, you need to deploy special self-replicating bombs designed for 
# you by the brightest scientists on Bunny Planet. There are two types: Mach bombs 
# (M) and Facula bombs (F). The bombs, once released into the LAMBCHOP's inner workings, 
# will automatically deploy to all the strategic points you've identified and destroy
# them at the same time. 
# 
# But there's a few catches. First, the bombs self-replicate via one of two distinct processes: 
# Every Mach bomb retrieves a sync unit from a Facula bomb; for every Mach bomb, a Facula bomb is created;
# Every Facula bomb spontaneously creates a Mach bomb.
# 
# For example, if you had 3 Mach bombs and 2 Facula bombs, they could either produce 
# 3 Mach bombs and 5 Facula bombs, or 5 Mach bombs and 2 Facula bombs. 
# The replication process can be changed each cycle. 
# 
# Second, you need to ensure that you have exactly the right number of Mach and
# Facula bombs to destroy the LAMBCHOP device. Too few, and the device might survive.
# Too many, and you might overload the mass capacitors and create a singularity
# at the heart of the space station - not good! 
# 
# And finally, you were only able to smuggle one of each type of bomb - one Mach,
# one Facula - aboard the ship when you arrived, so that's all you have to start 
# with. (Thus it may be impossible to deploy the bombs to destroy the LAMBCHOP, 
# but that's not going to stop you from trying!) 
# 
# You need to know how many replication cycles (generations) it will take to generate
# the correct amount of bombs to destroy the LAMBCHOP. Write a function solution(M, F)
# where M and F are the number of Mach and Facula bombs needed. Return the fewest
# number of generations (as a string) that need to pass before you'll have the exact 
# number of bombs necessary to destroy the LAMBCHOP, or the string "impossible"
# if this can't be done! M and F will be string representations of positive integers
# no larger than 10^50. For example, if M = "2" and F = "1", one generation would
# need to pass, so the solution would be "1". However, if M = "2" and F = "4", it
# would not be possible.
# 
# Languages
# =========
# To provide a Java solution, edit Solution.java
# To provide a Python solution, edit solution.py
# 
# Test cases
# ==========
# Your code should pass the following test cases.Note that it may also be run against hidden test cases not shown here.
# 
# -- Java cases --
# Input:Solution.solution('4', '7')Output:    4
# 
# Input:Solution.solution('2', '1')Output:    1
# -- Python cases --
# Input:solution.solution('4', '7')Output:    4
# 
# Input:solution.solution('2', '1')Output:    1
