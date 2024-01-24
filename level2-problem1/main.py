# This equation will sum numbers from a to a+d*n with matching step of d
# a = first term, d = radius (step), n = number of steps
#   n(2a + (n - 1) * d)
# = -------------------
#           2
# In our case d will always be 1

# The desired id number can found by finding the sum of 2 arithmetic sequences like so:
#
#       arthSeq(1, 1, x) + arthSeq(x, 1, y - 1),
#
# Instead of calling these functions manually we can just add these two equations together and simplify

# = arthSeq(1, 1, x) + arthSeq(x, 1, y - 1)
#
#   x(2 + x - 1)   (y - 1)(2x + y - 1 - 1)
# = ------------ + -----------------------
#         2                   2      
#
#   x^2 + x   y^2 + 2xy - 2y - 2x - y + 2
# = ------- + ---------------------------
#      2                  2      
#
#   y^2 + x^2 + 2xy - 3y - x + 2
# = ----------------------------
#                2

# def arthSeq(firstTerm, step, count):
#     return count / 2 * (2 * firstTerm + (count - 1) * step)

def solution(x, y):
    # return str(int(arthSeq(1, 1, x) + arthSeq(x, 1, y - 1)))
    return str(int((y ** 2 + x ** 2 + 2 * x * y - 3 * y - x + 2) / 2))
print(solution(1, 1))
print(solution(3, 2))
print(solution(5, 10))
print(solution(100000, 100000))

# | 11
# | 7 12
# | 4 8 13
# | 2 5 9 14
# | 1 3 6 10 15

def arithmeticSequence(a, d, n):
    # a = first term
    # d = radius (step)
    # n = number of steps
    # return n / 2 * (2 * a + (n - 1) * d)
    return n / 2 * (2 * a + n - 1)
def solutionV5(x, y):
    return str(int(arithmeticSequence(1, 1, x) + arithmeticSequence(x, 1, y - 1)))
print(solutionV5(1, 1))
print(solutionV5(3, 2))
print(solutionV5(5, 10))
print(solutionV5(100000, 100000))

def solutionV4(x, y):
    # This function can be determined by expanding solutionV3 or solutionV5
    return str(int((y ** 2 + 2 * x * y - 3 * y - x + x ** 2 + 2) / 2))
print(solutionV4(1, 1))
print(solutionV4(3, 2))
print(solutionV4(5, 10))
print(solutionV4(100000, 100000))

def sumBetween(min, max):
    return int((max - min + 1) * (min + max) / 2)
def solutionV3(x, y):
    return sumBetween(1, x) + sumBetween(x, x + y - 2)
# print(solutionV3(3, 3))

import time
def test(x, y):
    start = time.time()
    currX = 1
    currY = 1
    result = []
    # Generate (x , y) up to max in both vars
    while currX <= x:
        # print('x = ' + str(currX))
        while currY <= y:
            # print('y = ' + str(currY))
            # result.append(solutionV3(currX, currY))
            result.append(solutionV4(currX, currY))
            # result.append(solutionV5(currX, currY))
            currY += 1
        currY = 1
        currX += 1

    end = time.time()
    print('##########')
    print(x, y)
    # print(result)
    print(end - start)
# test(5000, 5000)

# def sumFrom(min, count):
#     result = 0
#     while count > 0:
#         result += min
#         min += 1
#         count -= 1
#     return result
# def solutionV2(x, y):
#     return sumFrom(1, x) + sumFrom(x, y - 1)
# print(solutionV2(1, 5))
# solutionV2(1, 1) # 1
# solutionV2(2, 1) # 3
# solutionV2(3, 1) # 6
# solutionV2(4, 1) # 10


# def solution(x, y):
#     # result = None
#     result = 1
#     counter = 1
#     inc = 2
#     while counter < x:
#         result += inc
#         inc += 1
#         counter += 1
#     print(result)
#     # Now just do the same thing with the other dimension of the table
# solution(1, 1) # 1
# solution(2, 1) # 3
# solution(3, 1) # 6
# solution(4, 1) # 10

# THE SOLUTION:
# You could write out the whole graph until we reach the write coordinate
# OR we can predict the number from the coordinates

# INSTEAD OF ADDING NUMBERS SEQUENTIALLY
# 1+2+3+4+5 === (5*6)/2


# THE PROBLEM:
# Bunny Worker Locations
# ======================
# 
# Keeping track of Commander Lambda's many bunny workers is starting to get tricky.
# You've been tasked with writing a program to match bunny worker IDs to cell 
# locations.The LAMBCHOP doomsday device takes up much of the interior of Commander 
# Lambda's space station, and as a result the work areas have an unusual layout. 
# They are stacked in a triangular shape, and the bunny workers are given i
# numerical IDs starting from the corner, as follows:
# | 7
# | 4 8
# | 2 5 9
# | 1 3 6 10
# Each cell can be represented as points (x, y), with x being the distance from 
# the vertical wall, and y being the height from the ground. For example, the 
# bunny worker at (1, 1) has ID 1, the bunny worker at (3, 2) has ID 9, and the 
# bunny worker at (2,3) has ID 8. This pattern of numbering continues indefinitely 
# (Commander Lambda has been adding a LOT of workers). 
# 
# Write a function solution(x, y) which returns the worker ID of the bunny at location (x, y). 
# Each value of x and y will be at least 1 and no greater than 100,000. 
# Since the worker ID can be very large, return your solution as a string representation of the number.
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
# Input:Solution.solution(3, 2)Output:    9
# 
# Input:Solution.solution(5, 10)Output:    96
# -- Python cases --
# Input:solution.solution(5, 10)Output:    96
# 
# Input:solution.solution(3, 2)Output:    9
