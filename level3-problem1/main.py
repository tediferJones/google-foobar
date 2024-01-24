# For example, [1, 2, 3 , 4, 5, 6] has the triples: 
# [1, 2, 4], [1, 2, 6], [1, 3, 6], making the solution 3 total.
# Find out how many lucky triples can be made from a given array of numbers
# A lucky triple is three numbers (x, y, z) where
# y % x === 0
# z % y === 0

# def getNext(arr, start):
#     result = [arr[start]]
#     i = start
#     while i < len(arr):
#         if arr[i] % arr[start] == 0:


# def solutionRec(arr, start=0, result=[]):
#     temp = []
#     if len(result) > 0 and arr[start] % result[-1] == 0:
#         temp.append(arr[start])

# ONLY SCAN THE POSSIBLE SECOND NUMBERS (l[1 : -1])
# Get all factors before that number
# Get all multiplicants after that number
# number of triples for the given second number = factors * multiplicants
# add that number to count
# return count
def solutionV8(l):
    length = len(l)
    count = 0
    for secondIndex in range(1, length - 1):
        factorCount = 0
        multCount = 0

        for factorIndex in range(secondIndex):
            if l[secondIndex] % l[factorIndex] == 0:
                factorCount += 1

        for multIndex in range(secondIndex + 1, length):
            if l[multIndex] % l[secondIndex] == 0:
                multCount += 1

        count += factorCount * multCount
    return count
# print(solutionV8([1,1,1]))
# print(solutionV8([1,2,3,4,5,6]))
# print(solutionV8([1, 2, 3, 4, 5, 6, 7, 8, 9]))
# print(solutionV8([2, 3, 4, 5, 6, 7, 8, 9, 12]))
# print(solutionV8([2, 3, 4, 5, 6, 7, 8, 9, 12, 16]))
# print(solutionV8([2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 24, 48, 96]))
# print(solutionV8([1, 1, 1]))
# print(solutionV8([1] * 100))

def solutionV7(l):
    length = len(l)
    result = 0
    # pairs = [] * l
    pairs = [ [] for _ in range(length) ]
    for firstIndex in range(length - 2):
        for secondIndex in range(length - 1):
            if l[secondIndex] % l[firstIndex] == 0:
                pairs[firstIndex].append(secondIndex)

    # for thirdIndex in range(2, length):
    print(pairs)
# solutionV7([1,2,3,4,5,6])


def solutionDadEdition(l):
    length = len(l)
    result = 0
    for firstIndex in range(length - 2):
        for secondIndex in range(firstIndex + 1, length - 1):
            if l[secondIndex] % l[firstIndex] == 0:
                for thirdIndex in range(secondIndex + 1, length):
                    if l[thirdIndex] % l[secondIndex] == 0:
                        result += 1
    return result
# print(solutionDadEdition([1,2,3,4,5,6]))
# print(solutionDadEdition([1,1,1]))
# print(solutionDadEdition([1] * 2000))

# def solutionV6(l):
#     length = len(l)
#     pairsObj = {}
#     pairs = [0] * length
#     count = 0
# 
#     for firstIndex,firstNum in enumerate(l):
#         for secondIndex,secondNum in enumerate(l[ firstIndex : -1], firstIndex):
#             if secondNum % firstNum == 0:
#                 # if pairsObj[firstIndex] == None
#                 pairs[secondIndex] += 1
#     return pairs
# print(solutionV6([1,2,3,4,5,6]))


def solutionV5(l):
    length = len(l)
    pairs = [0] * length
    count = 0
    # for firstIndex in range(len(l)):
    for secondIndex,secondNum in enumerate(l[1 : ], 1):
        for firstIndex,firstNum in enumerate(l[secondIndex : ]):
            if secondNum % firstNum == 0:
                pairs[secondIndex] += 1
    
    for thirdIndex,thirdNum in enumerate(l[2 : ], 2):
        for secondIndex,secondNum in enumerate(l[thirdIndex: ]):
            if thirdNum % secondNum == 0:
                count += pairs[secondIndex]

    return count
# print(solutionV5([1,2,3,4,5,6]))
# print(solutionV5([1] * 2000))


def solutionV4(l):
    count = 0
    length = len(l)
    pairs = [0] * length

    for secondIndex in range(1, length):
        for firstIndex in range(secondIndex):
            if l[secondIndex] % l[firstIndex] == 0:
                pairs[secondIndex] += 1

    for thirdIndex in range(2, length):
        for secondIndex in range(1, thirdIndex):
            if l[thirdIndex] % l[secondIndex] == 0:
                count += pairs[secondIndex]
    return count
# print(solutionV4([1,1,1]))
# print(solutionV4([1] * 2000))

def getFactors(num, arr):
    factors = []
    for factor in arr:
        if factor % num == 0:
            factors.append(factor)
    return factors

def solution(l):
    # result = []
    result = 0
    for firstIndex,firstNum in enumerate(l[ : -2]):
        secondOptions = getFactors(firstNum, l[ firstIndex + 1 : ])
        for secondIndex,secondNum in enumerate(secondOptions[ : -1 ]):
            thirdNums = getFactors(secondNum, secondOptions[secondIndex + 1 : ])
            result += len(thirdNums)
            # for thirdNum in getFactors(secondNum, secondOptions[secondIndex + 1 : ]):
            #     # result.append([firstNum, secondNum, thirdNum])
            #     result += 1

    # return len(result)
    return result
# print(solution([1,2,3,4,5,6]))
# print(solution([1] * 2000))
# print(solution([1] * 1000))
# print(solution([1] * 1000))

def solutionV3(l):
    result = []
    for firstIndex,firstNum in enumerate(l[ : -2]):
        secondOptions = getFactors(firstNum, l[ firstIndex + 1 : ])
        # for secondIndex,secondNum in enumerate(secondOptions):
        for secondIndex,secondNum in enumerate(secondOptions[ : -1 ]):
            for thirdNum in getFactors(secondNum, secondOptions[secondIndex + 1 : ]):
                result.append([firstNum, secondNum, thirdNum])
        # secondOptions = getFactors(firstNum, l[ firstIndex + 1 : ])
        # for secondIndex,secondNum in enumerate(secondOptions):
        #     thirdOptions = getFactors(secondNum, secondOptions[ secondIndex + 1 :  ])
        #     for thirdNum in thirdOptions:
        #         result.append([firstNum, secondNum, thirdNum])
    return len(result)
# print(solutionV3([1, 2, 3, 4, 5, 6, 7, 8, 9]))
# print(solutionV3([2, 3, 4, 5, 6, 7, 8, 9, 12]))
# print(solutionV3([2, 3, 4, 5, 6, 7, 8, 9, 12, 16]))
# print(solutionV3([2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 24, 48, 96]))
# print(solutionV3([1, 1, 1]))

def solutionV2(l):
    result = []
    for firstIndex,firstNum in enumerate(l[ : -2]):
        options = []
        # for secondIndex,secondNum in enumerate(l[firstIndex + 1 : ], firstIndex + 1):
        #     if secondNum % firstNum == 0:
        #         options.append(secondNum)

        # for factor in enumerate(l[firstIndex + 1 : ], firstIndex + 1):
        for factor in l[firstIndex + 1 : ]:
            if factor % firstNum == 0:
                options.append(factor)
        # options = list(filter(lambda num: num % firstNum == 0, l[firstIndex + 1 : ]))
        
        for secondIndex,secondNum in enumerate(options):
            # for thirdIndex,thirdNum in enumerate(options[ secondIndex + 1 : ]):
            for thirdNum in options[ secondIndex + 1 : ]:
                if thirdNum % secondNum == 0:
                    result.append([firstNum, secondNum, thirdNum])
    return len(result)
# print(solutionV2([1, 2, 3, 4, 5, 6, 7, 8, 9]))
# print(solutionV2([2, 3, 4, 5, 6, 7, 8, 9, 12]))
# print(solutionV2([2, 3, 4, 5, 6, 7, 8, 9, 12, 16]))
# print(solutionV2([2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 24, 48, 96]))
# print(solutionV2([1, 1, 1]))


def solutionV1(l):
    # print(l)
    result = []
    for firstIndex,firstNum in enumerate(l[ : -2]):
        for secondIndex,secondNum in enumerate(l[firstIndex + 1 : -1], firstIndex + 1):
            if secondNum % firstNum == 0:
                for thirdIndex,thirdNum in enumerate(l[secondIndex + 1 : ], secondIndex + 1):
                    if thirdNum % secondNum == 0:
                        result.append([firstNum, secondNum, thirdNum])
    # print(result)
    return len(result)
# print('TESTED RESULTS')
# print(solutionV1([1, 2, 3, 4, 5, 6, 7, 8, 9]))
# print(solutionV1([2, 3, 4, 5, 6, 7, 8, 9, 12]))
# print(solutionV1([2, 3, 4, 5, 6, 7, 8, 9, 12, 16]))
# print(solutionV1([2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 24, 48, 96]))
# print(solutionV1([1, 1, 1]))
# print(solutionV2([1] * 100))


# solutionRec([1,2,3])
def iterSolution(arr):
    result = []
    index1 = 0
    while index1 < len(arr):
        index2 = index1 + 1
        # print(str(arr[index1]))
        while index2 < len(arr):
            # print('    ' + str(arr[index2]))
            if arr[index2] % arr[index1] == 0:
                index3 = index2 + 1
                while index3 < len(arr):
                    # print('        ' + str(arr[index3]))
                    if arr[index3] % arr[index2] == 0:
                        result.append([arr[index1], arr[index2], arr[index3]])
                    index3 += 1
            index2 += 1
        index1 += 1
    # return result if len(result) > 0 else 0
    return len(result)
# print(iterSolution([1, 2, 3, 4, 5, 6, 7, 8, 9]))
# print(iterSolution([2, 3, 4, 5, 6, 7, 8, 9, 12]))
# print(iterSolution([2, 3, 4, 5, 6, 7, 8, 9, 12, 16]))
# print(iterSolution([1, 1, 1]))

import random
import time
def test(func, arr):
    # arrLength = int(random.random() * 2000)
    # arrLength = random.randint(0, 2000)
    startTime = time.time()
    for testInput in arr:
        func(testInput)
    # i = 0;
    # while i < 64:
    #     testArr = random.sample(range(1, 9999), 2000)
    #     # iterSolution(testArr)
    #     # solution(testArr)
    #     # solutionV2(testArr)
    #     # solutionV3(testArr)
    #     func(testArr)
    #     # result = solutionV2(testArr)
    #     # print(result == iterSolution(testArr) and result == solution(testArr))
    #     # if not result == iterSolution(testArr) and not result == solution(testArr):
    #     #     print("ANSWERS DO NOT MATCH, SOMETHING IS FUCKED UP")
    #     # print(solution(testArr) == iterSolution(testArr))
    #     i += 1
    # # print(testArr)
    endTime = time.time()
    print(endTime - startTime)
    return endTime - startTime
    # print(iterSolution(testArr))
test(solutionV8, [[1] * 2000])
test(solutionV4, [[1] * 2000])

# testIndex = 0
# times1 = []
# times2 = []
# while testIndex < 8:
#     inputs = [];
#     inputCounter = 0
#     while inputCounter < 64:
#         input = [1] * 2000
#         # input = []
#         # inputLength = 0
#         # while inputLength < 2000:
#         #     input.append(random.randrange(1, 999))
#         #     inputLength += 1
#         # inputs.append(random.sample(range(1, 9999), 2000))
#         inputs.append(input)
#         inputCounter += 1
#     # print(inputs)
# 
#     print('times1')
#     times1.append(test(solutionV8, inputs))
#     print('times2')
#     times2.append(test(solution, inputs))
#     print('##########')
#     testIndex += 1
# print('times1 AVERAGE:')
# times1Avg = sum(times1) / len(times1)
# print(times1Avg)
# print('times2 AVERAGE:')
# times2Avg = sum(times2) / len(times2)
# print(times2Avg)
# if times1Avg < times2Avg:
#     print('\ntimes1 is ' + str((1 - times1Avg / times2Avg) * 100) + '% faster\n')
# else:
#     print('\ntimes2 is ' + str((1 - times2Avg / times1Avg) * 100) + '% faster\n')



# def solution(l):
#     for num,i in l:
#         getNext(i, l)
    # result = []
    # i = 0
    # while i < len(l):
    #     currentArr = [l[i]]
    #     currentI = i
    #     getNext(l, currentI)
    #     # while currentI < len(l):
    #     # for currentI in l:
    #     #     if l[currentI] % currentArr[-1] == 0:
    #     #         currentArr.append(l[currentI])
    #     #         if len(currentArr) == 3:
    #     #             break

    #     #     currentI += 1
    # i += 1



# Find the Access Codes
# =====================

# In order to destroy Commander Lambda's LAMBCHOP doomsday device, you'll need access 
# to it. But the only door leading to the LAMBCHOP chamber is secured with a unique 
# lock system whose number of passcodes changes daily. Commander Lambda gets a report 
# every day that includes the locks' access codes, but only the Commander knows how 
# to figure out which of several lists contains the access codes. You need to find 
# a way to determine which list contains the access codes once you're ready to go 
# in.

# Fortunately, now that you're Commander Lambda's personal assistant, Lambda 
# has confided to you that all the access codes are "lucky triples" in order to make 
# it easier to find them in the lists. A "lucky triple" is a tuple (x, y, z) where 
# x divides y and y divides z, such as (1, 2, 4). With that information, you can 
# figure out which list contains the number of access codes that matches the number 
# of locks on the door when you're ready to go in (for example, if there's 5 passcodes,
# you'd need to find a list with 5 "lucky triple" access codes).

# Write a function solution(l) that takes a list of positive integers l and counts 
# the number of " lucky triples" of (li, lj, lk) where the list indices meet the 
# requirement i < j < k.  The length of l is between 2 and 2000 inclusive.  The elements 
# of l are between 1 and 999999 inclusive.  The solution fits within a signed 32-
# bit integer. Some of the lists are purposely generated without any access codes 
# to throw off spies, so if no triples are found, return 0. 

# For example, [1, 2, 3 , 4, 5, 6] has the triples: 
# [1, 2, 4], [1, 2, 6], [1, 3, 6], making the solution 3 total.

# Languages
# =========
# To provide a Java solution, edit Solution.java
# To provide a Python solution, edit solution.py

# Test cases
# ==========
# Your code should pass the following test cases.Note that it may also be run against hidden test cases not shown here.
# 
# -- Java cases --
# Input:Solution.solution([1, 1, 1])Output:    1
# 
# Input:Solution.solution([1, 2, 3, 4, 5, 6])Output:    3
# -- Python cases --
# Input:solution.solution([1, 1, 1])Output:    1
# 
# Input:solution.solution([1, 2, 3, 4, 5, 6])Output:    3
