import timeit

def isPrimeV2(num, factorsArr):
    # all integers greater than one can be represented as a mulplication of prime numbers
    # This is more effecient than checking every number, especially for large numbers
    maxFactor = pow(num, 0.5)
    i = 0
    while factorsArr[i] <= maxFactor:
        if num % factorsArr[i] == 0:
            return False
        i += 1
    return num > 1

def getDigits(num, index, charLength):
    # only return the digits between index and index + 5
    result = ''
    # digits = str(num)
    # digitCount = 0
    # for digit in digits:
    #     if charLength + digitCount >= index and charLength + digitCount < index + 5:
    #         result += digit
    #     digitCount += 1
    for i, digit in enumerate(str(num)):
        if charLength + i >= index and charLength + i < index + 5:
            result += digit
        i += 1
    return result

def solutionV5(index):
    if index < 0 or isinstance(index, float):
        raise Exception('Argument must be an integer greater than or equal to 0')

    charLength = 2
    primeArr = [3] # This array should include 2, but no odd number will ever be divisble by 2
    currentNum = 5
    # Since 2 & 3 do not get checked, they may need to be added manually
    newIdNum = '' if index > 1 else '23'[index:]

    while charLength < index + 5:
        if isPrimeV2(currentNum, primeArr):
            if charLength + len(str(currentNum)) > index:
                newIdNum += getDigits(currentNum, index, charLength)
            primeArr.append(currentNum)
            charLength += len(str(currentNum))
        currentNum += 2

    return newIdNum
print(solutionV5(0))
print(solutionV5(3))
print(solutionV5(500))
print(solutionV5(1000))

import sys
# print(sys.getrecursionlimit())
# sys.setrecursionlimit(11000)
sys.setrecursionlimit(10118)
# print(sys.getrecursionlimit())
def solutionV4Rec(index, charLength=1, primeArr=[2], currentNum=3, result=''):
    # print(charLength)
    if index < 1 and result == '':
        result = '2'

    if charLength < index + 5: # and isPrimeV2(currentNum, primeArr):
        if isPrimeV2(currentNum, primeArr):
            if charLength + len(str(currentNum)) > index:
                result += getDigits(currentNum, index, charLength)
            primeArr.append(currentNum)
            charLength += len(str(currentNum))
        currentNum += 2
        result = solutionV4Rec(index, charLength, primeArr, currentNum, result)
    return result
print(solutionV4Rec(0))
print(solutionV4Rec(3))
print(solutionV4Rec(500))
print(solutionV4Rec(10000))
print(format(timeit.timeit(solutionV4Rec(2250), number=10000), '.16f'))
print('##########')
    

# THIS IS THE MOST PERFORMANT SOLUTION
# Why?
# We only check odd numbers
# Odd numbers are only checked for factors up to the number's square root
# Odd numbers are only checked against previous prime numbers
#   Because all integers greater than 1 can be represented as
#   a multiplication of prime numbers
# The result is compounded as new prime numbers are found
#   This means we dont have to convert the whole array into a 
#   string just to slice out a small section of it
# The while loop will terminate as soon we have parsed 
# the last digit containing numbers needed for the given index
def solutionV3(index):
    if index < 0 or isinstance(index, float):
        raise Exception('Argument must be an integer greater than or equal to 0')

    charLength = 1
    primeArr = [2]
    currentNum = 3
    # Since the number 2 does not get checked, it may need to be added manually
    newIdNum = '' if index > 0 else '2'

    while charLength < index + 5:
        if isPrimeV2(currentNum, primeArr):
            if charLength + len(str(currentNum)) > index:
                newIdNum += getDigits(currentNum, index, charLength)
            primeArr.append(currentNum)
            charLength += len(str(currentNum))
        currentNum += 2

    return newIdNum
print(solutionV3(0))
print(solutionV3(3))
print(solutionV3(500))
print(solutionV3(10000))
print(format(timeit.timeit(solutionV3(2250), number=10000), '.16f'))
# print(format(timeit.timeit(test(solutionV3)), '.16f'))
# print(timeit.timeit(solutionV3(10000), number=1000))
# test(solutionV3)
print('##########')


def solutionV2(index):
    if index < 0 or isinstance(index, float):
        raise Exception('Argument must be an integer greater than or equal to 0')

    charLength = 1
    primeArr = [2]
    currentNum = 3

    while charLength < index + 5:
        if isPrimeV2(currentNum, primeArr):
            primeArr.append(currentNum)
            charLength += len(str(currentNum))
        currentNum += 2

    primeString = ''
    for num in primeArr:
        primeString += str(num)

    return primeString[index : index + 5]
print(solutionV2(0))
print(solutionV2(3))
print(solutionV2(500))
print(solutionV2(10000))
print('##########')


def isPrime(num):
    maxFactor = pow(num, 0.5)
    i = 2
    while i <= maxFactor:
        if num % i == 0:
            return False
        i += 1
    return num > 1

def solution(index):
    if index < 0 or isinstance(index, float):
        raise Exception('Argument must be an integer greater than or equal to 0')

    primeString = '2'
    currentNum = 3
    while len(primeString) < index + 5:
        if isPrime(currentNum):
            primeString += str(currentNum)
        # Start with an odd number then increment by 2
        # No point in checking even numbers, all even numbers can be divided by 2
        currentNum += 2

    # Uncomment this to print the full primeString
    # print(primeString)
    return primeString[index : index + 5]

print(solution(0))
print(solution(3))
print(solution(500))
print(solution(10000))
# print(solution(1000000))
# print(solution(3.1))
# print(solution(-0.01))
# print(solution(-1))

import time
def test(i=7500, max=10000):
    start = time.time()
    # i = startIndex
    print(i, max)
    while i < max:
        # solution(i)
        # solutionV2(i)
        solutionV3(i)
        # solutionV4Rec(i)
        i += 1
    end = time.time()
    # print(startIndex, max)
    print(end - start)
test()

# def generatePrimeString(minLength):
#     # primeString = ''
#     # currentNum = 2
#     primeString = '2'
#     currentNum = 3
#     while len(primeString) < minLength:
#         if isPrime(currentNum):
#             primeString += str(currentNum)
#         currentNum += 2
#     return primeString
# 
# def getId(index):
#     return generatePrimeString(index + 5)[ index : index + 5 ]
# 
# print(getId(0))
# print(getId(3))
# 
# print(generatePrimeString(7))
# print(isPrime(2))
# print(isPrime(4))
# print(isPrime(17))
# print(isPrime(99))
# print(isPrime(16))
