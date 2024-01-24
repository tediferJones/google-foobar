def isPrime(num, factorsArr):
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
    for i, digit in enumerate(str(num)):
        if charLength + i >= index and charLength + i < index + 5:
            result += digit
        i += 1
    return result

def solution(i):
    if i < 0 or isinstance(i, float):
        raise Exception('Argument must be an integer greater than or equal to 0')

    charLength = 2
    # This array should include 2, but we only check odd numbers and no odd number will ever be divisible by 2
    primeArr = [3] 
    currentNum = 5
    # Since 2 & 3 do not get checked, they may need to be added manually
    newIdNum = '' if i > 1 else '23'[i:]

    while charLength < i + 5:
        if isPrime(currentNum, primeArr):
            if charLength + len(str(currentNum)) > i:
                newIdNum += getDigits(currentNum, i, charLength)
            primeArr.append(currentNum)
            charLength += len(str(currentNum))
        currentNum += 2

    return newIdNum
print(solution(0))
print(solution(3))
print(solution(500))
print(solution(10000))

import time
def test(startIndex=7500, max=10000):
    start = time.time()
    i = startIndex
    while i < max:
        solution(i)
        i += 1
    end = time.time()
    print(startIndex, max)
    print(end - start)
# test()
