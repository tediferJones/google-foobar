# Matrix operations
# row operations:
# add
# subtract
# multiply or divide by integer

# 1. convert goofy shit into IQR
#    - Cant we just flip the matrix over?
#       NO, what if all absorbing cases are not in order
# 2. Once you can get matrix R and matrix Q
# 3. Sub tract Q - I
# 4. Then gauss Jordan that shit, to inverse it
# 5. Once we have the inverse we have F
# 6. Do F * R (in this order) and some portion of that resulting matrix will container our answers

# The height of Q is dictated by the number of transient states
# The width of Q = maxRow size - number of absorbing cases

# TO-DO:
#   Figure out how to inverse I - Q, the result of this will be matrix F
#   [ DONE ] subtract() should also simplify/reduce the fraction
#   create a function to multiply matrices
#   [ DONE ] Create a function to generate the new matrix order

def fixAbsorbingRows(m):
    for rowIndex in range(len(m)):
        if m[rowIndex] == [0] * len(m[rowIndex]):
            m[rowIndex][rowIndex] = 1

def getNewMatrixFormat(m):
    absorbing = []
    transient = []
    for rowIndex in range(len(m)):
        if m[rowIndex] == [0] * len(m[rowIndex]):
            absorbing.append(rowIndex)
        else:
            transient.append(rowIndex)
    return absorbing + transient

def convertMatrix(m, orderArr):
    result = []
    for rowNum in orderArr:
        newRow = []
        for rowIndex in orderArr:
            newRow.append(m[rowNum][rowIndex])
        result.append(newRow)
    return result

class FakeFrac:
    def __init__(self, numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator

    def getFloat(self):
        return self.numerator / self.denominator

    def getString(self):
        return str(self.numerator) + '/' + str(self.denominator)

def convertToFracs(m):
    # Consider making this a pure function
    result = []
    for rowIndex in range(len(m)):
        denominator = sum(m[rowIndex])
        result.append([])
        for cellIndex in range(len(m[rowIndex])):
            result[rowIndex].append(FakeFrac(m[rowIndex][cellIndex], denominator))

    return result

def getRQ(m):
    result = []
    absorbingState = [0] * len(m)
    # Get rid of all absorbing states
    for rowIndex in range(len(m)):
        cell = m[rowIndex][rowIndex]
        if cell.numerator != 1 and cell.denominator != 1:
            result.append(m[rowIndex])

    # print('PRINTING RESULT')
    # ppm(result)
    r = []
    q = []
    absorbingCount = len(m) - len(result)
    # Slice the remaining arrays to get r and q
    for rowIndex in range(len(result)):
        r.append(result[rowIndex][ : absorbingCount])
        q.append(result[rowIndex][absorbingCount : ])
    return [r,q]

def generateIdMatrix(size):
    idMatrix = []
    for rowIndex in range(size):
        idMatrix.append([])
        for cellIndex in range(size):
            cell = FakeFrac(1, 1) if rowIndex == cellIndex else FakeFrac(0, 1)
            idMatrix[rowIndex].append(cell)
    return idMatrix

def getGCD(a, b):
    while b:
        a, b = b, a%b
    return a
# print(getGCD(12,16))

def simplifyFrac(frac):
    if frac.numerator == 0:
        return FakeFrac(0, 1)

    if frac.numerator == frac.denominator:
        return FakeFrac(1, 1)

    gcd = getGCD(frac.numerator, frac.denominator)
    return FakeFrac(frac.numerator // gcd, frac.denominator // gcd)

def subtract(a, b):
    # print(a, b)
    if a.denominator == b.denominator:
        return FakeFrac(a.numerator - b.numerator, a.denominator)
    else:
        newDenominator = a.denominator * b.denominator
        newNumeratorA = a.numerator * b.denominator
        newNumeratorB = b.numerator * a.denominator
        return simplifyFrac(FakeFrac(newNumeratorA - newNumeratorB, newDenominator))

def subtractMatrices(m1, m2):
    result = []
    for rowIndex in range(len(m1)):
        result.append([])
        for cellIndex in range(len(m1[rowIndex])):
            result[rowIndex].append(
                subtract(m1[rowIndex][cellIndex], m2[rowIndex][cellIndex])
            )
    return result

def addFracs(a, b):
    if a.denominator == b.denominator:
        return FakeFrac(a.numerator + b.numerator, a.denominator)
    else:
        newDenominator = a.denominator * b.denominator
        newNumeratorA = a.numerator * b.denominator
        newNumeratorB = b.numerator * a.denominator
        return simplifyFrac(FakeFrac(newNumeratorA + newNumeratorB, newDenominator))

def multiplyFracs(a, b):
    return simplifyFrac(FakeFrac(a.numerator * b.numerator, a.denominator * b.denominator))

def multiplyMatrices(m1, m2):
    # Dimensions of the final matrix can be found by doing:
    # m1 row Count by m2 Col count
    # result = [[FakeFrac(0,1) for x in range(3)] for y in range(3)]
    # result = [[FakeFrac(0,1) for x in range(len(m1))] for y in range(len(m2[0]))]
    result = [[FakeFrac(0,1) for x in range(len(m2[0]))] for y in range(len(m1))]
 
    # explicit for loops
    for i in range(len(m1)):
        for j in range(len(m2[0])):
            for k in range(len(m2)):
                # resulted matrix
                # result[i][j] += matrix1[i][k] * matrix2[k][j]

                # multiply the current selection together
                # then sum it with exists value
                # print(i,j,k)
                product = multiplyFracs(m1[i][k], m2[k][j])
                result[i][j] = addFracs(result[i][j], product)

    return result

def invert2x2(m):
    # m = [
    #     [a,b]
    #     [c,d]
    # ]
    # m^-1 = 1 / (a*d - b*c) * m
    result = []
    a = m[0][0]
    b = m[0][1]
    c = m[1][0]
    d = m[1][1]
    denominator = subtract(multiplyFracs(a,d), multiplyFracs(b,c))
    # scalar = multiplyFracs(FakeFrac(1,1), denominator)
    scalar = multiplyFracs(FakeFrac(1,1), FakeFrac(denominator.denominator, denominator.numerator))
    print('SCALER VALUE')
    print(scalar.getString())
    # Flip the signs for b and c in given matrix
    m[0][1] = FakeFrac(m[0][1].numerator * -1, m[0][1].denominator)
    m[1][0] = FakeFrac(m[1][0].numerator * -1, m[1][0].denominator)

    for rowIndex in range(len(m)):
        result.append([])
        for cellIndex in range(len(m[rowIndex])):
            result[rowIndex].append(multiplyFracs(m[rowIndex][cellIndex], scalar))

    return result


# Pretty Print Matrix
def ppm(m):
    # for row in m:
    #     print(row)
    for rowIndex in range(len(m)):
        row = []
        for cellIndex in range(len(m[rowIndex])):
            if isinstance(m[rowIndex][cellIndex], FakeFrac):
                # print('found a fakeFrac')
                row.append(m[rowIndex][cellIndex].getString())
            else:
                row.append(m[rowIndex][cellIndex])
        print(row)

# ppm(multiplyMatrices(
#     [
#         [FakeFrac(1,1), FakeFrac(-2,3)],
#         [FakeFrac(0,1), FakeFrac(1,1)],
#     ],[
#         [FakeFrac(1,1), FakeFrac(2,3)],
#         [FakeFrac(0,1), FakeFrac(1,1)]
#     ]
# ))
# ppm(multiplyMatrices(matrix1, matrix2))

def formatAnswer(arr):
    answer = []
    highestDenominator = 0
    for frac in arr:
        if frac.denominator > highestDenominator:
            highestDenominator = frac.denominator
    # print(highestDenominator)
    for frac in arr:
        if frac.denominator == highestDenominator:
            answer.append(frac.numerator)
        else:
            factor = highestDenominator // frac.denominator
            # print('FACTOR')
            # print(factor)
            answer.append(frac.numerator * factor)
    answer.append(highestDenominator)
    return answer


def solution(m):
    # rq = getRQ(m)
    # ppm(rq)
    print('Original Matrix')
    ppm(m)
    print('################')
    print('Get new matrix ordering')
    matrixFormat = getNewMatrixFormat(m)
    print(matrixFormat)
    print('################')
    print('Fixed absorbing states')
    fixAbsorbingRows(m)
    ppm(m)
    print('################')
    print('Convert to IROQ format')
    # We need to figure out how to generate this array dynamically
    test = convertMatrix(m, matrixFormat)
    ppm(test)
    print('################')
    print('Convert entire matrix to fractions')
    fracs = convertToFracs(test)
    ppm(fracs)
    print('################')
    print('Get R and Q')
    r, q = getRQ(fracs)
    ppm(r)
    ppm(q)
    print('################')
    print('Generate a generic identity matrix')
    ppm(generateIdMatrix(4))
    # WE HAVE R AND Q, NOW DO Q - I
    print('################')
    print('Subtract I from Q')
    matrixToInvert = subtractMatrices(generateIdMatrix(len(q)), q)
    ppm(matrixToInvert)
    print('################')
    print('Get inverse of I - Q, i.e F')
    invertedMatrix = invert2x2(matrixToInvert)
    ppm(invertedMatrix)
    print('################')
    print('GET FR')
    ppm(invertedMatrix)
    ppm(r)
    fr = multiplyMatrices(invertedMatrix, r)
    ppm(fr)
    print('################')
    print('THE ANSWERS')
    result = fr[0]
    ppm([result])
    print('################')
    print('Format answers into correct format')
    answers = formatAnswer(result)
    print(answers)

# solution([
#     [1,0],
#     [0,1]
# ])
solution([
    [0, 2, 1, 0, 0],
    [0, 0, 0, 3, 4],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])
solution([
    [0, 1, 0, 0, 0, 1], 
    [4, 0, 0, 3, 2, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
])

# ABSORBING MARKOV CHAINS ARE THE KEY TO SOLVING RECURSIVE PROBABILITY
# absorbing states (i.e. terminal cases): s2, s3, s4, s5
# EXAMPLE:
# [0, 1, 0, 0, 0, 1], 
# [4, 0, 0, 3, 2, 0],
# [0, 0, 0, 0, 0, 0],
# [0, 0, 0, 0, 0, 0],
# [0, 0, 0, 0, 0, 0],
# [0, 0, 0, 0, 0, 0],


# Simplified Problem:
# You are given a 2D array containg ints >= 0
# Each array is different stage of the process
# An array containing all 0's is terminal state, thats what we need to get to
# Always start with first array, we will call is s0
# each value,index corresponds to probability,nextStage (i.e. the list index of the next stage)
# Determine the probability for each possible 'route' to a 'terminal' state 

# It is technically possible for a 

# Example:
# s0: [0,1,0],
# s1: [0,0,1],
# s2: [0,0,0],

# All possible routes:
# s0 -1> s1 -1> s2

# s0: [0,1,0],
# s1: [1,0,1],
# s2: [0,0,0],

# s0 -1> s1 -1/2> s0 -1> s1 -1/2> s2
# s0 -1> s1 -1/2> s2 

# s0: [1, 1]
# s1: [0, 0]

# [0, 1, 0, 0, 0, 1], 
# [4, 0, 0, 3, 2, 0],
# [0, 0, 0, 0, 0, 0],
# [0, 0, 0, 0, 0, 0],
# [0, 0, 0, 0, 0, 0],
# [0, 0, 0, 0, 0, 0],

# 1/2 * 4/9 = 4/18 = 2/9
# 2*2/9*9
# 2^3/9^3

# 1/2 * 1/3 = 1/6
# 1/6 * 4/18 = 4/108 = 2/54 = 1/27

# 1/6 * 4/9 = 4/54

# 4/18 * 1/3 = 4/54

# Doomsday Fuel
# =============
# Making fuel for the LAMBCHOP's reactor core is a tricky process because of the 
# exotic matter involved. It starts as raw ore, then during processing, begins randomly 
# changing between forms, eventually reaching a stable form. There may be multiple 
# stable forms that a sample could ultimately reach, not all of which are useful as fuel. 

# Commander Lambda has tasked you to help the scientists increase fuel creation efficiency 
# by predicting the end state of a given ore sample. You have carefully studied the 
# different structures that the ore can take and which transitions it undergoes. 
# It appears that, while random, the probability of each structure transforming is 
# fixed. That is, each time the ore is in 1 state, it has the same probabilities 
# of entering the next state (which might be the same state).  You have recorded 
# the observed transitions in a matrix. The others in the lab have hypothesized more 
# exotic forms that the ore can become, but you haven't seen all of them.

# Write a function solution(m) that takes an array of array of nonnegative ints representing 
# how many times that state has gone to the next state and return an array of ints 
# for each terminal state giving the exact probabilities of each terminal state, represented 
# as the numerator for each state, then the denominator for all of them at the end 
# and in simplest form. The matrix is at most 10 by 10. It is guaranteed that no 
# matter which state the ore is in, there is a path from that state to a terminal 
# state. That is, the processing will always eventually end in a stable state. The 
# ore starts in state 0. The denominator will fit within a signed 32-bit integer 
# during the calculation, as long as the fraction is simplified regularly.

# For example consider the matrix m:[
#     [0,1,0,0,0,1], # s0, the initial state, goes to s1 and s5 with equal probability
#     [4,0,0,3,2,0], # s1 can become s0, s3, or s4, but with different probabilities
#     [0,0,0,0,0,0], # s2 is terminal, and unreachable (never observed in practice)
#     [0,0,0,0,0,0], # s3 is terminal
#     [0,0,0,0,0,0], # s4 is terminal
#     [0,0,0,0,0,0], # s5 is terminal
# ]
# 
# So, we can consider different paths to terminal states, such as:
#     s0 -> s1 -> s3
#     s0 -> s1 -> s0 -> s1 -> s0 -> s1 -> s4
#     s0 -> s1 -> s0 -> s5
# Tracing the probabilities of each, we find that
#     s2 has probability 0
#     s3 has probability 3/14
#     s4 has probability 1/7
#     s5 has probability 9/14
# 
# So, putting that together, and making a common denominator, gives an answer in the form of
# [s2.numerator, s3.numerator, s4.numerator, s5.numerator, denominator] which is
# [0, 3, 2, 9, 14].

# Languages
# =========
# To provide a Java solution, edit Solution.java
# To provide a Python solution, edit solution.py

# Test cases
# ==========
# Your code should pass the following test cases.
# Note that it may also be run against hidden test cases not shown here.

# -- Java cases --
# Input:Solution.solution({
#     {0, 1, 0, 0, 0, 1}, 
#     {4, 0, 0, 3, 2, 0}, 
#     {0, 0, 0, 0, 0, 0}, 
#     {0, 0, 0, 0, 0, 0}, 
#     {0, 0, 0, 0, 0, 0}, 
#     {0, 0, 0, 0, 0, 0}
# })
# Output: [0, 3, 2, 9, 14]

# Input:Solution.solution({
#     {0, 2, 1, 0, 0}, 
#     {0, 0, 0, 3, 4}, 
#     {0, 0, 0, 0, 0},
#     {0, 0, 0, 0, 0},
#     {0, 0, 0, 0, 0}
# })
# Output: [7, 6, 8, 21]

# -- Python cases --
# Input:solution.solution([
#     [0, 2, 1, 0, 0],
#     [0, 0, 0, 3, 4],
#     [0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0]
# ])
# Output: [7, 6, 8, 21]

# Input:solution.solution([
#     [0, 1, 0, 0, 0, 1], 
#     [4, 0, 0, 3, 2, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0]
# ])
# Output: [0, 3, 2, 9, 14]
