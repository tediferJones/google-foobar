import itertools,copy
from fractions import Fraction
# itertools
# copy
# Fraction

def fixAbsorbingRows(m):
    for rowIndex in range(len(m)):
        if m[rowIndex] == [0] * len(m[rowIndex]):
            m[rowIndex][rowIndex] = 1

def getNewMatrixFormat(m):
    absorbing = []
    transient = []
    for rowIndex in range(len(m)):
        remainingRow = m[rowIndex][ : rowIndex] + m[rowIndex][rowIndex + 1 : ]
        if remainingRow == [0] * (len(m[rowIndex]) - 1):
        # if m[rowIndex] == [0] * len(m[rowIndex]):
            absorbing.append(rowIndex)
        else:
            transient.append(rowIndex)
    # print('TRANSIENT STATES')
    # print(transient)
    if len(absorbing) == 1:
        return 'CHEAT'
    return absorbing + transient

def convertMatrix(m, orderArr):
    result = []
    for rowNum in orderArr:
        newRow = []
        for rowIndex in orderArr:
            newRow.append(m[rowNum][rowIndex])
        result.append(newRow)
    return result

########
def getGCD(a, b):
    while b:
        a, b = b, a%b
    return a
# print(getGCD(12,16))

def simplifyFrac(frac):
    if frac.numerator == 0:
        return Fraction(0, 1)

    if frac.numerator == frac.denominator:
        return Fraction(1, 1)

    gcd = getGCD(frac.numerator, frac.denominator)
    return Fraction(frac.numerator // gcd, frac.denominator // gcd)
########

# class Fraction:
#     def __init__(self, numerator, denominator):
#         self.numerator = numerator
#         self.denominator = denominator
# 
#     def __str__(self):
#         return str(self.numerator) + '/' + str(self.denominator)
# 
#     def __add__(self, otherFrac):
#         a = self
#         b = otherFrac
#         if isinstance(otherFrac, Fraction):
#             if a.denominator == b.denominator:
#                 return Fraction(a.numerator + b.numerator, a.denominator)
#             else:
#                 newDenominator = a.denominator * b.denominator
#                 newNumeratorA = a.numerator * b.denominator
#                 newNumeratorB = b.numerator * a.denominator
#                 return simplifyFrac(Fraction(newNumeratorA + newNumeratorB, newDenominator))
#         else: 
#             raise TypeError("Unsupported operand type for FRAC ADDITION")
# 
#     def __radd__(self, other):
#         if isinstance(other, int):
#             return self.__add__(Fraction(other, 1))
#         else: 
#             raise TypeError("Unsupported operand type for FRAC ADD BETWEEN TYPES ONLY SUPPORTS INTS")
# 
#     def __sub__(self, otherFrac):
#         if isinstance(otherFrac, Fraction):
#             a = self
#             b = otherFrac
#             if a.denominator == b.denominator:
#                 return Fraction(a.numerator - b.numerator, a.denominator)
#             else:
#                 newDenominator = a.denominator * b.denominator
#                 newNumeratorA = a.numerator * b.denominator
#                 newNumeratorB = b.numerator * a.denominator
#                 return simplifyFrac(Fraction(newNumeratorA - newNumeratorB, newDenominator))
#         else: 
#             raise TypeError("Unsupported operand type for FRAC SUBTRACTION")
# 
#     def __mul__(self, otherFrac):
#         if isinstance(otherFrac, Fraction):
#             a = self
#             b = otherFrac
#             return simplifyFrac(Fraction(a.numerator * b.numerator, a.denominator * b.denominator))
#         elif isinstance(otherFrac, int):
#             a = self
#             b = Fraction(otherFrac, 1)
#             return simplifyFrac(Fraction(a.numerator * b.numerator, a.denominator * b.denominator))
#         else: 
#             raise TypeError("Unsupported operand type for FRAC MULT")
#     
#     def __rmul__(self, other):
#         if isinstance(other, int):
#             return self.__mul__(Fraction(other, 1))
#         else: 
#             raise TypeError("Unsupported operand type for FRAC MULT BETWEEN TYPES ONLY SUPPORTS INTS")
# 
#     def __div__(self, otherFrac):
#         if isinstance(otherFrac, Fraction):
#             a = self
#             b = otherFrac
#             return simplifyFrac(Fraction(a.numerator * b.denominator, a.denominator * b.numerator))
#         else:
#             raise TypeError("Unsupported operand type for FRAC DIVISION")
# 
#     def __rdiv__(self, other):
#         if isinstance(other, int):
#             return self.__div__(Fraction(other, 1))
#         else:
#             raise TypeError("Unsupported operand type for FRAC DIVISION BETWEEN TYPE ONLY SUPPORTS INTS")
# 
#     # def getFloat(self):
#     #     return self.numerator / self.denominator
# 
#     # def getString(self):
#     #     return str(self.numerator) + '/' + str(self.denominator)
# # print(Fraction(1,6) + Fraction(1,2))
# # print(Fraction(1,6) - Fraction(3,6))
# # print(Fraction(4,5) - Fraction(12,10))
# # print(Fraction(1,2) * Fraction(8,1))
# # print(Fraction(1,2) * 4)
# # print(4 * Fraction(1,2))

def convertToFracs(m):
    # Consider making this a pure function
    result = []
    for rowIndex in range(len(m)):
        denominator = sum(m[rowIndex])
        result.append([])
        for cellIndex in range(len(m[rowIndex])):
            result[rowIndex].append(Fraction(m[rowIndex][cellIndex], denominator))

    return result

def getRQ(m):
    result = []
    # absorbingState = [0] * len(m)
    # Get rid of all absorbing states
    for rowIndex in range(len(m)):
        cell = m[rowIndex][rowIndex]
        # if cell.numerator != 1 and cell.denominator != 1:
        # if not (cell.numerator == 1 and cell.denominator == 1):
        if cell.numerator != cell.denominator:
            # print(rowIndex)
            # print(cell)
            result.append(m[rowIndex])

    r = []
    q = []
    absorbingCount = len(m) - len(result)
    # print('NUMBER OF ABSORBING CASES')
    # print(absorbingCount)
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
            cell = Fraction(1, 1) if rowIndex == cellIndex else Fraction(0, 1)
            idMatrix[rowIndex].append(cell)
    return idMatrix

def subtractFracs(a, b):
    # print(a, b)
    if a.denominator == b.denominator:
        return Fraction(a.numerator - b.numerator, a.denominator)
    else:
        newDenominator = a.denominator * b.denominator
        newNumeratorA = a.numerator * b.denominator
        newNumeratorB = b.numerator * a.denominator
        return simplifyFrac(Fraction(newNumeratorA - newNumeratorB, newDenominator))

def subtractMatrices(m1, m2):
    result = []
    for rowIndex in range(len(m1)):
        result.append([])
        for cellIndex in range(len(m1[rowIndex])):
            result[rowIndex].append(
                subtractFracs(m1[rowIndex][cellIndex], m2[rowIndex][cellIndex])
            )
    return result

def addFracs(a, b):
    if a.denominator == b.denominator:
        return Fraction(a.numerator + b.numerator, a.denominator)
    else:
        newDenominator = a.denominator * b.denominator
        newNumeratorA = a.numerator * b.denominator
        newNumeratorB = b.numerator * a.denominator
        return simplifyFrac(Fraction(newNumeratorA + newNumeratorB, newDenominator))

def multiplyFracs(a, b):
    return simplifyFrac(Fraction(a.numerator * b.numerator, a.denominator * b.denominator))

def multiplyMatrices(m1, m2):
    # Dimensions of the final matrix can be found by doing:
    # m1 row Count by m2 Col count
    # result = [[Fraction(0,1) for x in range(3)] for y in range(3)]
    # result = [[Fraction(0,1) for x in range(len(m1))] for y in range(len(m2[0]))]
    result = [[Fraction(0,1) for x in range(len(m2[0]))] for y in range(len(m1))]
 
    for i in range(len(m1)):
        for j in range(len(m2[0])):
            for k in range(len(m2)):
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
    # m^-1 = 1 / (a*d - b*c) * [
    #                             [a, -b],
    #                             [-c, d],
    #                          ]
    result = []
    a = m[0][0]
    b = m[0][1]
    c = m[1][0]
    d = m[1][1]
    denominator = subtractFracs(multiplyFracs(a,d), multiplyFracs(b,c))
    # scalar = multiplyFracs(Fraction(1,1), denominator)
    scalar = multiplyFracs(Fraction(1,1), Fraction(denominator.denominator, denominator.numerator))
    # print('SCALER VALUE')
    # print(scalar.getString())
    # Flip the signs for b and c in given matrix
    m[0][1] = Fraction(m[0][1].numerator * -1, m[0][1].denominator)
    m[1][0] = Fraction(m[1][0].numerator * -1, m[1][0].denominator)

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
            if isinstance(m[rowIndex][cellIndex], Fraction):
                # print('found a fakeFrac')
                row.append(m[rowIndex][cellIndex].__str__())
            else:
                row.append(m[rowIndex][cellIndex])
        print(row)

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
            answer.append(frac.numerator * factor)
    answer.append(highestDenominator)
    return answer

####################################################################
def transposeMatrix(m):
    return map(list,zip(*m))

def getMatrixMinor(m,i,j):
    return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]

def getMatrixDeternminant(m):
    #base case for 2x2 matrix
    if len(m) == 2:
        return m[0][0]*m[1][1]-m[0][1]*m[1][0]

    determinant = 0
    for c in range(len(m)):
        determinant += ((-1)**c)*m[0][c]*getMatrixDeternminant(getMatrixMinor(m,0,c))
    return determinant

def getMatrixInverse(m):
    determinant = getMatrixDeternminant(m)
    #special case for 2x2 matrix:
    if len(m) == 2:
        return [[m[1][1]/determinant, -1*m[0][1]/determinant],
                [-1*m[1][0]/determinant, m[0][0]/determinant]]

    #find matrix of cofactors
    cofactors = []
    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor = getMatrixMinor(m,r,c)
            cofactorRow.append(((-1)**(r+c)) * getMatrixDeternminant(minor))
        cofactors.append(cofactorRow)
    cofactors = transposeMatrix(cofactors)
    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c]/determinant
    return cofactors
####################################################################

# NEW CHANGES
def getUnreachableRows(m):
    allRows = []
    for rowIndex in range(len(m)):
        allRows.append(rowIndex)
    rows = [0]

    for cellIndex in range(len(m[0])):
        if m[0][cellIndex] != 0:
            rows.append(cellIndex)
    # for rowIndex 
    # print("ROWS")
    # print(rows)
    # temp = allRow
    # for rowIndex in allRows:
    for rowIndex in rows:
        # print('CurrentRow')
        # print(m[rowIndex])
        for cellIndex in range(len(m[0])):
            # print(rowIndex, cellIndex)
            if m[rowIndex][cellIndex] != 0 and cellIndex not in rows:
                # reachableRows.append(cellIndex)
                # print('ADDING')
                # print(cellIndex)
                rows.append(cellIndex)

    # print(rows)
    for rowIndex in allRows:
        if m[rowIndex] == [0] * len(m[0]) and rowIndex not in rows:
            # print('FOUND AN ABSORBING CASE')
            # print(rowIndex)
            rows.append(rowIndex)

    # print(rows)
    # print(allRows)
    # print(allRows - rows)
    test = []
    for row in allRows:
        if row not in rows:
            test.append(row)
    # print(test)

    # print(reachableRows)
    # return rows
    return test
# print(getUnreachableRows([
#     [0,1,0,0],
#     [0,0,0,1],
#     [0,0,1,0],
#     [0,0,0,0],
# ]))

def fixUnreachableRows(m, rowsToRemove):
    for rowIndex in range(len(m)):
        # if rowIndex in rowsToRemove:
        #     del m[rowIndex]
        for rowToRemove in rowsToRemove:
            del m[rowIndex][rowToRemove]
    for rowToRemove in rowsToRemove:
        del m[rowToRemove]
    # print(m)
    return m
# fixUnreachableRows([
#     [1,1,0],
#     [1,1,0],
#     [0,0,0],
# ], [1])


# WRITE A CATCH ALL, IF THERE IS ONLY ONE ABSORBING STATE, just return [1,1]

def solution(m):
    # Your code here
    # Fix for if s0 is an absorbing state
    if m[0] == [0] * len(m):
        absorbingCount = 0
        # print('FIRST CASE IS THE ONLY POSSIBILITY')
        for rowIndex in range(len(m)):
            if m[rowIndex] == [0] * len(m[rowIndex]):
                absorbingCount += 1
        # print(absorbingCount)
        result = [1]
        for absorbingCase in range(absorbingCount - 1):
            result.append(0)
        result.append(1)
        return result

# NEW CHANGES
    unreachableRows = getUnreachableRows(m)
    m = fixUnreachableRows(m, unreachableRows)
    
    matrixFormat = getNewMatrixFormat(m)
    # print(matrixFormat)
    if matrixFormat == 'CHEAT':
        return [1,1]
    fixAbsorbingRows(m)
    test = convertMatrix(m, matrixFormat)
    fracs = convertToFracs(test)
    r, q = getRQ(fracs)
    # ppm(r)
    # ppm(q)
    if len(r) == 1:
        # print('SHORTCUT')
        # print(formatAnswer(r[0]))
        return formatAnswer(r[0])
    matrixToInvert = subtractMatrices(generateIdMatrix(len(q)), q)
    # invertedMatrix = invert2x2(matrixToInvert)
    invertedMatrix = getMatrixInverse(matrixToInvert)
    fr = multiplyMatrices(invertedMatrix, r)
    answers = formatAnswer(fr[0])
    # print(answers)
    return answers

def printSolution(m):
    # rq = getRQ(m)
    # ppm(rq)
    print('Original Matrix')
    ppm(m)
    print('################')
    print('Get new matrix ordering')
# NEW CHANGES
#     unreachableRows = getUnreachableRows(m)
#     m = fixUnreachableRows(m, unreachableRows)
    # ppm(m)
    print('WE ARE HERE')
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
    print('THIS IS R')
    ppm(r)
    print('THIS IS Q')
    ppm(q)
#     if len(r) == 1:
#         print('SHORTCUT')
#         # print(formatAnswer(r[0]))
#         return formatAnswer(r[0])
    # print('################')
    # print('Generate a generic identity matrix')
    # ppm(generateIdMatrix(4))
    # WE HAVE R AND Q, NOW DO Q - I
    print('################')
    print('Subtract I from Q')
    matrixToInvert = subtractMatrices(generateIdMatrix(len(q)), q)
    ppm(matrixToInvert)
    print('################')
    print('Get inverse of I - Q, i.e F')
    # invertedMatrix = invert2x2(matrixToInvert)
    invertedMatrix = getMatrixInverse(matrixToInvert)
    # invertedMatrix = inv_by_gauss_jordan(matrixToInvert)
    ppm(invertedMatrix)
    # print('################')
    # invertedMatrixV2 = removeFromSimpleArray(invertedMatrix)
    # print('REMOVE SIMPLE ARRAY')
    # ppm(invertedMatrixV2)
    print('################')
    print('GET FR')
    print('INVERTED MATRIX. i.e. matrix F')
    ppm(invertedMatrix)
    print('R')
    ppm(r)
    # fr = multiplyMatrices(invertedMatrixV2, r)
    fr = multiplyMatrices(invertedMatrix, r)
    print('FR')
    ppm(fr)
    print('################')
    print('THE ANSWERS')
    result = fr[0]
    ppm([result])
    print('################')
    print('Format answers into correct format')
    answers = formatAnswer(result)
    print(answers)


# print(solution([
#     [0,0,0,0,0],
#     [0,0,0,0,0],
#     [0,0,0,0,0],
#     [0,0,0,0,0],
#     [0,0,0,0,1],
# ]))

# print(printSolution([
# print(solution([[0]]))

# THIS ONE DOESNT WORK, but also doesnt seem to fix anything
# print(printSolution([
#     [0, 0, 0],
#     [0, 1, 0],
#     [0, 0, 1],
# ]))

print(solution([[0]]))
print(solution([
    [0, 2, 1, 0, 0],
    [0, 0, 0, 3, 4],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]))
print(solution([
    [0, 1, 0, 0, 0, 1], 
    [4, 0, 0, 3, 2, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]))

print(solution([
    [0,0,0,0,0],
    [0,1,0,0,0],
    [0,0,1,0,0],
    [0,0,0,0,0],
    [0,0,0,0,1],
]))
# print(solution([
#     [0, 0, 0, 0, 0, 0],
#     [0, 1, 0, 0, 0, 1],
#     [0, 0, 0, 0, 0, 0],
#     [4, 0, 0, 3, 2, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
# ]))

