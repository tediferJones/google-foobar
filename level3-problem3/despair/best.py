# import itertools,copy
# from fractions import Fraction

import itertools,copy
from fractions import Fraction

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


def subtractMatrices(m1, m2):
    result = []
    for rowIndex in range(len(m1)):
        result.append([])
        for cellIndex in range(len(m1[rowIndex])):
            result[rowIndex].append(
                # subtractFracs(m1[rowIndex][cellIndex], m2[rowIndex][cellIndex])
                m1[rowIndex][cellIndex] - m2[rowIndex][cellIndex]
            )
    return result


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
                # product = multiplyFracs(m1[i][k], m2[k][j])
                product = m1[i][k] * m2[k][j]
                # result[i][j] = addFracs(result[i][j], product)
                result[i][j] += product

    return result


# Pretty Print Matrix
def ppm(m):
    # for row in m:
    #     print(row)
    if isinstance(m, SimpleArray):
        result = []
        for rowIndex in range(m.shape[0]):
            result.append([])
            for cellIndex in range(m.shape[1]):
                result[rowIndex].append(m.data[rowIndex][cellIndex])
        m = result

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




def solution(m):
    # Your code here
    # TESTING
    # If first row is absorbing case, thats the only possible outcome
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

        # return [1]

    # IF THERE IS ONLY ONE ABSORBING CASE, we can return [1,1]



    matrixFormat = getNewMatrixFormat(m)
    # print('MATRIX FORMAT')
    # print(matrixFormat)
    # print(matrixFormat)
    fixAbsorbingRows(m)
    test = convertMatrix(m, matrixFormat)
    fracs = convertToFracs(test)
    r, q = getRQ(fracs)
    if len(r) == 1:
        # print('SHORTCUT')
        # print(formatAnswer(r[0]))
        return formatAnswer(r[0])
    matrixToInvert = subtractMatrices(generateIdMatrix(len(q)), q)
    # invertedMatrix = invert2x2(matrixToInvert)
    invertedMatrix = getMatrixInverse(matrixToInvert)
    # invertedMatrix = inv_by_gauss_jordan(matrixToInvert)
    # invertedMatrixV2 = removeFromSimpleArray(invertedMatrix)
    # print('REMOVE SIMPLE ARRAY')
    # ppm(invertedMatrixV2)
    fr = multiplyMatrices(invertedMatrix, r)
    # fr = multiplyMatrices(invertedMatrixV2, r)
    answers = formatAnswer(fr[0])
    # print(answers)
    return answers


