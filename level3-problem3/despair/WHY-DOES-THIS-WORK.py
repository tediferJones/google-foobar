# import itertools,copy
from fractions import Fraction

def fixAbsorbingRows(m):
    for rowIndex in range(len(m)):
        if m[rowIndex] == [0] * len(m[rowIndex]):
            m[rowIndex][rowIndex] = 1

def getNewMatrixFormat(m):
    # ppm(m)
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
    # print(m)
    result = []
    for rowIndex in range(len(m)):
        denominator = float(sum(m[rowIndex]))
        # print(denominator)
        result.append([])
        for cellIndex in range(len(m[rowIndex])):
            # result[rowIndex].append(Fraction(m[rowIndex][cellIndex], denominator))
            # SWITCH TO DECIMALS
            # print('CELL VALUE')
            # if m[rowIndex][cellIndex] != 0:
            #     print('FOUND NONE ZERO')
            #     # print(type(m[rowIndex][cellIndex]))
            #     # print('1 / 3 is:')
            #     # print(1/3.0)
            #     print(m[rowIndex][cellIndex] / denominator)

            result[rowIndex].append(m[rowIndex][cellIndex] / denominator)
    # print(result)

    return result

def getRQ(m):
    # print('IN getRQ')
    # ppm(m)
    result = []
    # absorbingState = [0] * len(m)
    # Get rid of all absorbing states
    for rowIndex in range(len(m)):
        cell = m[rowIndex][rowIndex]
        # if cell.numerator != 1 and cell.denominator != 1:
        # if not (cell.numerator == 1 and cell.denominator == 1):
        # if cell.numerator != cell.denominator:
        if cell != 1.0:
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

def formatAnswerV2(arr):
    result = []
    highestDenominator = 0
    for decimal in arr:
        frac = Fraction(decimal).limit_denominator()

        if frac.denominator > highestDenominator:
            highestDenominator = frac.denominator

        result.append(frac)

    answer = []
    for frac in result:
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

def gcd(a ,b):
    if b==0:
        return a
    else:
        return gcd(b,a%b)   

def sanitize(M):
    needed = M[0]
    to_fraction = [Fraction(i).limit_denominator() for i in needed]
    lcm = 1
    for i in to_fraction:
        if i.denominator != 1:
            lcm = i.denominator
    for i in to_fraction:
        if i.denominator != 1:
            lcm = lcm*i.denominator/gcd(lcm, i.denominator)
    to_fraction = [(i*lcm).numerator for i in to_fraction]
    to_fraction.append(lcm)
    return to_fraction

def formatAnswerV2(arr):
    result = []
    for decimal in arr:
        frac = Fraction(decimal).limit_denominator()

        result.append(frac)

    getLcd = [] 
    for frac in result:
        getLcd.append(frac)
    while len(getLcd) > 1:
        getLcd[0] = gcd(getLcd[0], getLcd[1])
        del getLcd[1]

    commonDenominator = getLcd[0].denominator
    # print(commonDenominator)

    answer = []
    for frac in result:
        answer.append(frac.numerator * (commonDenominator // frac.denominator))
    answer.append(commonDenominator)

    return answer




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


# NEW CHANGES
#     unreachableRows = getUnreachableRows(m)
#     m = fixUnreachableRows(m, unreachableRows)

    matrixFormat = getNewMatrixFormat(m)
    # print('MATRIX FORMAT')
    # print(matrixFormat)
    # print(matrixFormat)
    fixAbsorbingRows(m)
    test = convertMatrix(m, matrixFormat)
    # ppm(test)
    # print('##################')
    fracs = convertToFracs(test)
    # ppm(fracs)
    r, q = getRQ(fracs)

    # print("THIS IS R")
    # ppm(r)
    # print("THIS IS Q")
    # ppm(q)

    # if len(r) == 1:
    #     # print('SHORTCUT')
    #     # print(formatAnswer(r[0]))
    #     return formatAnswer(r[0])

    matrixToInvert = subtractMatrices(generateIdMatrix(len(q)), q)
    # invertedMatrix = invert2x2(matrixToInvert)
    invertedMatrix = getMatrixInverse(matrixToInvert)
    # invertedMatrix = inv_by_gauss_jordan(matrixToInvert)
    # invertedMatrixV2 = removeFromSimpleArray(invertedMatrix)
    # print('REMOVE SIMPLE ARRAY')
    # ppm(invertedMatrixV2)
    fr = multiplyMatrices(invertedMatrix, r)
    # fr = multiplyMatrices(invertedMatrixV2, r)
    # print('ANSWERS')
    # print(fr)
    # answers = formatAnswer(fr[0])
    # print('THIS IS FR[0]')
    # print(fr[0])
    answers = formatAnswerV2(fr[0])
    # answers = sanitize(fr)
    # print(answers)
    return answers


