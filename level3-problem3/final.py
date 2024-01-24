from fractions import Fraction

def fixAbsorbingRows(m):
    result = []
    for rowIndex in range(len(m)):
        result.append([])
        for cellIndex in range(len(m)):
            if rowIndex == cellIndex and m[rowIndex] == [0] * len(m[rowIndex]):
                result[rowIndex].append(1)
            else:
                result[rowIndex].append(m[rowIndex][cellIndex])
    return result

def getNewMatrixFormat(m):
    absorbing, transient = [], []
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
    result = []
    for rowIndex in range(len(m)):
        denominator = sum(m[rowIndex])
        result.append([])
        for cellIndex in range(len(m[rowIndex])):
            result[rowIndex].append(
                Fraction(m[rowIndex][cellIndex], denominator).limit_denominator()
            )
    return result

def generateIdMatrix(size):
    result = []
    for rowIndex in range(size):
        result.append([])
        for cellIndex in range(size):
            result[rowIndex].append(
                1 if rowIndex == cellIndex else 0
            )
    return result

def getRQ(m):
    r, q, transientStates = [], [], []
    for rowIndex in range(len(m)):
        if m[rowIndex][rowIndex] != 1.0:
            transientStates.append(m[rowIndex])

    absorbingCount = len(m) - len(transientStates)
    # Slice the transientStates to get r and q
    for rowIndex in range(len(transientStates)):
        r.append(transientStates[rowIndex][ : absorbingCount])
        q.append(transientStates[rowIndex][absorbingCount : ])
    return [r,q]

def subtractMatrices(m1, m2):
    result = []
    for rowIndex in range(len(m1)):
        result.append([])
        for cellIndex in range(len(m1[rowIndex])):
            result[rowIndex].append(
                m1[rowIndex][cellIndex] - m2[rowIndex][cellIndex]
            )
    return result

def multiplyMatrices(m1, m2):
    result = [[Fraction(0,1) for x in range(len(m2[0]))] for y in range(len(m1))]
    for i in range(len(m1)):
        for j in range(len(m2[0])):
            for k in range(len(m2)):
                # multiply the current selection together
                # then sum it with existing value
                result[i][j] += m1[i][k] * m2[k][j]
    return result

def gcd(a ,b):
    if b==0:
        return a
    else:
        return gcd(b,a%b)   

def formatAnswer(arr):
    lcd = Fraction(1,1)
    for i in range(len(arr)):
        lcd = gcd(lcd, arr[i])

    answer = []
    for frac in arr:
        answer.append(frac.numerator * (lcd.denominator // frac.denominator))
    answer.append(lcd.denominator)

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

# This problem can be represented as a markov chain
# To solve, we must 
#   1. Replace integer probabilities with fractions and 
#      fill in absorbing cases with a 1 in the appropiate column
#   2. Organize the matrix into IORQ format
#   3. Find the inverse of Q-I, which we'll call F
#   4. Multiply F by R,
#   5. Extract the probabilities from the row corresponding
#      to s0 and format them with a common denominator
#
# Example = [             Example IORQ = [    (without fractions)
#                                  I  I  I    O  O
#     [0, 2, 1, 0, 0],          I [1, 0, 0, | 0, 0] O
#     [0, 0, 0, 3, 4],          I [0, 1, 0, | 0, 0] O
#     [0, 0, 0, 0, 0],          I [0, 0, 1, | 0, 0] O
#     [0, 0, 0, 0, 0],            ----------+------
#     [0, 0, 0, 0, 0],          R [1, 0, 0, | 0, 2] Q
#                               R [0, 3, 4, | 0, 0] Q
# ]                       ]        R  R  R    Q  Q

def solution(m):
    # Your code here
    if m[0] == [0] * len(m):
        result = [1]
        for rowIndex in range(len(m)):
            if m[rowIndex] == [0] * len(m[rowIndex]):
                result.append(0)
        result[-1] = 1
        return result

    # Reformat the matrix into proper IORQ format, with fractions
    iorq = convertToFracs(
        convertMatrix(
            fixAbsorbingRows(m), getNewMatrixFormat(m)
        )
    )
    r, q = getRQ(iorq)

    # Get inverse of I-Q
    f = getMatrixInverse(
        subtractMatrices(generateIdMatrix(len(q)), q)
    )
    fr = multiplyMatrices(f, r)
    return formatAnswer(fr[0])



def ppm(m):
    for row in m:
        # print(row)
        result = []
        for cell in row:
            result.append(cell.__str__())
        print(result)

print(solution([
    [0,2,1,0,0],
    [0,0,0,3,4],
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0],
]))
