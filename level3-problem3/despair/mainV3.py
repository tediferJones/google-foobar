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



###############################################################################

def isiterable(data):
	try:
		chkit = iter(data)
	except TypeError, te:
		return False
	return True

def array(data):
	arr = SimpleArray()
	arr.shape = []
	if not isiterable(data):
		raise ValueError("Input must be iterable")
	arr._ValidateShape(data, 0)
	arr.data = data
	arr.shape = tuple(arr.shape)
	return arr

def det(mat):
	if not isinstance(mat, SimpleArray) and isiterable(mat):		
		mat = array(mat)
	if len(mat.shape) != 2:
		raise NotImplementedError("Only implemented for 2D matricies")
	if mat.shape[0] != mat.shape[1]:
		raise ValueError("Matrix must be square")
	n = mat.shape[0]
	
	#Leibniz formula for the determinant
	total2 = 0.0
	for perm in itertools.permutations(range(n)):
		total1 = 1.0
		for i, j in zip(range(n), perm):
			total1 *= mat.data[i][j]
		total2 += perm_parity(list(perm)) * total1
	return total2

def perm_parity(lst):
	'''\
	Given a permutation of the digits 0..N in order as a list, 
	returns its parity (or sign): +1 for even parity; -1 for odd.
	From https://code.activestate.com/recipes/578227-generate-the-parity-or-sign-of-a-permutation/
	'''
	parity = 1
	for i in range(0,len(lst)-1):
		if lst[i] != i:
			parity *= -1
			mn = min(range(i,len(lst)), key=lst.__getitem__)
			lst[i],lst[mn] = lst[mn],lst[i]
	return parity

def identity(dim):
	mat = zeros((dim, dim))
	for i in range(dim):
		mat.data[i][i] = 1.0
	return mat

def zeros(shape):
	arr = SimpleArray()
	arr.shape = tuple(shape)
	if not isiterable(shape):
		raise ValueError("Shape must be iterable")
	arr.data = []
	def GenData(shape, leafs):
		dim = shape.pop(0)
		newLeafs = []
		if len(shape) == 0:
			for l in leafs:
				for i in range(dim):
					l.append(0.0)
		else:
			for l in leafs:
				for i in range(dim):
					nl = []
					l.append(nl)
					newLeafs.append(nl)
			GenData(shape, newLeafs)
			
	GenData(list(arr.shape)[:], [arr.data])
	return arr

class SimpleArray(object):
	def __init__(self):
		self.shape = []
		self.data = None

	def _ValidateShape(self, data, ind):
		if not isiterable(data):
			return
		if ind >= len(self.shape):
			self.shape.append(len(data))
		else:
			if self.shape[ind] != len(data):
				raise ValueError("Inconsistent shape")
		
		for val in data:
			self._ValidateShape(val, ind+1)

	def __repr__(self):
		return (self.__class__.__name__+"("+str(self.data)+")")

	def dot(self, rhs):
		#This also standard matrix multiplication
		if not isinstance(rhs, self.__class__) and isiterable(rhs):		
			rhs = array(rhs)
		if len(self.shape) != 2 or len(rhs.shape) != 2:
			raise NotImplementedError("Only implemented for 2D matricies")
		if self.shape[1] != rhs.shape[0]:
			raise ValueError("Matrix size mismatch")
		m = self.shape[1]
		result = zeros((self.shape[0], rhs.shape[1]))
		for i in range(result.shape[0]):
			for j in range(result.shape[1]):
				tot = 0.0
				for k in range(m):
					aik = self.data[i][k]
					bkj = rhs.data[k][j]
					tot += aik * bkj
				result.data[i][j] = tot

		return result

	def __add__(self, rhs):
		if not isinstance(rhs, self.__class__) and isiterable(rhs):		
			rhs = array(rhs)
		if self.shape != rhs.shape:
			raise ValueError("Matrix size mismatch")
		result = zeros(self.shape)
		def AddFunc(ind, a, b, out):
			if ind == len(self.shape):
				raise ValueError("Invalid matrix")
			if ind == len(self.shape) - 1:
				for i, (av, bv) in enumerate(zip(a, b)):
					out[i] = av + bv
			else:
				ind += 1
				for ar, br, outr in zip(a, b, out):
					AddFunc(ind, ar, br, outr)		

		AddFunc(0, self.data, rhs.data, result.data)
		return result

	def __mul__(self, rhs):
		if not isinstance(rhs, self.__class__) and isiterable(rhs):		
			rhs = array(rhs)

		if isinstance(rhs, self.__class__):
			#Do element-wise matrix multiplication

			if self.shape != rhs.shape:
				raise ValueError("Matrix size mismatch")
			result = zeros(self.shape)
			def MultFunc(ind, a, b, out):
				if ind == len(self.shape):
					raise ValueError("Invalid matrix")
				if ind == len(self.shape) - 1:
					for i, (av, bv) in enumerate(zip(a, b)):
						out[i] = av * bv
				else:
					ind += 1
					for ar, br, outr in zip(a, b, out):
						MultFunc(ind, ar, br, outr)		

			MultFunc(0, self.data, rhs.data, result.data)
			return result
		else:
			#Do scalar multiplication to entire matrix (element-wise)
			result = zeros(self.shape)
			def MultFunc2(ind, a, b, out):
				if ind == len(self.shape):
					raise ValueError("Invalid matrix")
				if ind == len(self.shape) - 1:
					for i, av in enumerate(a):
						out[i] = av * b
				else:
					ind += 1
					for ar, outr in zip(a, out):
						MultFunc2(ind, ar, b, outr)		

			MultFunc2(0, self.data, rhs, result.data)
			return result

	def __sub__(self, rhs):
		if not isinstance(rhs, self.__class__) and isiterable(rhs):		
			rhs = array(rhs)
		if self.shape != rhs.shape:
			raise ValueError("Matrix size mismatch")
		negRhs = rhs * -1
		return self + negRhs

	def conj(self):
		result = zeros(self.shape)
		def AddFunc(ind, a, out):
			if ind == len(self.shape):
				raise ValueError("Invalid matrix")
			if ind == len(self.shape) - 1:
				for i, av in enumerate(a):
					if isinstance(av, complex):
						out[i] = av.conjugate()
					else:
						out[i] = av
			else:
				ind += 1
				for ar, outr in zip(a, out):
					AddFunc(ind, ar, outr)		

		AddFunc(0, self.data, result.data)
		return result

	@property
	def T(self):
		if len(self.shape) <= 1:
			return self
		if len(self.shape) != 2:
			raise NotImplementedError("Only implemented for 1D and 2D matricies")	
		result = zeros((self.shape[1], self.shape[0]))
		for i in range(self.shape[0]):
			for j in range(self.shape[1]):
				result.data[j][i] = self.data[i][j]
		return result

	def copy(self):
		return array(copy.deepcopy(self.data))



def inv_by_gauss_jordan(mat, eps=1e-8):
	#Find inverse based on gauss-jordan elimination.

	if not isinstance(mat, SimpleArray) and isiterable(mat):		
		mat = array(mat)
	if len(mat.shape) != 2:
		raise NotImplementedError("Only implemented for 2D matricies")
	if mat.shape[0] != mat.shape[1]:
		raise ValueError("Matrix must be square")
	mdet = det(mat)
	# print(mdet)
	# print(eps)
	# print(abs(mdet) < eps)
	if abs(mdet) < eps:
		raise RuntimeError("Matrix is not invertible (its determinant is zero)")

	#Create aux matrix
	n = mat.shape[0]
	auxmat = identity(n)

	#Convert to echelon (triangular) form
	mat = copy.deepcopy(mat)
	for i in range(n):
		#Find highest value in this column
		maxv = 0.0
		maxind = None
		for r in range(i, n):
			v = mat.data[r][i]
			if maxind is None or abs(v) > maxv:
				maxv = abs(v)
				maxind = r
		
		if maxind != i:
			#Swap this to the current row, for numerical stability
			mat.data[i], mat.data[maxind] = mat.data[maxind], mat.data[i]
			auxmat.data[i], auxmat.data[maxind] = auxmat.data[maxind], auxmat.data[i]

		activeRow = mat.data[i]
		activeAuxRow = auxmat.data[i]
		for r in range(i+1, n):
			scale = float(mat.data[r][i]) / float(mat.data[i][i])
			cursorRow = mat.data[r]
			cursorAuxRow = auxmat.data[r]
			for c in range(n):
				cursorRow[c] -= scale * activeRow[c]
				cursorAuxRow[c] -= scale * activeAuxRow[c]

	#Finish elimination
	for i in range(n-1, -1, -1):
		activeRow = mat.data[i]
		activeAuxRow = auxmat.data[i]
		for r in range(i, -1, -1):
			cursorRow = mat.data[r]
			cursorAuxRow = auxmat.data[r]
			if r == i:
				scaling = activeRow[i]
				for c in range(n):
					cursorRow[c] /= scaling
					cursorAuxRow[c] /= scaling
			else:
				scaling = cursorRow[i]
				for c in range(n):
					cursorRow[c] -= activeRow[c] * scaling
					cursorAuxRow[c] -= activeAuxRow[c] * scaling
			
	return auxmat

###############################################################################
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




def removeFromSimpleArray(m):
    if isinstance(m, SimpleArray):
            result = []
            for rowIndex in range(m.shape[0]):
                result.append([])
                for cellIndex in range(m.shape[1]):
                    result[rowIndex].append(m.data[rowIndex][cellIndex])
            # m = result
            return result
    else:
        return 'Not a simple array'

# print(Fraction(1,2))
# print(inv_by_gauss_jordan([
#     # [Fraction(1,1), Fraction(2,1), Fraction(-1,1)],
#     # [Fraction(2,1), Fraction(1,1), Fraction(2,1)],
#     # [Fraction(-1,1), Fraction(2,1), Fraction(1,1)],
#     [Fraction(1,1), Fraction(-2,3)],
#     [Fraction(0,1), Fraction(1,1)],
# ]))

def solutionV2(m):
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
    print('THIS IS R')
    ppm(r)
    print('THIS IS Q')
    ppm(q)
    if len(r) == 1:
        print('SHORTCUT')
        # print(formatAnswer(r[0]))
        return formatAnswer(r[0])
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

print(solutionV2([
    [0, 2, 1, 0, 0],
    [0, 0, 0, 3, 4],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]))
print(solutionV2([
    [0, 1, 0, 0, 0, 1], 
    [4, 0, 0, 3, 2, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]))
print(solutionV2([
    [0,0,0,0,0],
    [0,1,0,0,0],
    [0,0,1,0,0],
    [0,0,0,0,0],
    [0,0,0,0,1],
]))
print(solutionV2([
    [0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0],
    [4, 0, 0, 3, 2, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]))
# ppm(getMatrixInverse([
# #     [Fraction(1,1), Fraction(2,1), Fraction(-1,1)],
# #     [Fraction(2,1), Fraction(1,1), Fraction(2,1)],
# #     [Fraction(-1,1), Fraction(2,1), Fraction(1,1)],
#     [Fraction(1, 1), Fraction(-2, 3)],
#     [Fraction(0, 1), Fraction(1, 1)],
# ]))

