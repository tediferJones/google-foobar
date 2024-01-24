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

########
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
########

class FakeFrac:
    def __init__(self, numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator

    def __str__(self):
        return str(self.numerator) + '/' + str(self.denominator)

    def __add__(self, otherFrac):
        a = self
        b = otherFrac
        if isinstance(otherFrac, FakeFrac):
            if a.denominator == b.denominator:
                return FakeFrac(a.numerator + b.numerator, a.denominator)
            else:
                newDenominator = a.denominator * b.denominator
                newNumeratorA = a.numerator * b.denominator
                newNumeratorB = b.numerator * a.denominator
                return simplifyFrac(FakeFrac(newNumeratorA + newNumeratorB, newDenominator))
        else: 
            raise TypeError("Unsupported operand type for FRAC ADDITION")

    def __radd__(self, other):
        if isinstance(other, int):
            return self.__add__(FakeFrac(other, 1))
        else: 
            raise TypeError("Unsupported operand type for FRAC ADD BETWEEN TYPES ONLY SUPPORTS INTS")

    def __sub__(self, otherFrac):
        if isinstance(otherFrac, FakeFrac):
            a = self
            b = otherFrac
            if a.denominator == b.denominator:
                return FakeFrac(a.numerator - b.numerator, a.denominator)
            else:
                newDenominator = a.denominator * b.denominator
                newNumeratorA = a.numerator * b.denominator
                newNumeratorB = b.numerator * a.denominator
                return simplifyFrac(FakeFrac(newNumeratorA - newNumeratorB, newDenominator))
        else: 
            raise TypeError("Unsupported operand type for FRAC SUBTRACTION")

    def __mul__(self, otherFrac):
        if isinstance(otherFrac, FakeFrac):
            a = self
            b = otherFrac
            return simplifyFrac(FakeFrac(a.numerator * b.numerator, a.denominator * b.denominator))
        elif isinstance(otherFrac, int):
            a = self
            b = FakeFrac(otherFrac, 1)
            return simplifyFrac(FakeFrac(a.numerator * b.numerator, a.denominator * b.denominator))
        else: 
            raise TypeError("Unsupported operand type for FRAC MULT")
    
    def __rmul__(self, other):
        if isinstance(other, int):
            return self.__mul__(FakeFrac(other, 1))
        else: 
            print(type(other))
            raise TypeError("Unsupported operand type for FRAC MULT BETWEEN TYPES ONLY SUPPORTS INTS")

    def __div__(self, otherFrac):
        if isinstance(otherFrac, FakeFrac):
            a = self
            b = otherFrac
            return simplifyFrac(FakeFrac(a.numerator * b.denominator, a.denominator * b.numerator))
        if isinstance(otherFrac, int):
            # otherFrac = FakeFrac(otherFrac, 1)
            # return simplifyFrac(FakeFrac(self.))
            a = self
            b = FakeFrac(otherFrac, 1)
            return simplifyFrac(FakeFrac(a.numerator * b.denominator, a.denominator * b.numerator))

        else:
            print(type(otherFrac))
            raise TypeError("Unsupported operand type for FRAC DIVISION")

    def __rdiv__(self, other):
        if isinstance(other, int):
            return self.__div__(FakeFrac(other, 1))
        else:
            raise TypeError("Unsupported operand type for FRAC DIVISION BETWEEN TYPE ONLY SUPPORTS INTS")

    def __abs__(self):
        newNum = self.numerator
        newDen = self.denominator
        if newNum < 0:
            newNum *= -1
        if newDen < 0:
            newDen *= -1
        return FakeFrac(newNum, newDen)

    # def __lt__(self, other):
    #     print('USING LESS THAN ON FRACTION')



    # def getFloat(self):
    #     return self.numerator / self.denominator

    # def getString(self):
    #     return str(self.numerator) + '/' + str(self.denominator)
# print(FakeFrac(1,6) + FakeFrac(1,2))
# print(FakeFrac(1,6) - FakeFrac(3,6))
# print(FakeFrac(4,5) - FakeFrac(12,10))
# print(FakeFrac(1,2) * FakeFrac(8,1))
# print(FakeFrac(1,2) * 4)
# print(4 * FakeFrac(1,2))
# print(abs(FakeFrac(1,-2)))

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
            cell = FakeFrac(1, 1) if rowIndex == cellIndex else FakeFrac(0, 1)
            idMatrix[rowIndex].append(cell)
    return idMatrix

def subtractFracs(a, b):
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
                subtractFracs(m1[rowIndex][cellIndex], m2[rowIndex][cellIndex])
            )
    return result

def addFracs(a, b):
    if a.denominator == b.denominator:
        return simplifyFrac(FakeFrac(a.numerator + b.numerator, a.denominator))
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
    # scalar = multiplyFracs(FakeFrac(1,1), denominator)
    scalar = multiplyFracs(FakeFrac(1,1), FakeFrac(denominator.denominator, denominator.numerator))
    # print('SCALER VALUE')
    # print(scalar.getString())
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
            if isinstance(m[rowIndex][cellIndex], FakeFrac):
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

def eliminate(r1, r2, col, target=0):
    fac = (r2[col]-target) / r1[col]
    for i in range(len(r2)):
        r2[i] -= fac * r1[i]

def gauss(a):
    for i in range(len(a)):
        if a[i][i] == 0:
            for j in range(i+1, len(a)):
                if a[i][j] != 0:
                    a[i], a[j] = a[j], a[i]
                    break
            else:
                raise ValueError("Matrix is not invertible")
        for j in range(i+1, len(a)):
            eliminate(a[i], a[j], i)
    for i in range(len(a)-1, -1, -1):
        for j in range(i-1, -1, -1):
            eliminate(a[i], a[j], i)
    for i in range(len(a)):
        eliminate(a[i], a[i], i, target=1)
    return a

def inverse(a):
    tmp = [[] for _ in a]
    for i,row in enumerate(a):
        assert len(row) == len(a)
        tmp[i].extend(row + [0]*i + [1] + [0]*(len(a)-i-1))
    gauss(tmp)
    ret = []
    for i in range(len(tmp)):
        ret.append(tmp[i][len(tmp[i])//2:])
    return ret

####################################################################

def inverse_matrix(matrix):
    # Check if the matrix is square
    if len(matrix) != len(matrix[0]):
        raise ValueError("Input matrix must be square")

    n = len(matrix)

    # Initialize the identity matrix
    identity_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        identity_matrix[i][i] = 1

    # Perform Gaussian elimination to calculate the inverse
    for col in range(n):
        # Find the pivot row
        pivot_row = col
        for row in range(col + 1, n):
            if abs(matrix[row][col]) > abs(matrix[pivot_row][col]):
                pivot_row = row

        # Swap rows to make the pivot element non-zero
        matrix[col], matrix[pivot_row] = matrix[pivot_row], matrix[col]
        identity_matrix[col], identity_matrix[pivot_row] = identity_matrix[pivot_row], identity_matrix[col]

        # Scale the row to have a 1 in the pivot position
        pivot_element = matrix[col][col]
        if pivot_element == 0:
            raise ValueError("Matrix is singular and cannot be inverted")
        for j in range(n):
            matrix[col][j] /= pivot_element
            identity_matrix[col][j] /= pivot_element

        # Eliminate other rows
        for i in range(n):
            if i != col:
                factor = matrix[i][col]
                for j in range(n):
                    matrix[i][j] -= factor * matrix[col][j]
                    identity_matrix[i][j] -= factor * identity_matrix[col][j]

    return identity_matrix

####################################################################

import itertools,copy

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
	# total2 = 0.0
	total2 = FakeFrac(0,1)
	for perm in itertools.permutations(range(n)):
        # total1 = FakeFrac(1,1)
		# total1 = 1.0
		total1 = FakeFrac(1,1)
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
		# mat.data[i][i] = 1.0
		mat.data[i][i] = FakeFrac(1,1)
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
					# l.append(0.0)
					l.append(FakeFrac(0,1))
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
	mdetCompareValue = abs(mdet.numerator / mdet.denominator)
	# print(mdetCompareValue)
	# if abs(mdet) < eps:
	if mdetCompareValue < eps:
		# print('ppop')
		# print(mdet)
		# print(eps)
		# print(abs(mdet) < eps)
		raise RuntimeError("Matrix is not invertible (its determinant is zero)")

	#Create aux matrix
	n = mat.shape[0]
	# print(n)
	auxmat = identity(n)
	# auxmat = generateIdMatrix(n)
	# print(auxmat)

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
			# scale = float(mat.data[r][i]) / float(mat.data[i][i])
			# print(mat.data[r][i], mat.data[i][i])
			scale = mat.data[r][i] / mat.data[i][i]
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




# IT IS POSSIBLE THAT THERE IS ONLY ONE TERMINAL CASE BUT MULTIPLE WAYS TO GET THERE

# IF WE CAN CREATE FUNCTIONS TO OVERRIDE +,-,*,/ we can use normal math on these fractions
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
    # invertedMatrix = getMatrixInverse(matrixToInvert)
    invertedMatrix = inv_by_gauss_jordan(matrixToInvert)
    invertedMatrixV2 = removeFromSimpleArray(invertedMatrix)
    print('REMOVE SIMPLE ARRAY')
    ppm(invertedMatrixV2)
    fr = multiplyMatrices(invertedMatrixV2, r)
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
    # invertedMatrix = getMatrixInverse(matrixToInvert)
    invertedMatrix = inv_by_gauss_jordan(matrixToInvert)
    ppm(invertedMatrix)
    print('################')
    invertedMatrixV2 = removeFromSimpleArray(invertedMatrix)
    print('REMOVE SIMPLE ARRAY')
    ppm(invertedMatrixV2)
    print('################')
    print('GET FR')
    print('INVERTED MATRIX. i.e. matrix F')
    ppm(invertedMatrix)
    print('R')
    ppm(r)
    fr = multiplyMatrices(invertedMatrixV2, r)
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

test = inv_by_gauss_jordan([
# test = invert2x2([
#     [1, 2, -1],
#     [2, 1, 2],
#     [-1, 2, 1],
#     [FakeFrac(1, 1), FakeFrac(2, 1), FakeFrac(-1, 1)],
#     [FakeFrac(2, 1), FakeFrac(1, 1), FakeFrac(2, 1)],
#     [FakeFrac(-1, 1), FakeFrac(2, 1), FakeFrac(1, 1)],
#     [FakeFrac(1, 1), FakeFrac(-2, 3)],
#     [FakeFrac(0, 1), FakeFrac(1, 1)],
    [FakeFrac(1,1), FakeFrac(2,1), FakeFrac(3,1)],
    [FakeFrac(0,1), FakeFrac(1,1), FakeFrac(4,1)],
    [FakeFrac(5,1), FakeFrac(6,1), FakeFrac(0,1)],
])
# print(test.shape)
ppm(test)
# print(inv_by_gauss_jordan([
#     [FakeFrac(1, 1), FakeFrac(2, 1), FakeFrac(-1, 1)],
#     [FakeFrac(2, 1), FakeFrac(1, 1), FakeFrac(2, 1)],
#     [FakeFrac(-1, 1), FakeFrac(2, 1), FakeFrac(1, 1)],
# ]))

# ppm(inverse_matrix([
#     [1, 2, -1],
#     [2, 1, 2],
#     [-1, 2, 1]
# ]))
# ppm(inverse_matrix([
#     [1, 2, -1],
#     [2, 1, 2],
#     [-1, 2, 1],
# ]))
# ppm(inverse([
#     [1, -4, 2],
#     [-2, 1, 3],
#     [2, 6, 8],
# ]))
# print(ppm(multiplyMatrices([
#     [FakeFrac(6, 1), FakeFrac(1, 1), FakeFrac(-1, 1)],
#     [FakeFrac(-2, 1), FakeFrac(-3, 1), FakeFrac(0, 1)]
# ], [
#     [FakeFrac(2, 1), FakeFrac(4, 1), FakeFrac(-1, 1), FakeFrac(6, 1)],
#     [FakeFrac(-4, 1), FakeFrac(1, 1), FakeFrac(3, 1), FakeFrac(-3, 1)],
#     [FakeFrac(5, 1), FakeFrac(-2, 1), FakeFrac(7, 1), FakeFrac(0, 1)],
# ])))
# solution([
#     [1,0],
#     [0,1]
# ])
# whst are the smallest matrix that still works

# Maybe the solution lies in how we handle "transient cases" that are unreachable

# print(solution([
#     [0, 0, 2, 0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#     [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
# ]))
# print(solutionV2([
#     [0, 0, 0, 0, 0, 0],
#     [0, 1, 0, 0, 0, 1],
#     [0, 0, 0, 0, 0, 0],
#     [4, 0, 0, 3, 2, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
# ]))
# print(solutionV2([
#     [0, 1],
#     [0, 0],
# ]))
# print(solution([
#     [0, 2, 1, 0, 0],
#     [0, 0, 0, 3, 4],
#     [0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0]
# ]))
# print(solution([
#     [0, 1, 0, 0, 0, 1], 
#     [4, 0, 0, 3, 2, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
# ]))
# print(solution([
#     [0, 1, 0, 0, 0, 1, 0, 0], 
#     [4, 0, 0, 3, 2, 0, 1, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 0, 0, 0, 0, 1, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 0, 0, 0, 0, 0, 1],
#     [0, 0, 0, 0, 0, 0, 0, 0],
# ]))

# TEST MATRIX INVERSE IS CORRECT
# test1 = [
#     [1, -4, 2],
#     [-2, 1, 3],
#     [2, 6, 8],
# ]
# print('ORIGINAL MATRIX')
# ppm(test1)
# print('INVERTED MATRIX')
# ppm(getMatrixInverse(test1))
# 
# test2 = [
#     [5, 6, 6, 8],
#     [2, 2, 2, 8],
#     [6, 6, 2, 8],
#     [2, 3, 6, 7],
# ]
# print('ORIGINAL MATRIX')
# ppm(test2)
# print('INVERTED MATRIX')
# ppm(getMatrixInverse(test2))


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
