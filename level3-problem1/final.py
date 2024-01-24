# Only scan the possible second numbers (l[1 : -1])
# Get all factors before that number
# Get all multiples after that number
# The number of triples for the given second number = factors * multiples
# Add that number to count

def solution(l):
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
print(solution([1,1,1]))
print(solution([1,2,3,4,5,6]))
print(solution([1, 2, 3, 4, 5, 6, 7, 8, 9]))
print(solution([2, 3, 4, 5, 6, 7, 8, 9, 12]))
print(solution([2, 3, 4, 5, 6, 7, 8, 9, 12, 16]))
print(solution([2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 24, 48, 96]))
print(solution([1, 1, 1]))
print(solution([1] * 100))


