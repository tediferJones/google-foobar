# The tree can be traversed backwards from any
# point to the head (1,1) by doing the following:
#   1. Get the min and max of M and F,
#   2. Subtract min from max until value of max is less than min
#      - this will also indicate the number of steps taken
#   3. Min is now less than max, swap the variables and repeat
# Once the min value is less than or equal to 1, we are done

def solution(M, F):
    M, F = int(M), int(F)
    hi, lo = max(M, F), min(M, F)
    count = 0

    while lo > 1:
        count += hi // lo
        hi = hi % lo
        hi, lo = lo, hi

    return str(count + hi - 1) if lo == 1 else 'impossible'

print(solution(1,1))
print(solution(1,2))
print(solution(3,2))
print(solution(3,5))
print(solution(8,5))
print(solution(2,4))
print(solution(7,79))


