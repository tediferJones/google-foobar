# This equation will sum numbers from a to a+d*n with matching step of d
# a = first term, d = radius (step), n = number of steps
#   n(2a + (n - 1) * d)
# = -------------------
#           2
# In our case d will always be 1

# The desired id number can found by finding the sum of 2 arithmetic sequences like so:
#
#       arthSeq(1, 1, x) + arthSeq(x, 1, y - 1),
#
# Instead of calling these functions manually we can just add these two equations together and simplify

# = arthSeq(1, 1, x) + arthSeq(x, 1, y - 1)
#
#   x(2 + x - 1)   (y - 1)(2x + y - 1 - 1)
# = ------------ + -----------------------
#         2                   2      
#
#   x^2 + x   y^2 + 2xy - 2y - 2x - y + 2
# = ------- + ---------------------------
#      2                  2      
#
#   y^2 + x^2 + 2xy - 3y - x + 2
# = ----------------------------
#                2

def solution(x, y):
    return str(int((y ** 2 + x ** 2 + 2 * x * y - 3 * y - x + 2) / 2))
print(solution(1, 1))
print(solution(3, 2))
print(solution(5, 10))
print(solution(100000, 100000))

# | 11
# | 7 12
# | 4 8 13
# | 2 5 9 14
# | 1 3 6 10 15


