# Simplified problem:
# Given the board below, determine the shortest path a knight can take between to numbers
# (its just like the knights travails problem from TOP)
# Example: https://github.com/tediferJones/odin-knights-travails

# -------------------------
# | 0| 1| 2| 3| 4| 5| 6| 7|
# -------------------------
# | 8| 9|10|11|12|13|14|15|
# -------------------------
# |16|17|18|19|20|21|22|23|
# -------------------------
# |24|25|26|27|28|29|30|31|
# -------------------------
# |32|33|34|35|36|37|38|39|
# -------------------------
# |40|41|42|43|44|45|46|47|
# -------------------------
# |48|49|50|51|52|53|54|55|
# -------------------------
# |56|57|58|59|60|61|62|63|
# -------------------------

# IT IS POSSIBLE TO OPERATE DIRECTLY ON THE NUMBERS
# No its not, what happen if you move down/right from 31

# Step 1: convert numbers to coordinates
# Step 2: create a tree structure that will run all 8 possible moves the current square
#           - This needs to be a breadth first search
# Step 3: keep running the tree structure till we hit dest coordinate
# Step 4: return the number of steps

# Special Case handler in python
# Javascript:
# x = {
#   func: (x, y) => x * y,
#   func2: (x, y) => x + y,
# }
# Python:
# x = {
#     "func": lambda x, y: x * y,
#     "func2": lambda x, y: x + y
# }

class Coordinate:
    def __init__(self, x, y, count):
        self.x = x
        self.y = y
        self.count = count
# print(Coordinate(1,1,9).count)

# Each move shall be named according to position on a clock, like so:
#  11#1
# 10 #  2
# ###K###
# 8  #  4
#   7#5

# This can be simplified
# possibleMoves = {
#     '1': lambda x, y, count: Coordinate(x + 1, y + 2, count + 1),
#     '2': lambda x, y, count: Coordinate(x + 2, y + 1, count + 1),
#     '4': lambda x, y, count: Coordinate(x + 2, y - 1, count + 1),
#     '5': lambda x, y, count: Coordinate(x + 1, y - 2, count + 1),
#     '7': lambda x, y, count: Coordinate(x - 1, y - 2, count + 1),
#     '8': lambda x, y, count: Coordinate(x - 2, y - 1, count + 1),
#     '10': lambda x, y, count: Coordinate(x - 2, y + 1, count + 1),
#     '11': lambda x, y, count: Coordinate(x - 1, y + 2, count + 1),
# }
# x = possibleMoves['1'](4,4,1)
# print(x.x)
# print(x.y)
def moveFunc(xDiff, yDiff):
    return lambda x, y, count: Coordinate(x + xDiff, y + yDiff, count + 1)
# Create functions for each unique move
# THIS COULD JUST BE AN ARRAY
# we could also just use one function
# instead of returning 8 different functions
possibleMoves = {
    '1': moveFunc(1, 2),
    '2': moveFunc(2, 1),
    '4': moveFunc(2, -1),
    '5': moveFunc(1, -2),
    '7': moveFunc(-1, -2),
    '8': moveFunc(-2, -1),
    '10': moveFunc(-2, 1),
    '11': moveFunc(-1, 2),
}

# def generateNextMoveFuncs():
#     test = [1, 2, -1, -2]

def prettyPrint(test):
    print('Test: ' + str(test.x) + ',' + str(test.y) + ' in ' + str(test.count))

# def numToCoordinates(num):
#     # x = num // 8 # its an 8X8 board, so each row contains 8 spots
#     # y = num % 8
#     # print(x, y)
#     x = num % 8 # its an 8X8 board, so each row contains 8 spots
#     y = num // 8
#     return Coordinate(x, y, 0)
# print(19)
# prettyPrint(numToCoordinates(19))
# print(29)
# prettyPrint(numToCoordinates(29))
# print(39)
# prettyPrint(numToCoordinates(39))
# print('Test: ' + str(test.x) + ',' + str(test.y))

def solution(src, dest):
    # USE A QUEUE
    # start = numToCoordinates(src)
    # end = numToCoordinates(dest)
    start = Coordinate(src % 8, src // 8, 0)
    end = Coordinate(dest % 8, dest // 8, 0)
    # print('From ' + str(src) + ' to ' + str(dest))
    # print('Start: ' + str(start.x) + ',' + str(start.y) + ' End: ' + str(end.x) + ','+ str(end.y))
    queue = [start]
    i = 0

    # Move through the queue append all valid moves
    # Theoretically a knight should be able to make it to any square on the board
    # if given enough moves, so we dont even have to worry about indexing outside the array
    while queue[i].x != end.x or queue[i].y != end.y:
        for move in possibleMoves:
            nextMove = possibleMoves[move](queue[i].x, queue[i].y, queue[i].count)
            # Valid moves are those that dont already exist in the list,
            # and dont fall outside the boundries of the chess board

            if not 0 <= nextMove.x <= 7 or not 0 <= nextMove.y <= 7:
                continue

            moveIsUnq = True
            for existingMove in queue:
                if existingMove.x == nextMove.x and existingMove.y == nextMove.y:
                    moveIsUnq = False
                    break
                    # continue
            if not moveIsUnq:
                continue
            # moveIsValid = 0 <= nextMove.x <= 7 and 0 <= nextMove.y <=7
            # moveIsUnq = not any(existingMove.x == nextMove.x and existingMove.y == nextMove.y for existingMove in queue)

            # if moveIsUnq and 0 <= nextMove.x <= 7 and 0 <= nextMove.y <=7:
            # # if moveIsUnq and moveIsValid:
            #     queue.append(nextMove)
            queue.append(nextMove)
        i += 1
    # print(queue)
    # for node in queue:
    #     print(node.x, node.y, node.count)
    # print(i)
    # prettyPrint(queue[i])
    # print('i = ' + str(i))
    return queue[i].count

print(solution(9, 9))
print(solution(9, 19))
print(solution(9, 29))
print(solution(9, 39))
print(solution(19, 36))
print(solution(0, 1))
# solution(7, 56)

import time
def test():
    start = 0
    end = 0
    # highestMoveCount =  0
    timeStart = time.time()
    while start < 64:
        while end < 64:
            solution(start, end)
            # print('MOVECOUNT' + str(solution(start, end)))
            # if (solution(start, end) > highestMoveCount):
            #     highestMoveCount = solution(start, end)
            end += 1
        end = 0
        start += 1
    timeEnd = time.time()
    print(timeEnd - timeStart)
    # print(highestMoveCount)
test()
testCounter = 0
while testCounter < 16:
    test()
    testCounter += 1


# Don't Get Volunteered!
# ======================
# 
# As a henchman on Commander Lambda's space station, you're expected to be resourceful,
# smart, and a quick thinker. It's not easy building a doomsday device and ordering 
# the bunnies around at the same time, after all! In order to make sure that everyone 
# is sufficiently quick-witted, Commander Lambda has installed new flooring outside
# the henchman dormitories. It looks like a chessboard, and every morning and evening 
# you have to solve a new movement puzzle in order to cross the floor. That would 
# be fine if you got to be the rook or the queen, but instead, you have to be the 
# knight. Worse, if you take too much time solving the puzzle, you get "volunteered" 
# as a test subject for the LAMBCHOP doomsday device!
#
# To help yourself get to and from your bunk every day, write a function called
# solution(src, dest) which takes in two parameters: the source square, on which 
# you start, and the destination square, which is where you need to land to solve
# the puzzle.  The function should return an integer representing the smallest 
# number of moves it will take for you to travel from the source square to the 
# destination square using a chess knight's moves (that is, two squares in any 
# direction immediately followed by one square perpendicular to that direction, 
# or vice versa, in an "L" shape).  Both the source and destination squares will 
# be an integer between 0 and 63, inclusive, and are numbered like the example 
# chessboard below:
# -------------------------
# | 0| 1| 2| 3| 4| 5| 6| 7|
# -------------------------
# | 8| 9|10|11|12|13|14|15|
# -------------------------
# |16|17|18|19|20|21|22|23|
# -------------------------
# |24|25|26|27|28|29|30|31|
# -------------------------
# |32|33|34|35|36|37|38|39|
# -------------------------
# |40|41|42|43|44|45|46|47|
# -------------------------
# |48|49|50|51|52|53|54|55|
# -------------------------
# |56|57|58|59|60|61|62|63|
# -------------------------
# 
# Languages
# =========
# To provide a Python solution, edit solution.py
# To provide a Java solution, edit Solution.java
# 
# Test cases
# ==========
# Your code should pass the following test cases.Note that it may also be run against hidden test cases not shown here.
# 
# -- Python cases --
# Input:solution.solution(19, 36)
# Output:    1
# 
# Input:solution.solution(0, 1)
# Output:    3
# -- Java cases --
# Input:Solution.solution(19, 36)
# Output:    1
# 
# Input:Solution.solution(0, 1)
# Output:    3
# 
# 
# Use verify [file] to test your solution and see how it does.
# When you are finished editing your code, use submit [file] to submit your answer.
# If your solution passes the test cases, it will be removed from your home folder.
