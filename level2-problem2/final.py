class Coordinate:
    def __init__(self, x, y, count):
        self.x = x
        self.y = y
        self.count = count

def moveFunc(xDiff, yDiff):
    return lambda x, y, count: Coordinate(x + xDiff, y + yDiff, count + 1)

def solution(src, dest):
    start = Coordinate(src % 8, src // 8, 0)
    end = Coordinate(dest % 8, dest // 8, 0)
    # print('From ' + str(src) + ' to ' + str(dest))
    # print('Start: ' + str(start.x) + ',' + str(start.y) + ' End: ' + str(end.x) + ','+ str(end.y))
    queue = [start]
    i = 0

    possibleMoves = [
        moveFunc(1, 2),
        moveFunc(2, 1),
        moveFunc(2, -1),
        moveFunc(1, -2),
        moveFunc(-1, -2),
        moveFunc(-2, -1),
        moveFunc(-2, 1),
        moveFunc(-1, 2),
    ]

    # Move through the queue append all valid moves
    # Theoretically a knight should be able to make it to any square on the board
    # if given enough moves, so we dont even have to worry about indexing outside the array
    while queue[i].x != end.x or queue[i].y != end.y:
        for move in possibleMoves:
            # nextMove = possibleMoves[move](queue[i].x, queue[i].y, queue[i].count)
            nextMove = move(queue[i].x, queue[i].y, queue[i].count)
            # Valid moves are those that dont already exist in the list,
            # and dont fall outside the boundries of the chess board

            if not 0 <= nextMove.x <= 7 or not 0 <= nextMove.y <= 7:
                continue

            moveIsUnq = True
            for existingMove in queue:
                if existingMove.x == nextMove.x and existingMove.y == nextMove.y:
                    moveIsUnq = False
                    break
            if not moveIsUnq:
                continue

            queue.append(nextMove)
        i += 1

    return queue[i].count

print(solution(9, 9))
print(solution(9, 19))
print(solution(9, 29))
print(solution(9, 39))
print(solution(19, 36))
print(solution(0, 1))
# solution(7, 56)

def solutionV2(src, dest):
    # Convert src/dest from number to coordinates
    start = Coordinate(src % 8, src // 8, 0)
    end = Coordinate(dest % 8, dest // 8, 0)
    queue = [start]
    i = 0

    # Generate a function for each possible move the knight can make
    possibleMoves = [
        moveFunc(1, 2),
        moveFunc(2, 1),
        moveFunc(2, -1),
        moveFunc(1, -2),
        moveFunc(-1, -2),
        moveFunc(-2, -1),
        moveFunc(-2, 1),
        moveFunc(-1, 2),
    ]

    # Move through the queue and append all valid moves
    # Theoretically a knight should be able to make it to any square on the board
    # if given enough moves, so we dont have to worry about indexing outside the array
    while queue[i].x != end.x or queue[i].y != end.y:
        for move in possibleMoves:
            nextMove = move(queue[i].x, queue[i].y, queue[i].count)
            # Valid moves are those that dont already exist in the list,
            # and dont fall outside the boundries of the chess board

            if not 0 <= nextMove.x <= 7 or not 0 <= nextMove.y <= 7:
                continue

            moveExists = False
            for existingMove in queue:
                if existingMove.x == nextMove.x and existingMove.y == nextMove.y:
                    moveExists = True
                    break
            if moveExists:
                continue

            queue.append(nextMove)
        i += 1

    return queue[i].count


import time
def test():
    start = 0
    end = 0
    # highestMoveCount =  0
    timeStart = time.time()
    while start < 64:
        while end < 64:
            solutionV2(start, end)
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


