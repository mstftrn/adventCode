# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import functools
import statistics


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


def solvepuzzle1(s):
    for i in s:
        if (2020 - i) in s:
            print(i * (2020 - i))
            return


def solvepuzzle2(s):
    for i in s:
        for j in s:
            if (2020 - i - j) in s:
                print(i * j * (2020 - i - j))
                return


def solvepuzzleday1part1(l):
    count = 0
    for c in range(len(l) - 1):
        if l[c] < l[c + 1]:
            count += 1
    return count


def solvepuzzleday1part2(l):
    count = 0
    for c in range(len(l) - 3):
        if l[c] + l[c + 1] + l[c + 2] < l[c + 1] + l[c + 2] + l[c + 3]:
            count += 1
    return count


# Press the green button in the gutter to run the script.
def solvepuzzleday2part1(l):
    horizontalpos = 0
    depthpos = 0
    for comm in l:
        c = comm.split()
        if c[0] == "forward":
            horizontalpos += int(c[1])
        elif c[0] == "down":
            depthpos += int(c[1])
        elif c[0] == "up":
            depthpos -= int(c[1])
    return horizontalpos * depthpos


def solvepuzzleday2part2(l):
    horizontalpos = 0
    depthpos = 0
    aim = 0
    for comm in l:
        c = comm.split()
        if c[0] == "forward":
            horizontalpos += int(c[1])
            depthpos += aim * int(c[1])
        elif c[0] == "down":
            aim += int(c[1])
        elif c[0] == "up":
            aim -= int(c[1])
    return horizontalpos * depthpos


def solvepuzzleday3part1(l):
    gamma = ""
    epsilon = ""
    strlen = len(l[0].strip())
    counts_gamma = []
    for i in range(strlen):
        counts_gamma.append([0, 0])
    # print(counts_gamma)

    for r in l:
        for i in range(strlen):
            if r[i] == "0":
                counts_gamma[i][0] += 1
            else:
                counts_gamma[i][1] += 1
    for i in range(strlen):
        if counts_gamma[i][0] > counts_gamma[i][1]:
            gamma += "0"
            epsilon += "1"
        else:
            gamma += "1"
            epsilon += "0"
    return int(gamma, 2) * int(epsilon, 2)


def bitcriteria(gamma, l, pos):
    newl = []
    for k in l:
        if gamma[pos] == k[pos]:
            newl.append(k)
    return newl


def calculategammaepsilon(l):
    gamma = ""
    epsilon = ""
    strlen = len(l[0].strip())
    counts_gamma = []
    for i in range(strlen):
        counts_gamma.append([0, 0])

    for r in l:
        for i in range(strlen):
            if r[i] == "0":
                counts_gamma[i][0] += 1
            else:
                counts_gamma[i][1] += 1
    for i in range(strlen):
        if counts_gamma[i][0] > counts_gamma[i][1]:
            gamma += "0"
            epsilon += "1"
        else:
            gamma += "1"
            epsilon += "0"
    return gamma, epsilon


def solvepuzzleday3part2(l):
    newl = l
    pos = 0
    while True:
        gamma, epsilon = calculategammaepsilon(newl)
        newl = bitcriteria(gamma, newl, pos)
        pos += 1
        if len(newl) == 1:
            break
    oxygen = newl[0]

    pos = 0
    newl = l
    while True:
        gamma, epsilon = calculategammaepsilon(newl)
        newl = bitcriteria(epsilon, newl, pos)
        pos += 1
        if len(newl) == 1:
            break
    co2scrubber = newl[0]

    return int(oxygen, 2) * int(co2scrubber, 2)


def flagNums(bingoNums, bingoexploded, n):
    for i in range(len(bingoNums)):
        for j in range(5):
            if bingoNums[i][j] == n:
                bingoexploded[i][j] = True


def checkBingo(bingoexploded, bingoNums):
    for i in range(int(len(bingoexploded) / 5)):
        for j in range(5):
            if (bingoexploded[i * 5 + j][0] and bingoexploded[i * 5 + j][1] and bingoexploded[i * 5 + j][2] and
                    bingoexploded[i * 5 + j][3] and bingoexploded[i * 5 + j][4]):
                return i
        for j in range(5):
            if (bingoexploded[i * 5][j] and bingoexploded[i * 5 + 1][j] and bingoexploded[i * 5 + 2][j] and
                    bingoexploded[i * 5 + 3][j] and bingoexploded[i * 5 + 4][j]):
                return i
    return -1


def calculateBingoSum(mtrx, bingoexploded, bingoNums):
    sum = 0
    for j in range(5):
        for k in range(5):
            if not (bingoexploded[mtrx * 5 + j][k]):
                sum += int(bingoNums[mtrx * 5 + j][k])
    return sum


def solvepuzzleday4part1(file1):
    nums = file1.readline().split(',')
    bingoNums = []
    file1.readline()
    while True:
        for i in range(5):
            row = file1.readline().split()
            bingoNums.append(row)

        if file1.readline() == "":
            break
    bingoexploded = []
    for i in range(len(bingoNums)):
        bingoexploded.append([False] * 5)

    for n in nums:
        flagNums(bingoNums, bingoexploded, n)
        i = checkBingo(bingoexploded, bingoNums)
        if i > -1:
            print(int(n) * calculateBingoSum(i, bingoexploded, bingoNums))
            return


def checkBingoAll(bingoexploded, bingoNums, boardswon):
    for i in range(int(len(bingoexploded) / 5)):
        for j in range(5):
            if (bingoexploded[i * 5 + j][0] and bingoexploded[i * 5 + j][1] and bingoexploded[i * 5 + j][2] and
                    bingoexploded[i * 5 + j][3] and bingoexploded[i * 5 + j][4]):
                boardswon[i] = True
        for j in range(5):
            if (bingoexploded[i * 5][j] and bingoexploded[i * 5 + 1][j] and bingoexploded[i * 5 + 2][j] and
                    bingoexploded[i * 5 + 3][j] and bingoexploded[i * 5 + 4][j]):
                boardswon[i] = True


def solvepuzzleday4part2(file1):
    nums = file1.readline().split(',')
    bingoNums = []
    file1.readline()
    while True:
        for i in range(5):
            row = file1.readline().split()
            bingoNums.append(row)

        if file1.readline() == "":
            break
    bingoexploded = []
    for i in range(len(bingoNums)):
        bingoexploded.append([False] * 5)
    boardswon = [False] * (int(len(bingoNums) / 5))

    lastmtrx = -1
    for n in nums:
        flagNums(bingoNums, bingoexploded, n)
        checkBingoAll(bingoexploded, bingoNums, boardswon)
        if boardswon.count(False) == 1:
            lastmtrx = boardswon.index(False)
            break
    print(lastmtrx)
    # i = checkBingo(bingoexploded, bingoNums)
    # if i > -1:
    #     boardswon[i] = True
    #     if boardswon.count(False) == 1:
    #         lastmtrx=boardswon.index(False)
    #         break
    newBingoNums = bingoNums[lastmtrx * 5:lastmtrx * 5 + 5]
    newBingoExploded = bingoexploded[lastmtrx * 5:lastmtrx * 5 + 5]
    print(newBingoNums)
    for n in nums:
        flagNums(newBingoNums, newBingoExploded, n)
        i = checkBingo(newBingoExploded, newBingoNums)
        if i > -1:
            print(int(n) * calculateBingoSum(i, newBingoExploded, newBingoNums))
            return


def addLineCoords(start, end):
    points = set()
    m1, m2 = int(start[0]), int(end[0])
    n1, n2 = int(start[1]), int(end[1])

    if m1 == m2:
        larger, smaller = max(n1, n2), min(n1, n2)
        for i in range(smaller, larger + 1):
            points.add((m1, i))
    if n1 == n2:
        larger, smaller = max(m1, m2), min(m1, m2)
        for i in range(smaller, larger + 1):
            points.add((i, n1))

    return points


def solvepuzzleday5part1(file1):
    Lines = file1.readlines()
    mtr = []
    for l in Lines:
        k = l.split()
        start = k[0].split(",")
        end = k[2].split(',')
        if start[0] != end[0] and start[1] != end[1]:
            continue
        mtr.append(addLineCoords(start, end))
    mtr2 = mtr.copy()
    resultingCoords = set()
    for i in range(len(mtr)):
        for j in range(i + 1, len(mtr2)):
            resultingCoords = resultingCoords.union(mtr[i].intersection(mtr[j]))
    print(len(resultingCoords))


def addLineCoords2(start, end):
    points = set()
    m1, m2 = start[0], end[0]
    n1, n2 = start[1], end[1]

    if m1 == m2:
        larger, smaller = max(n1, n2), min(n1, n2)
        for i in range(smaller, larger + 1):
            points.add((m1, i))
    elif n1 == n2:
        larger, smaller = max(m1, m2), min(m1, m2)
        for i in range(smaller, larger + 1):
            points.add((i, n1))
    else:  # 45 degrees forwards in x
        if m1 > m2:
            if n1 > n2:
                for i in range(m1, m2 - 1, -1):
                    points.add((i, n1))
                    n1 -= 1
            else:
                for i in range(m1, m2 - 1, -1):
                    points.add((i, n1))
                    n1 += 1
        else:
            if n1 > n2:
                for i in range(m1, m2 + 1):
                    points.add((i, n1))
                    n1 -= 1
            else:
                for i in range(m1, m2 + 1):
                    points.add((i, n1))
                    n1 += 1
    return points


def isNot45degrees(start, end):
    return abs(start[0] - end[0]) != abs(start[1] - end[1])


def solvepuzzleday5part2(file1):
    Lines = file1.readlines()
    mtr = []
    for l in Lines:
        k = l.split()
        p1 = k[0].split(',')
        start = int(p1[0]), int(p1[1])
        p2 = k[2].split(',')
        end = int(p2[0]), int(p2[1])
        mtr.append(addLineCoords2(start, end))

    resultingCoords = set()

    for i in range(len(mtr)):
        for j in range(i + 1, len(mtr)):
            resultingCoords = resultingCoords.union(mtr[i].intersection(mtr[j]))

    print(len(resultingCoords))


def daypasses(ttls):
    newttls = []
    for i in range(len(ttls)):
        ttls[i] -= 1
        if ttls[i] == -1:
            newttls.append(8)
            ttls[i] = 6
    ttls.extend(newttls)


def solvepuzzleday6part1(file1):
    ttls = list(map(lambda x: int(x), file1.readline().split(',')))
    for i in range(80):
        daypasses(ttls)

    print(len(ttls))


def addLanterntoDays(i, starting, numdays, lookupTable):
    #    if lookupTable[i][starting] != -1:
    #        return lookupTable[i][starting]

    j = i + starting
    sum = 0
    if j > numdays - 1:
        return sum
    sum += 1
    sum += addLanterntoDays(j, 7, numdays, lookupTable)
    sum += addLanterntoDays(j, 9, numdays, lookupTable)

    return sum


def solvepuzzleday6part2_old(file1):
    numdays = 15
    ttls = list(map(lambda x: int(x), file1.readline().split(',')))
    sum = len(ttls)
    lookupTable = [[-1] * 10] * numdays
    for starting in ttls:
        sum += addLanterntoDays(0, starting, numdays, lookupTable)
    print(sum)


def solvepuzzleday6part2(file1):
    numdays = 256
    ttls = list(map(lambda x: int(x), file1.readline().split(',')))
    olderlantern = [[0] * 7]
    for i in range(numdays):
        olderlantern.append([0] * 7)
    for t in ttls:
        olderlantern[0][t] += 1
    newbabies = [[0] * 9]
    for i in range(numdays):
        newbabies.append([0] * 9)
    for i in range(numdays):
        newbabies[i + 1][8] = olderlantern[i][0]
        newbabies[i + 1][8] += newbabies[i][0]
        olderlantern[i + 1][6] = olderlantern[i][0]
        olderlantern[i + 1][6] += newbabies[i][0]
        for j in range(6):
            olderlantern[i + 1][j] = olderlantern[i][j + 1]
        for j in range(8):
            newbabies[i + 1][j] = newbabies[i][j + 1]
    # for i in range(numdays+1):
    #    print("After day",i, olderlantern[i], newbabies[i])
    print(sum(newbabies[numdays]) + sum(olderlantern[numdays]))


def solvepuzzleday7part1(file1):
    initialpositions = list(map(lambda x: int(x), file1.readline().split(',')))
    med = int(statistics.median(initialpositions))
    sum = functools.reduce(lambda x, y: x + y, map(lambda z: abs(med - z), initialpositions))
    print(sum)


def crabcost(dist):
    return int(dist * (dist + 1) / 2)


def solvepuzzleday7part2(file1):
    initialpositions = list(map(lambda x: int(x), file1.readline().split(',')))
    mn = int(statistics.mean(initialpositions))
    mn2 = mn + 1
    sum = functools.reduce(lambda x, y: x + y, map(lambda z: crabcost(abs(mn - z)), initialpositions))
    sum2 = functools.reduce(lambda x, y: x + y, map(lambda z: crabcost(abs(mn2 - z)), initialpositions))
    print(min(sum, sum2))


def solvepuzzleday8part1(file1):
    Lines = file1.readlines()
    sum = 0
    for l in Lines:
        lparts = l.split('|')
        lnums = lparts[0].strip().split(' ')
        lres = lparts[1].strip().split(' ')
        sum += len(list(filter(lambda x: len(x) in (2, 3, 4, 7), lres)))
    print(sum)


def textminustext(txt1, txt2):
    t1 = set(txt1)
    t2 = set(txt2)
    r = ''
    for t in t1.difference(t2):
        r += t
    return r


def test6(lnums, numstrings, segments):
    for l in lnums:
        if len(l) == 6:  # 6,9,0
            dif = textminustext(numstrings[1], l)
            if len(dif) == 1:  # 6
                numstrings[6] = l
                segments[2] = dif
                segments[5] = textminustext(numstrings[1], dif)
    for l in lnums:
        if len(l) == 5:  # 2,3,5
            dif = textminustext(segments[2], l)
            if len(dif) == 1:  # 5
                numstrings[5] = l
            dif = textminustext(segments[5], l)
            if len(dif) == 1:  # 2
                numstrings[2] = l
    segments[4] = textminustext(numstrings[6], numstrings[5])
    segments[1] = functools.reduce(textminustext, [numstrings[8], numstrings[2], segments[5]])
    segments[6] = functools.reduce(textminustext, [numstrings[8], numstrings[4], segments[0], segments[4]])
    segments[3] = functools.reduce(textminustext, [numstrings[4], segments[1], segments[2], segments[5]])


#  aaaa    0000
# b    c  1    2
# b    c  1    2
#  dddd    3333
# e    f  4    5
# e    f  4    5
#  gggg    6666

def mapnums(s: set, segments):
    length = len(s)
    if length == 2:
        return 1
    elif length == 3:
        return 7
    elif length == 4:
        return 4
    elif length == 7:
        return 8
    elif length == 6:  # 0,6,9
        if segments[3] not in s:
            return 0
        elif segments[4] not in s:
            return 9
        elif segments[2] not in s:
            return 6
    else:  # 2,3,5
        if segments[2] not in s:
            return 5
        elif segments[4] not in s:
            return 3
        else:
            return 2


def solvepuzzleday8part2(file1):
    Lines = file1.readlines()
    numsStrings = [''] * 10
    segments = [''] * 7
    sum = 0
    for l in Lines:
        lparts = l.split('|')
        lnums = lparts[0].strip().split(' ')
        lres = lparts[1].strip().split(' ')
        for n in lnums:
            length = len(n)
            if length == 2:
                numsStrings[1] = n
            elif length == 3:
                numsStrings[7] = n
            elif length == 4:
                numsStrings[4] = n
            elif length == 7:
                numsStrings[8] = n
        segments[0] = textminustext(numsStrings[7], numsStrings[1])
        test6(lnums, numsStrings, segments)
        resultingnumber = ''
        for k in lres:
            resultingnumber += str(mapnums(set(k), segments))
        sum += int(resultingnumber)
    print(sum)


def lessthanfourneighbors(j, i, mtrx):
    if mtrx[j][i] < mtrx[j][i - 1] and mtrx[j][i] < mtrx[j + 1][i] and mtrx[j][i] < mtrx[j][i + 1] and \
           mtrx[j][i] < mtrx[j - 1][i]:
        return True
    return False



def solvepuzzleday9part1(file1):
    Lines = file1.readlines()
    for i in range(len(Lines)):
        Lines[i] = Lines[i].strip()
    linelength = len(Lines[0])
    matrixheight = len(Lines)
    mtrx = [[10] * (linelength + 2)]
    for i in range(len(Lines)):
        ltemp = [10] + list(map(int, list(Lines[i]))) + [10]
        mtrx.append(ltemp)
    mtrx.append([10] * (linelength + 2))
    for i in range(len(mtrx)):
        print(mtrx[i])
    sum = 0
    for i in range(1, linelength + 1):
        for j in range(1, matrixheight + 1):
            if lessthanfourneighbors(j, i, mtrx):
                sum += 1+mtrx[j][i]
    print(sum)


def traverse(mtrx, i, j, visitedCells):
    sum=1
    visitedCells.add((i,j))
    if mtrx[i][j-1]<9 and (i, j-1) not in visitedCells:
        sum+=traverse(mtrx, i, j-1, visitedCells)
    if mtrx[i+1][j]<9 and (i+1, j) not in visitedCells:
        sum+=traverse(mtrx, i+1, j, visitedCells)
    if mtrx[i][j+1] < 9 and (i, j+1) not in visitedCells:
        sum += traverse(mtrx, i, j+1, visitedCells)
    if mtrx[i-1][j] < 9 and (i-1, j) not in visitedCells:
        sum += traverse(mtrx, i-1, j, visitedCells)
    return sum


def solvepuzzleday9part2(file1):
    Lines = file1.readlines()
    for i in range(len(Lines)):
        Lines[i] = Lines[i].strip()
    linelength = len(Lines[0])
    matrixheight = len(Lines)
    mtrx = [[10] * (linelength + 2)]
    for i in range(len(Lines)):
        ltemp = [10] + list(map(int, list(Lines[i]))) + [10]
        mtrx.append(ltemp)
    mtrx.append([10] * (linelength + 2))
    for i in range(len(mtrx)):
        print(mtrx[i])
    visitedCells=set()
    totalsums=[]
    for i in range(1, matrixheight+1):
        for j in range(1, linelength+1):
            if mtrx[i][j]!=9:
                if (i,j) not in visitedCells:
                    totalsums.append(traverse(mtrx, i, j, visitedCells))
    totalsums.sort()
    print(totalsums[-1]*totalsums[-2]*totalsums[-3])




if __name__ == '__main__':
    file1 = open('inputday9part1.txt', 'r')
    # read file
    # file1 = open('inputday7part1.txt', 'r')
    # solvepuzzleday5part1(file1)
    # solvepuzzleday5part2(file1)
    # solvepuzzleday6part1(file1)
    # solvepuzzleday6part2(file1)
    # solvepuzzleday8part2(file1)
    # solvepuzzleday9part1(file1)
    solvepuzzleday9part2(file1)

    # solvepuzzleday7part1(file1)
    # solvepuzzleday7part2(file1)
    # Lines = file1.readlines()
    # s={int(l) for l in Lines}
    # l=[int(strn) for strn in Lines]
    # print(solvepuzzleday2part1(Lines))
    # print(solvepuzzleday1part1(l))
    # print(solvepuzzleday3part1(Lines))
    # print(solvepuzzleday3part2(Lines))
    # count = 0
    # # Strips the newline character
    # for line in Lines:
    #     count += 1
    #     print("Line{}: {}".format(count, line.strip()))
    # solvepuzzleday4part1(file1)
    # solvepuzzleday4part2(file1)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
