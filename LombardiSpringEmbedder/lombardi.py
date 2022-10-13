#############################################################################################
# This is the main file for the Lombardi Spring Embedder
# It's a force directed graph layout algorithm with Lombardi forces and circular arcs.
# This file is a copy of fdp.py but with additional functionality.
# Usage example is at the bottom.
# Can be used with simple dot files or graphml (or modify classes.py to parse other formats).
# I realize it's sparsely commented so if you need help just shoot me an email.
# Author: Roman Chernobelskiy, romanc@email.arizona.edu (University of Arizona)
# Developed on Python 2.5
#############################################################################################
from __future__ import division
import os
import sys
import pygame
from PIL import Image
import aggdraw
from classes import *
from random import random as rand
from random import shuffle as shuffle

# fdp settings
maxIters = 600
EXPFACTOR = 1.2
K = 0.3
K2 = K * K
useNew = True

# lombardi settings
maxDiff = .1  # max difference between two tangent diffs, for arc to be possible, in radians
rfKopp = .5  # rotational force constant, for how opp tangents affect my rotation(want my tan to match opp)
rfKadj = .1  # rotational force constant, for how my tangents affect my rotation(want my tan close to my edge)
tangentialK = .9  # tangential force constant
finalRound = .03  # reserve a percent of iterations at the end just for my forces
shufflePercent = .4  # percentage of iterations to shuffle on
shuffleSamples = 20  # max number of shuffle attempts
modeLombardi = False

# LIVE display settings
live = False
fps = 100
everyOtherFrame = True
tanLen = 30  # screen px
drawEdges = False
drawTans = False

# global algorithm vars
screen = 0
clock = 0
sw, sh = 500, 500  # screen width&height, should be square
padding = 10  # some padding for the screen
bl = [0, 0]  # bounding box
tr = [0, 0]  # bounding box
shuffleEvery = int(round(1 / shufflePercent))  # iteration mod to shuffle on
maxDeterministicShuffle = 1
while factorial(maxDeterministicShuffle + 1) < shuffleSamples: maxDeterministicShuffle += 1


def init():
    global screen, clock
    Wd = EXPFACTOR * (K / 2.0)
    Ht = EXPFACTOR * (K / 2.0)

    for n in nodes():
        n.y = Ht * (2.0 * rand() - 1.0)
        n.degree = len([x for x in n.nodes()])
        n.aIncrement = (2 * m.pi) / n.degree

    for e in edges():
        e.factor = 1.0
        e.dist = K
        e.pa = e.p.nextTanAngle()
        e.qa = e.q.nextTanAngle()

    if live:
        pygame.init()
        screen = pygame.display.set_mode([sw + padding * 2, sh + padding * 2])
        screen.fill([255, 255, 255])
        clock = pygame.time.Clock()


def cool(t0, i):
    return (t0 * (maxIters - i)) / maxIters


def adjust(temp, iteration):
    if temp <= 0:
        print("frozen!")
        return

    for n in nodes():
        n.fx = 0
        n.fy = 0
        n.tx = 0
        n.ty = 0
        n.rf = 0

    if iteration < (maxIters * (1 - finalRound)) or not modeLombardi:
        for i in range(len(nodeList)):
            for j in range(i + 1, len(nodeList)):
                repApply(nodeList[i], nodeList[j])

        for e in edges():
            attrApply(e.p, e.q, e)
    if modeLombardi:
        if iteration % shuffleEvery == 0:
            for n in nodes():
                shuffleTans(n)
        for n in nodes():
            rfApply(n)
        for n in nodes():
            tfApply(n)

        updateAngle(temp)

    updatePos(temp)


def repApply(p, q):
    dx = q.x - p.x
    dy = q.y - p.y
    distance2 = dx * dx + dy * dy
    while distance2 == 0:
        dx = 5 - m.round(rand() * 10)
        dy = 5 - m.round(rand() * 10)
        distance2 = dx * dx + dy * dy

    force = 0
    if useNew:
        force = K2 / (m.sqrt(distance2) * distance2)  # this is dist^3
    else:
        force = K2 / distance2
    q.fx += dx * force
    q.fy += dy * force
    p.fx -= dx * force
    p.fy -= dy * force


# renamed dist2 to distance2
def attrApply(p, q, e):
    dx = q.x - p.x
    dy = q.y - p.y
    distance2 = dx * dx + dy * dy
    while distance2 == 0:
        dx = 5 - m.round(rand() * 10)
        dy = 5 - m.round(rand() * 10)
        distance2 = dx * dx + dy * dy
    # renamed dist to distance
    distance = m.sqrt(distance2)
    force = 0
    if useNew:
        force = (e.factor * (distance - e.dist)) / distance
    else:
        force = (e.factor * distance) / e.dist
    q.fx -= dx * force
    q.fy -= dy * force
    p.fx += dx * force
    p.fy += dy * force


def shuffleTans(n):
    neighbors = [nn for nn in n.nodes()]
    bestVal = rfComputeTot(n)
    bestCombo = [n.tanAngleRel(nn) for nn in neighbors]
    if n.degree <= maxDeterministicShuffle:
        for angles in permutations([n.tanAngleRel(nn) for nn in neighbors]):
            for i in range(n.degree): n.setAngle(neighbors[i], angles[i])
            val = rfComputeTot(n)
            if val < bestVal:
                bestVal = val
                bestCombo = angles[:]
    else:  # random shuffle
        for k in range(
                n.degree):  # this one is optimizeable(switch unhappy ones)(or try permutations with happy ones in place?)
            for j in range(n.degree):
                if k != j:
                    angles = bestCombo[:]
                    angles[k] = bestCombo[j]
                    angles[j] = bestCombo[k]
                    for i in range(n.degree): n.setAngle(neighbors[i], angles[i])
                    val = rfComputeTot(n)
                    if val < bestVal:
                        bestVal = val
                        bestCombo = angles[:]
        angles = bestCombo[:]
        for z in range(shuffleSamples):
            shuffle(angles)
            for i in range(n.degree): n.setAngle(neighbors[i], angles[i])
            val = rfComputeTot(n)
            if val < bestVal:
                bestVal = val
                bestCombo = angles[:]
    for i in range(n.degree): n.setAngle(neighbors[i], bestCombo[i])


def rfComputeTot(n):
    rf = 0.0
    for nn in n.nodes():
        opti = pie(n.edgeAngle(nn) - nn.diffAngle(n))
        rot = pie(opti - n.tanAngle(nn))
        rf += abs(rot) * rfKadj
        rot = n.diffAngle(nn)
        rf += abs(rot) * rfKopp
    # squared????
    # rf+=rot*rot#squared????
    # sq root??? cuz sq doesn't work for <1 as expected
    return rf  # squared????


def rfComputeNet(n):
    rf = 0.0
    for nn in n.nodes():
        # opp tan
        opti = pie(n.edgeAngle(nn) - nn.diffAngle(n))
        rot = pie(opti - n.tanAngle(nn))
        rf += rot * rfKopp
        # my tan
        rf -= n.diffAngle(nn) * rfKadj
    return rf


def rfApply(n):
    n.rf = rfComputeNet(n)


def tfApply(n):
    for nn in n.nodes():  # move n according to nn
        avg = (n.diffAngle(nn) - nn.diffAngle(n)) / 2.0
        aopti = pie2(nn.tanAngle(n) + avg)  # absolute optimal edge angle
        rot = pie(aopti - nn.edgeAngle(n))
        len = dist(n, nn)
        optip = (nn.x + m.cos(aopti) * len, nn.y + m.sin(aopti) * len)
        # make force proprtional to rotation proposed
        n.tx += (optip[0] - n.x) * tangentialK  # *(abs(rot)/m.pi)
        n.ty += (optip[1] - n.y) * tangentialK  # *(abs(rot)/m.pi)


def updateAngle(temp):
    for n in nodes():
        n.angle += n.rf * temp


def updatePos(temp):
    temp2 = temp * temp
    for i in range(2): tr[i] = 0
    for i in range(2): bl[i] = 0
    for n in nodes():
        n.fx += n.tx  # some other factor mb?
        n.fy += n.ty  # some other factor mb?
        len2 = n.fx * n.fx + n.fy * n.fy

        if len2 < temp2:
            n.x += n.fx
            n.y += n.fy
        else:  # limit by temp
            fact = temp / m.sqrt(len2)
            n.x += n.fx * fact
            n.y += n.fy * fact

        # bounding box
        if n.x < bl[0]:
            bl[0] = n.x
        elif n.x > tr[0]:
            tr[0] = n.x
        if n.y < bl[1]:
            bl[1] = n.y
        elif n.y > tr[1]:
            tr[1] = n.y


def finalStep():
    legit = True
    for e in edges():
        pd = e.diffAngle(e.p)
        qd = e.diffAngle(e.q)
        optidiff = (pd - qd) / 2
        adjust = abs(pd + qd) / 2
        if adjust > .01: legit = False
        if optidiff > pd:  # increase
            if e.p.degree == 1:
                e.pa = pie2(e.pa + adjust * 2)
            elif e.q.degree == 1:
                e.qa = pie2(e.qa + adjust * 2)
            else:
                e.pa = pie2(e.pa + adjust)
                e.qa = pie2(e.qa + adjust)
        else:
            if e.p.degree == 1:
                e.pa = pie2(e.pa - adjust * 2)
            elif e.q.degree == 1:
                e.qa = pie2(e.qa - adjust * 2)
            else:
                e.pa = pie2(e.pa - adjust)
                e.qa = pie2(e.qa - adjust)
        dd = abs(e.diffAngle(e.p) + e.diffAngle(e.q))
        if dd > .000001:
            print(e.diffAngle(e.p), e.diffAngle(e.q), abs)
    return legit


def draw(i, temp):
    tick_time = clock.tick(fps)  # milliseconds since last frame
    screen.fill((240, 240, 240))
    # center drawing and find bb scale
    sz = (tr[0] - bl[0], tr[1] - bl[1])
    if sz[0] > sz[1]:
        tr[1] += (sz[0] - sz[1]) / 2
        bl[1] -= (sz[0] - sz[1]) / 2
    else:
        tr[0] += (sz[1] - sz[0]) / 2
        bl[0] -= (sz[1] - sz[0]) / 2

    scale = sw / max(sz)
    toScreen = lambda n: (int(round((n.x - bl[0]) * scale) + padding), int(round((n.y - bl[1]) * scale) + padding))

    # Edges
    if drawEdges or not modeLombardi:
        for e in edges():
            pygame.draw.line(screen, (200, 200, 200), toScreen(e.p), toScreen(e.q), 1)

    if modeLombardi:
        if i < maxIters:  # Curves
            ctrlPtK = .4
            for e in edges():
                start = toScreen(e.p)
                end = toScreen(e.q)
                startLen = dist(start, end) * abs(e.diffAngle(e.p)) * ctrlPtK
                endLen = dist(start, end) * abs(e.diffAngle(e.q)) * ctrlPtK
                startCtrl = (start[0] + startLen * m.cos(e.tanAngle(e.p)), start[1] + startLen * m.sin(e.tanAngle(e.p)))
                endCtrl = (end[0] + endLen * m.cos(e.tanAngle(e.q)), end[1] + endLen * m.sin(e.tanAngle(e.q)))
                arcPoss = abs(e.diffAngle(e.p) + e.diffAngle(e.q)) < maxDiff
                pygame.draw.aalines(screen, (0 if arcPoss else 255, 255 if arcPoss else 0, 0), False,
                                    bezier([start, startCtrl, endCtrl, end]))
            # pygame.draw.line(screen, (100,100,100), start, startCtrl, 1)
            # pygame.draw.line(screen, (100,100,100), end, endCtrl, 1)
        else:  # Arcs
            for e in edges():
                if abs(e.diffAngle(e.p)) < (m.pi / 40):
                    pygame.draw.line(screen, (0, 255, 0), toScreen(e.p), toScreen(e.q), 2)
                else:
                    len = dist(e.p, e.q) / 10
                    pCtrl = Vector(e.p[0] + m.cos(e.tanAngle(e.p)) * len, e.p[1] + m.sin(e.tanAngle(e.p)) * len)
                    qCtrl = Vector(e.q[0] + m.cos(e.tanAngle(e.q)) * len, e.q[1] + m.sin(e.tanAngle(e.q)) * len)
                    center = intersection((-pCtrl.y + (e.p.x + e.p.y), pCtrl.x + (e.p.y - e.p.x)), e.p,
                                          (-qCtrl.y + (e.q.x + e.q.y), qCtrl.x + (e.q.y - e.q.x)), e.q)
                    radius = dist(toScreen(center), toScreen(e.p))
                    # pygame.draw.circle(screen, (0,0,255), toScreen(center), int(round(radius)), 1)
                    # big arc or small arc?
                    intr = intersection(e.p, pCtrl, e.q, qCtrl)
                    bigArc = dist(intr, pCtrl) > dist(intr, e.p)
                    pa = -fromToAngle(center, e.p)
                    qa = -fromToAngle(center, e.q)
                    if qa < pa:
                        qa += m.pi * 2
                    if (bigArc and qa - pa < m.pi) or (not bigArc and qa - pa > m.pi):
                        tmp = pa
                        pa = qa
                        qa = tmp
                        qa += m.pi * 2
                    rect = pygame.Rect(toScreen(center)[0] - radius, toScreen(center)[1] - radius, radius * 2,
                                       radius * 2)
                    # pygame.draw.rect(screen, (255,0,0), rect, 1)
                    pygame.draw.arc(screen, (0, 255, 0), rect, pa, qa, 2)

    # Tangents
    if drawTans and i < maxIters:
        for n in nodes():
            s = toScreen(n)
            for nn in n.nodes():
                endx = s[0] + tanLen * m.cos(n.tanAngle(nn))
                endy = s[1] + tanLen * m.sin(n.tanAngle(nn))
                pygame.draw.line(screen, (100, 100, 100), s, (endx, endy), 1)

    # Nodes
    for n in nodes(): pygame.draw.circle(screen, (0, 0, 0), toScreen(n), 7)

    # e=getEdge(nodeList[0], nodeList[1])
    # pygame.display.set_caption("aDiff=%d, bDiff=%d" % (int(e.diffAngle(nodeList[0])*57),int(e.diffAngle(nodeList[1])*57)))
    pygame.display.set_caption("%d, %f" % (i, temp))
    screen.blit(pygame.transform.flip(screen, False, True), (0, 0))  # flip y
    pygame.display.update()


def adjust2(i):
    fMult = .5
    for n in nodes():
        if n.degree == 1:
            continue
        iter = sorted([(n.tanAngle(nn), nn) for nn in n.nodes()], key=lambda x: x[0])
        iter.insert(0, iter[n.degree - 1])
        iter.append(iter[1])
        for i in range(1, n.degree + 1):
            tmp = pie2(iter[i + 1][0] - iter[i - 1][0])
            if abs(tmp) < .0001:
                tmp = m.pi * 2 - .0001
            optiTanAngle = pie2(iter[i - 1][0] + tmp / 2)
            rf = pie(optiTanAngle - iter[i][0]) ** 3
            nn = iter[i][1]
            # optiDiff = -nn.diffAngle(n)#rem!
            # rf += pie(optiDiff - n.diffAngle(nn))*combineK
            n.addForce(nn, rf * fMult)
            nn.addForce(n, -rf * fMult)

    for e in edges():
        e.pa += e.pf
        e.qa += e.qf
        e.pf = 0.0
        e.qf = 0.0


def checkQuit():
    for e in pygame.event.get():
        if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
            return True
    return False


def waitQuit():
    while not checkQuit():
        pass


def getScore():
    cnt = 0
    tot = 0.0
    for n in nodes():
        if n.degree == 1:
            continue
        angles = sorted([n.tanAngle(nn) for nn in n.nodes()])
        angles.append(angles[0])
        for i in range(n.degree):
            cnt += 1
            tot += abs(n.aIncrement - pie2(angles[i + 1] - angles[i]))  # squared takes care of neg sign
    tot /= cnt
    return 100 * (1 - tot / m.pi)


def do(file):
    one, two = 0, 0
    parseFile(file)  # parse file into global namespace
    init()  # place nodes

    T0 = K * m.sqrt(len(nodeList)) / 5.0
    for i in range(maxIters):
        adjust(cool(T0, i), i)
        if live:
            if not everyOtherFrame or i % 2 == 0:
                draw(i, cool(T0, i))
            if checkQuit():
                return
    if modeLombardi:
        legit = finalStep()
        one = getScore()
        if not legit:
            #############################
            # Save (Cache current pa and qa vals!)(use unused vars or somn, mb ctrl pt)
            if live:
                for i in range(100):
                    draw(int(i / (100 / maxIters)), 0)
                    if checkQuit():
                        return

            for i in range(200):
                adjust2(i)
                if live:
                    draw(int(i / (200 / maxIters)), 0)
                    if checkQuit():
                        return
            two = getScore()

            if two < one:
                # restorePrev()
                pass

            # double check circleness
            for e in edges():
                dd = abs(e.diffAngle(e.p) + e.diffAngle(e.q))
                if dd > .000001:
                    print(e.diffAngle(e.p), e.diffAngle(e.q), dd)

        if live:
            draw(maxIters, 0)
            pygame.display.set_caption("%f, %f" % (one, two))
    else:  # set tans to edges
        for e in edges():
            e.pa = e.edgeAngle(e.p)
            e.qa = e.edgeAngle(e.q)
        if live:
            draw(0, 0)

    if live:
        waitQuit()


# fix bounding box for arcs
def drawToFile(file):
    score = getScore()
    file = ".".join(file.split('.')[:-1])
    score = "_%.2f" % score
    add = "_" if modeLombardi else ""
    img = Image.new("RGB", (520, 520), (240, 240, 240))

    d = aggdraw.Draw(img)
    edgePen = aggdraw.Pen("black", 1)
    arcPen = aggdraw.Pen("black", 1)
    tanPen = aggdraw.Pen("black", 1)
    nodeBrush = aggdraw.Brush("black")
    dbgBrush = aggdraw.Brush("red")

    ppt = []
    # adjust bbx for curves
    if modeLombardi:
        for e in edges():
            if abs(e.diffAngle(e.p)) < (m.pi / 40):
                pass
            else:
                len = dist(e.p, e.q) / 10
                pCtrl = Vector(e.p[0] + m.cos(e.tanAngle(e.p)) * len, e.p[1] + m.sin(e.tanAngle(e.p)) * len)
                qCtrl = Vector(e.q[0] + m.cos(e.tanAngle(e.q)) * len, e.q[1] + m.sin(e.tanAngle(e.q)) * len)
                center = intersection((-pCtrl.y + (e.p.x + e.p.y), pCtrl.x + (e.p.y - e.p.x)), e.p,
                                      (-qCtrl.y + (e.q.x + e.q.y), qCtrl.x + (e.q.y - e.q.x)), e.q)
                radius = dist(center, e.p)
                mybl = Vector(center.x - radius, center.y - radius)
                mytr = Vector(center.x + radius, center.y + radius)
                # big arc or small arc?
                intr = intersection(e.p, pCtrl, e.q, qCtrl)
                bigArc = dist(intr, pCtrl) > dist(intr, e.p)
                pa = -fromToAngle(center, e.p)
                qa = -fromToAngle(center, e.q)

                if qa < pa:
                    qa += m.pi * 2
                if (bigArc and qa - pa < m.pi) or (not bigArc and qa - pa > m.pi):
                    tmp = pa
                    pa = qa
                    qa = tmp
                    qa += m.pi * 2
                pa = int(m.degrees(pa))
                qa = int(m.degrees(qa))
                while pa < qa:
                    ta = pa % 360
                    """
					if ta == 0 and mytr.x > tr[0]: ppt.append(Vector(mytr.x, center.y))
					if ta == 270 and mytr.y > tr[1]: ppt.append(Vector(center.x, mytr.y))
					if ta == 180 and mybl.x < bl[0]: ppt.append(Vector(mybl.x, center.y))
					if ta == 90 and mybl.y < bl[1]: ppt.append(Vector(center.x, mybl.y))
					"""
                    if ta == 0 and mytr.x > tr[0]:
                        tr[0] = mytr.x
                    if ta == 270 and mytr.y > tr[1]:
                        tr[1] = mytr.y
                    if ta == 180 and mybl.x < bl[0]:
                        bl[0] = mybl.x
                    if ta == 90 and mybl.y < bl[1]:
                        bl[1] = mybl.y
                    pa += 1

    # center drawing and find bb scale
    sz = (tr[0] - bl[0], tr[1] - bl[1])
    if sz[0] > sz[1]:
        tr[1] += (sz[0] - sz[1]) / 2
        bl[1] -= (sz[0] - sz[1]) / 2
    else:
        tr[0] += (sz[1] - sz[0]) / 2
        bl[0] -= (sz[1] - sz[0]) / 2

    scale = sw / max(sz)
    toScreen = lambda n: (int(round((n.x - bl[0]) * scale) + padding), int(round((n.y - bl[1]) * scale) + padding))

    for pt in ppt:
        d.ellipse(circleBBX(toScreen(pt), 2), None, dbgBrush)

    # Edges
    if drawEdges or not modeLombardi:
        for e in edges():
            d.line(ptsToList(toScreen(e.p), toScreen(e.q)), edgePen)

    if modeLombardi:
        for e in edges():
            if abs(e.diffAngle(e.p)) < (m.pi / 40):
                d.line(ptsToList(toScreen(e.p), toScreen(e.q)), arcPen)
            else:
                len = dist(e.p, e.q) / 10
                pCtrl = Vector(e.p[0] + m.cos(e.tanAngle(e.p)) * len, e.p[1] + m.sin(e.tanAngle(e.p)) * len)
                qCtrl = Vector(e.q[0] + m.cos(e.tanAngle(e.q)) * len, e.q[1] + m.sin(e.tanAngle(e.q)) * len)
                center = intersection((-pCtrl.y + (e.p.x + e.p.y), pCtrl.x + (e.p.y - e.p.x)), e.p,
                                      (-qCtrl.y + (e.q.x + e.q.y), qCtrl.x + (e.q.y - e.q.x)), e.q)
                radius = dist(toScreen(center), toScreen(e.p))
                # pygame.draw.circle(screen, (0,0,255), toScreen(center), int(round(radius)), 1)
                # big arc or small arc?
                intr = intersection(e.p, pCtrl, e.q, qCtrl)
                bigArc = dist(intr, pCtrl) > dist(intr, e.p)
                pa = -fromToAngle(center, e.p)
                qa = -fromToAngle(center, e.q)

                if qa < pa:
                    qa += m.pi * 2
                if (bigArc and qa - pa < m.pi) or (not bigArc and qa - pa > m.pi):
                    tmp = pa
                    pa = qa
                    qa = tmp
                    qa += m.pi * 2

                # rect = pygame.Rect(toScreen(center)[0]-radius, toScreen(center)[1]-radius, radius*2, radius*2)
                # pygame.draw.rect(screen, (255,0,0), rect, 1)
                # pygame.draw.arc(screen, (0,255,0), rect, pa, qa, 2)
                d.arc(circleBBX(toScreen(center), radius), m.degrees(pa), m.degrees(qa), arcPen)

    # Tangents
    if drawTans and modeLombardi:
        for n in nodes():
            s = toScreen(n)
            for nn in n.nodes():
                endx = s[0] + tanLen * m.cos(n.tanAngle(nn))
                endy = s[1] + tanLen * m.sin(n.tanAngle(nn))
                d.line(ptsToList(s, (endx, endy)), tanPen)

    # Nodes
    for n in nodes():
        d.ellipse(circleBBX(toScreen(n), 5), None, nodeBrush)

    # screen.blit(pygame.transform.flip(screen, False, True), (0,0))#flip y

    d.flush()
    img.save(file + score + add + ".png")
    return


def circleBBX(c, r):
    return [c[0] - r, c[1] - r, c[0] + r, c[1] + r]


def ptsToList(pt1, pt2):
    return [pt1[0], pt1[1], pt2[0], pt2[1]]


live = True
modeLombardi = True
# do('triGraph.dot')
# drawToFile('triGraph.dot')
do('runGraph.dot')
drawToFile('runGraph.dot')
# do("grafo123.21.graphml")
# drawToFile("grafo123.21.graphml")
