from __future__ import division
import math as m
import re

adjMat = {}
nodeList = []
nodeDict = {}
edgeList = []


def nodes():
    for x in nodeList:
        yield x


def edges():
    for x in edgeList:
        yield x


class Node:
    def __init__(self, id):
        self.id = id
        self.x = 0.0
        self.y = 0.0
        self.rf = 0.0
        self.fx = 0.0
        self.fy = 0.0
        self.tx = 0.0  # tangential
        self.ty = 0.0  # tangential
        self.degree = 0  # assigned on init
        self.curTan = -1
        self.angle = 0.0
        self.aIncrement = 0.0  # assigned on init
        nodeDict[id] = self
        nodeList.append(self)

    def __getitem__(self, key):
        if key == 0:
            return self.x
        if key == 1:
            return self.y
        return None

    def nodes(self):
        print("testing...")
        for x in adjMat[self.id].intr():
            yield nodeDict[x]

    def edges(self):
        for x in adjMat[self.id].intr():
            yield x

    def nextTanAngle(self):
        self.curTan += 1
        return self.aIncrement * self.curTan

    def tanAngle(self, n):  # n is to
        return getEdge(self, n).tanAngle(self)

    def tanAngleRel(self, n):  # n is to
        return getEdge(self, n).tanAngleRel(self)

    def edgeAngle(self, n):  # n is to
        return getEdge(self, n).edgeAngle(self)

    def diffAngle(self, n):  # n is to
        return getEdge(self, n).diffAngle(self)

    def setAngle(self, n, angle):  # n is to
        return getEdge(self, n).setAngle(self, angle)

    def addForce(self, n, f):  # n is to
        return getEdge(self, n).addForce(self, f)


class Edge:
    def __init__(self, p, q):
        self.dist = 0.0  # set on init
        self.factor = 0.0  # set on init
        self.cx = 0.0  # ctrl pt
        self.cy = 0.0  # ctrl pt
        self.pcx = 0.0  # ctrl pt
        self.pcy = 0.0  # ctrl pt
        self.x = 0.0  # midpt
        self.y = 0.0  # midpt
        self.fx = 0.0  # midpt
        self.fy = 0.0  # midpt
        edgeList.append(self)
        if p not in nodeDict:
            Node(p)
        if q not in nodeDict:
            Node(q)
        if p not in adjMat:
            adjMat[p] = {}
        if q not in adjMat:
            adjMat[q] = {}
        adjMat[p][q] = self
        adjMat[q][p] = self
        self.p = nodeDict[p]
        self.q = nodeDict[q]
        self.pa = 0.0  # p angle, set on init
        self.qa = 0.0  # q angle, set on init
        self.pf = 0.0  # p rot force
        self.qf = 0.0  # q rot force

    def __getitem__(self, key):
        if key == 0:
            return self.x
        if key == 1:
            return self.y
        return None

    def tanAngle(self, n):  # n is from
        return pie2(n.angle + self.tanAngleRel(n))

    def tanAngleRel(self, n):  # n is from
        return self.pa if n == self.p else self.qa

    def edgeAngle(self, n):  # n is from
        n1 = (self.p if n == self.p else self.q)
        n2 = (self.p if n == self.q else self.q)

        return pie2(m.atan2((n2.y - n1.y), (n2.x - n1.x)))

    def edgeAngleCtrl(self, n):  # n is from
        n1 = (self.p if n == self.p else self.q)
        n2 = self

        return pie2(m.atan2((n2.y - n1.y), (n2.x - n1.x)))

    # return pie2(m.atan2(-(n2.y-n1.y),(n2.x-n1.x)));
    # return pie2(2*m.pi-m.atan2((n2.y-n1.y),(n2.x-n1.x)));

    def diffAngle(self, n):  # n is from
        return pie(pie(self.tanAngle(n)) - pie(self.edgeAngle(n)))

    def setAngle(self, n, angle):  # n is from
        if n == self.p:
            self.pa = angle
        else:
            self.qa = angle

    def addForce(self, n, f):  # n is from
        if n == self.p:
            self.pf += f
        else:
            self.qf += f

    def ctrlPt(self):
        return Vector(self.cx, self.cy)


class Vector:  # Me+Vec2D
    __slots__ = ['x', 'y']

    def __init__(self, x_or_pair, y=None):
        if y is None:
            self.x = x_or_pair[0]
            self.y = x_or_pair[1]
        else:
            self.x = x_or_pair
            self.y = y

    def __getitem__(self, key):
        if key == 0:
            return self.x
        if key == 1:
            return self.y
        raise IndexError("Invalid subscript " + str(key) + " to Vector")

    def __len__(self):
        return 2

    def __add__(self, v):
        return Vector(self.x + v.x, self.y + v.y)

    def __sub__(self, v):
        return Vector(self.x - v.x, self.y - v.y)

    def __str__(self):
        return "(%.2f,%.2f)" % (self.x, self.y)

    def __mul__(self, factor):
        return Vector(self.x * factor, self.y * factor)

    def __rmul__(self, factor):
        return Vector(self.x * factor, self.y * factor)

    def __truediv__(self, factor):
        return Vector(self.x / factor, self.y / factor)

    def normalize(self):
        return self.scale(1 / dist((0, 0), self))

    def scale(self, k):
        return Vector(self.x * k, self.y * k)

    def angle(self):
        return pie2(m.atan2(self.y, self.x))


def getEdge(p, q):
    return adjMat[p.id][q.id]


def dist(n1, n2):
    return m.sqrt(dist2(n1, n2))


def dist2(n1, n2):
    return (n1[0] - n2[0]) * (n1[0] - n2[0]) + (n1[1] - n2[1]) * (n1[1] - n2[1])


def fromToAngle(f, t):
    return (Vector(t) - Vector(f)).angle()


def mag(x, y):
    return m.sqrt(x * x + y * y)


def pie(rad):  # -180 t 180
    rad = pie2(rad)
    return rad if rad <= m.pi else rad - 2 * m.pi


def pie2(rad):  # 0 to 360
    return rad % (2 * m.pi)


def initParse():
    adjMat.clear()
    del nodeList[:]
    nodeDict.clear()
    del edgeList[:]


def intersection(p1, p2, p3, p4):
    x1, x2, x3, x4 = p1[0], p2[0], p3[0], p4[0]
    y1, y2, y3, y4 = p1[1], p2[1], p3[1], p4[1]
    x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    return Vector(x, y)


# support dot or graphml
def parseFile(file):
    initParse()
    if file[-3:] == "adj":
        for line in open(file, 'r'):
            p = line.split(' ')[0]
            rest = line.split(' ')[1:]
            for q in rest:
                addOneHelper(p, q)
        return

    f = open(file, 'r').read()
    if file[-3:] == "dot":
        regex = r"(\w+) -- (\w+)"
    elif file[-3:] == "txt":  # Mathematica
        f = f.split('Line[')[1].split(']')[0] if 'Line' in f else f.split('}}}, {{{')[0]
        regex = r"{(\d+), (\d+)}"
    else:
        regex = r"source=\"(\w+)\" target=\"(\w+)\""
    # if decide to init nodes, must do first cuz edge auto add
    for p, q in re.findall(regex, f):
        addOneHelper(p, q)


def addOneHelper(p, q):
    if p not in nodeDict or q not in nodeDict or p not in adjMat or q not in adjMat or q not in adjMat[p]:
        Edge(p, q)


def permutations(iterable, r=None):  # from py site
    # permutations('ABCD', 2) --> AB AC AD BA BC BD CA CB CD DA DB DC
    # permutations(range(3)) --> 012 021 102 120 201 210
    pool = tuple(iterable)
    n = len(pool)
    r = n if r is None else r
    if r > n:
        return
    indices = range(n)
    cycles = range(n, n - r, -1)
    yield tuple(pool[i] for i in indices[:r])
    while n:
        for i in reversed(range(r)):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i + 1:] + indices[i:i + 1]
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                yield tuple(pool[i] for i in indices[:r])
                break
        else:
            return


def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


def bezier(pp, steps=30):
    """
    Calculate a Bézier curve from 4 control points and return a list of the resulting points.
	2007 Victor Blomqvist
	http://www.pygame.org/wiki/BezierCurve?parent=CookBook -
	The function uses the forward differencing algorithm described here:
	http://www.niksula.cs.hut.fi/~hkankaan/Homepages/bezierfast.html
    """
    p = [Vector(ppp) for ppp in pp]
    t = 1.0 / steps
    temp = t * t

    f = p[0]
    fd = 3 * (p[1] - p[0]) * t
    fdd_per_2 = 3 * (p[0] - 2 * p[1] + p[2]) * temp
    fddd_per_2 = 3 * (3 * (p[1] - p[2]) + p[3] - p[0]) * temp * t

    fddd = 2 * fddd_per_2
    fdd = 2 * fdd_per_2
    fddd_per_6 = fddd_per_2 / 3.0

    points = []
    for x in range(steps):
        points.append(f)
        f += fd + fdd_per_2 + fddd_per_6
        fd += fdd + fddd_per_2
        fdd += fddd
        fdd_per_2 += fddd_per_2
    points.append(f)
    return points


"""
a = Vector(.34 , .23)
b = Vector(.34 , .65)
c = Vector(.33 , .76)
d = 2*(a.x*(b.y-c.y) + b.x*(c.y-a.y) + c.x*(a.y-b.y))
print d
if d == 0: print "use midpoint of edge"
cx = ((a.y**2+a.x**2)*(b.y-c.y) + (b.y**2+b.x**2)*(c.y-a.y) + (c.y**2+c.x**2)*(a.y-b.y))/d
cy = ((a.y**2+a.x**2)*(c.x-b.x) + (b.y**2+b.x**2)*(a.x-c.x) + (c.y**2+c.x**2)*(b.x-a.x))/d
ct = Vector(cx,cy)
print cx,cy
print ((dist(ct,a) == dist(ct,b)) and (dist(ct,b) == dist(ct,c)))
"""
