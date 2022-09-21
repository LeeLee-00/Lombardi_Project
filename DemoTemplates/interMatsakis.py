#
# An implementation of Matsakis' Lombardi graph drawing algorithm.
#
# Usage: python interMatsakis.py graph.dot
#
# Produces .png files named 0.png - n.png at every step of the algorithm
# as well as copious terminal output
#
# Only does 3 iterations by default to keep from freezing due to the
# protruding edge problem.
#
# Nicolaos Matsakis. Transforming a random graph drawing into a lombardi drawing.
# CoRR, abs/1012.2202, 2010.
#
# Author: Katie Cunningham, University of Arizona, katiec42@email.arizona.edu

import pydot
import math
import sys
from subprocess import call

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, otherPoint):
        return math.sqrt( (self.x - otherPoint.x) * (self.x - otherPoint.x) + 
                      (self.y - otherPoint.y) * (self.y - otherPoint.y) )

class NodePlus:
    def __init__(self, node, graph):
        self.node = node
        edges = graph.get_edges()
        self.degree = calculateDegree(node, edges, graph)
        pos = node.get_pos().strip('"').split(',')
        self.location = Point(float(pos[0]), float(pos[1]))
        self.neighbors = calculateNeighbors(node, edges, graph)

    def move(self, newLocation):
        self.location = newLocation
        self.node.set_pos('"' + str(newLocation.x) + ',' + str(newLocation.y) + '"')
    

def calculateDegree(node, edges, graph):
    d = 0
    for e in edges:
        if graph.get_node(e.get_destination())[0].get_pos() == None or \
           graph.get_node(e.get_source())[0].get_pos() == None:
            continue
        if e.get_source() == node.get_name() or \
            e.get_destination() == node.get_name():
            d+=1
    return d

def getPointFromPos(pos):
    pos = pos.strip('"').split(',')
    return Point(float(pos[0]), float(pos[1]))

def calculateNeighbors(node, edges, graph):
    neighbors = []
    for e in edges:
        if e.get_source() == node.get_name():
            if graph.get_node(e.get_destination())[0].get_pos() != None:
                neighbors.append(graph.get_node(e.get_destination())[0])
                #[0] because pydot method technically returns a list
        elif e.get_destination() == node.get_name():
            if graph.get_node(e.get_source())[0].get_pos() != None:
                neighbors.append(graph.get_node(e.get_source())[0])
    return neighbors

# Angle from the x-axis, assuming the x-axis goes through fromPt
def getAngleFromZero(fromPt, toPt):

    x1 = fromPt.x
    y1 = fromPt.y
    x2 = toPt.x
    y2 = toPt.y
    angle = math.atan2((y2-y1),(x2-x1))

    answer = angle * 180/math.pi

    return between0and360(answer)

def between0and360(deg):
    deg = deg % (360)
    while deg < 0:
        deg += 360
    while deg >= 360:
        deg -= 360
    return deg

# Assumes all inputs are pydot.Nodes, for simplicity
def sortByCounterClockwise(neighbors, centralNode, farthestNeighbor):

    centralPt = getPointFromPos(centralNode.get_pos())
    startPt = getPointFromPos(farthestNeighbor.get_pos())
    theta_start = getAngleFromZero(centralPt, startPt)
    print "sorting nodes"
    print "theta_start=" + str(theta_start)

    neighbors = sorted(neighbors, key=lambda n : calcAdjustAngle(centralPt, n, theta_start))

    return neighbors

def calcAdjustAngle(centralPt, n, theta_start):
    toAdjust = getAngleFromZero(centralPt, getPointFromPos(n.get_pos()))
    if  toAdjust >= theta_start :
        return toAdjust
    else:
        return toAdjust + 360


def main():

    gvLayout = 'circo'

    # Read original file and read it into pydot
    instruction = gvLayout + " " + sys.argv[1] + " > .orig"
    call(instruction, shell=True)
    graph = pydot.graph_from_dot_file(".orig")
    instruction = gvLayout + " -Tpng " + sys.argv[1] + " > 0.png"
    call(instruction, shell=True)

    nodePlusses = []
    nodes = graph.get_nodes()
    for n in nodes:
        if n.get_pos() != None :
            nodePlusses.append(NodePlus(n, graph))

    print "Nodes in the graph"
    for n in nodePlusses:
        print n.node.to_string()
    print

    # Sort nodes by degree
    nodePlusses = sorted(nodePlusses, key=lambda node : node.degree, reverse=True)

    step = 0

    # change this if you want more or fewer iterations
    # it's set to a low number due to the problem of exploding edges
    for foo in range(3):

        for centralNode in nodePlusses:
            print "Now bary-izing " + centralNode.node.to_string()
            print "Degree=" + str(centralNode.degree)
        
            # Don't do anything if degree <= 1
            if int(centralNode.degree) > 1:
            
                farthestNeighbor = None
                farthestDist = 0.0

                for neighbor in centralNode.neighbors:
                    
                    curDist = getPointFromPos(neighbor.get_pos())\
                              .distance(centralNode.location)
                    print neighbor.get_name() + " dist=" + str(curDist)
                    if curDist > farthestDist:
                        farthestNeighbor = neighbor
                        farthestDist = curDist
                
                print "Farthest node is " + str(farthestDist) + " away=" + \
                       farthestNeighbor.get_name()
                            
                
                # We use the tangents, starting with this farthest neighbor as our 
                # baseline, in order to calculate the arc lengths.

                x_sum = 0 # sum of all x coordinates
                y_sum = 0 # sum of all y coordinates

                print "before"
                for n in centralNode.neighbors:
                    print n.to_string()
                
                centralNode.neighbors = sortByCounterClockwise(centralNode.neighbors, centralNode.node, farthestNeighbor)

                print "after"
                for n in centralNode.neighbors:
                    print n.to_string()
                print

                # for each neighbor, find the point of its straight line endpoint
                # This is the length of the arc from the tangent to the neighbor,
                # in the direction of the tangent.
                i = 0
                start_node = centralNode.neighbors[0]
                print "start_node=" + start_node.to_string()
                start_angle = getAngleFromZero(centralNode.location, getPointFromPos(start_node.get_pos()))
                if start_angle > 180:
                    start_angle = start_angle - 360
                for n in centralNode.neighbors:
                    
                    # Calculate the arc length, t
                    # t = x * theta / sin(theta)
                    # where x is the straight distance between the two points
                    # and theta is the angle between the line segment connecting
                    # the two points, and the tangent for that point
                    
                    # x
                    straightLineLength = getPointFromPos(n.get_pos()).distance(centralNode.location)
                    print "straightLineLength=" + str(straightLineLength)

                    angleFromZero = getAngleFromZero(centralNode.location, getPointFromPos(n.get_pos()))
                    if angleFromZero > 180:
                        angleFromZero = angleFromZero - 360
                    print "angleFromZero=" + str(angleFromZero)
                             
                    tangentAngle = start_angle + i * (360 / centralNode.degree)
                    tangentAngle = between0and360(tangentAngle)
                    if tangentAngle > 180:
                        tangentAngle = tangentAngle - 360
                    print "tangentAngle=" + str(tangentAngle)
                    
                    # theta
                    theta = abs(tangentAngle - angleFromZero)
                    if theta > 180:
                        theta = theta - 360
                    print "theta=" + str(theta)
                    
                    # arclength, t
                    if theta == 0:
                        arcLength = straightLineLength
                    else:
                        arcLength = straightLineLength * (theta * math.pi / 180) / math.sin(theta * math.pi / 180)
                    print "arcLength=" + str(arcLength)
                    
                    curvyX = math.cos(tangentAngle * math.pi / 180 ) * arcLength + centralNode.location.x
                    curvyY = math.sin(tangentAngle * math.pi / 180 ) * arcLength + centralNode.location.y
                    
                    print "curvyX=" + str(curvyX)
                    print "curvyY=" + str(curvyY)
                    

                    x_sum += curvyX
                    y_sum += curvyY

                    i+=1
                    
                    print "Done with neighbor " + str(i)
                    print

                    
                step += 1
            
                # Move the central vertex to a position of "zero cumulative force"
                # In this version we're gonna say that that's the barycenter
                # i.e. the average of all the points
                d = centralNode.degree
                print "Moving point from " + centralNode.node.get_pos()
                print "Moving point to " + "(" + str(x_sum / d) + "," + str(y_sum / d) + ")"
                centralNode.move(Point(x_sum / d, y_sum / d))

                out = open('.out', 'w')
                out.write(graph.to_string())
                out.close()
                instruction = "neato -Tpng -s -n .out > " + str(step)+ ".png"
                call(instruction, shell=True)

                print "---"
                print

            

    # Lastly, render the graph with circular edges
    # It might be possible to do this with dotty and after specifying shape
    # of edges in the DOT file

    # Remove hidden files
    instruction = "rm -f .orig .out .new"
    call(instruction, shell=True)


if __name__ == '__main__':
    main()

