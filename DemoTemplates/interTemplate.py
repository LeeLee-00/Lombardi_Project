#
# A template for a tool to aid in developing new graph algorithms
#
# Dependencies: Graphviz, pydot, python2
#
# For more functions, check out pydot documentation at
# http://code.google.com/p/pydot/downloads/detail?name=pydot.html
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


def main():

    # Choose from neato, fdp, sfdp, dot, circo, twopi
    gvLayout = 'circo'

    # Read original file and read it into pydot

    # I use command line functions for flexibility and familiarity
    # Check out the documentation for more functions
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


    step = 0
    bar = 5
    maxiter = 2

    for foo in range(bar):

        for n in nodePlusses:

            instruction = "fdp -Gmaxiter=" + maxiter + " .out > .new"
            call(instruction, shell=True)
            graph = pydot.graph_from_dot_file(".new")


            # Do some calculations





            

            
            # Move n
            print "Moving point from " + n.node.get_pos()
            print "Moving point to " + "(" + str(new_x) + "," + str(new_y) + ")"
            centralNode.move(Point(new_x, new_y))

            # Make a new image file
            out = open('.out', 'w')
            out.write(graph.to_string())
            out.close()
            # neato -n -s renders the graph with the given positions
            # and uses neato to render the edges.
            instruction = "neato -Tpng -s -n .out > " + str(step)+ ".png"
            call(instruction, shell=True)

            print "---"
            step +=1

            

    # Lastly, render the graph with circular edges
    # It might be possible to do this with dotty and after specifying shape
    # of edges in the DOT file

    # Remove hidden files
    instruction = "rm -f .orig .out .new"
    call(instruction, shell=True)


if __name__ == '__main__':
    main()

