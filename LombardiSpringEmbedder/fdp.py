#################################################################################################
# This is a python implementation of fdp from graphviz (a force directed graph layout algorithm).
# I only ported over the essentials and left out some optimizations like the grid.
# Usage example is at the bottom.
# Can be used with simple dot files or graphml (or modify classes.py to parse other formats).
# I realize it's sparsely commented so if you need help just shoot me an email.
# Author: Roman Chernobelskiy, romanc@email.arizona.edu (University of Arizona)
# Developed on Python 2.5
#################################################################################################
from __future__ import division
import pygame
import math as m
from classes import *
from random import random as rand

#fdp settings
maxIters = 600
EXPFACTOR = 1.2
K = 0.3
K2 = K*K
useNew = True

#LIVE display settings
live = True
fps = 30
everyOtherFrame = True
tanLen = 30 #screen px
drawEdges = False

#global algorithm vars
screen = 0
clock = 0
sw,sh = 500,500#screen width&height, should be square
padding = 10#some padding for the screen
bl = [0,0]#bounding box
tr = [0,0]#bounding box

def init():
	global screen, clock
	Wd = EXPFACTOR * (K / 2.0);
	Ht = EXPFACTOR * (K / 2.0);
	
	for n in nodes():
		n.x = Wd * (2.0*rand() - 1.0)
		n.y = Ht * (2.0*rand() - 1.0)
	
	for e in edges():
		e.factor = 1.0
		e.dist = K
	
	if(live):
		pygame.init()
		screen = pygame.display.set_mode([sw+padding*2,sh+padding*2])
		screen.fill([255,255,255])
		clock = pygame.time.Clock()
	
def cool(t0, i):
	return (t0*(maxIters-i))/maxIters

def adjust(temp):
	if(temp<=0):
		print "frozen!"
		return
	
	for n in nodes():
		n.fx = 0
		n.fy = 0
	
	for i in range(len(nodeList)):
		for j in range(i+1, len(nodeList)):
			applyRep(nodeList[i], nodeList[j])
	for e in edges():
		applyAttr(e.p, e.q, e)
	updatePos(temp)

def applyRep(p, q):
	dx = q.x-p.x
	dy = q.y-p.y
	dist2 = dx*dx + dy*dy
	while dist2 == 0:
		dx = 5-m.round(rand()*10)
		dy = 5-m.round(rand()*10)
		dist2 = dx*dx + dy*dy
		
	force = 0
	if useNew: force = K2/(m.sqrt(dist2)*dist2)
	else: force = K2/dist2
	q.fx += dx * force;
	q.fy += dy * force;
	p.fx -= dx * force;
	p.fy -= dy * force;
	
def applyAttr(p, q, e):
	dx = q.x-p.x
	dy = q.y-p.y
	dist2 = dx*dx + dy*dy
	while dist2 == 0:
		dx = 5-m.round(rand()*10)
		dy = 5-m.round(rand()*10)
		dist2 = dx*dx + dy*dy
	
	dist = m.sqrt(dist2)
	force = 0
	if useNew: force = (e.factor*(dist-e.dist))/dist
	else: force = (e.factor*dist)/e.dist
	q.fx -= dx * force;
	q.fy -= dy * force;
	p.fx += dx * force;
	p.fy += dy * force;

	
def updatePos(temp):
	temp2 = temp*temp
	for i in range(2): tr[i] = 0
	for i in range(2): bl[i] = 0
	for n in nodes():
		len2 = n.fx*n.fx + n.fy*n.fy
		
		if len2 < temp2:
			n.x += n.fx
			n.y += n.fy
		else: #limit by temp
			fact = temp/m.sqrt(len2)
			n.x += n.fx * fact
			n.y += n.fy * fact
		
		#bounding box
		if n.x < bl[0]: bl[0] = n.x
		elif n.x > tr[0]: tr[0] = n.x
		if n.y < bl[1]: bl[1] = n.y
		elif n.y > tr[1]: tr[1] = n.y
	
def draw():
	tick_time = clock.tick(fps) # milliseconds since last frame
	screen.fill((255,255,255))
	#center drawing and find bb scale
	sz = (tr[0]-bl[0], tr[1]-bl[1])
	if sz[0] > sz[1]:
		tr[1]+=(sz[0]-sz[1])/2;
		bl[1]-=(sz[0]-sz[1])/2;
	else:
		tr[0]+=(sz[1]-sz[0])/2;
		bl[0]-=(sz[1]-sz[0])/2;
	
	scale = sw/max(sz)
	toScreen = lambda n: (int(round((n.x-bl[0])*scale)+padding), int(round((n.y-bl[1])*scale)+padding))
	
	for e in edges():
		pygame.draw.line(screen, (155,155,155), toScreen(e.p), toScreen(e.q), 3)
		
	for n in nodes():
		pygame.draw.circle(screen, (0,0,0), toScreen(n), 8)
	
	pygame.display.update()

def drawToFile(file):
	#use last dot as split pt, replace with svg or png or w/e
	pass
	
	
def checkQuit():
	for e in pygame.event.get():
		if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
			return True
	return False
	
def waitQuit():
	while not checkQuit(): pass

def do(file):
	parseFile(file)#parse file into global namespace
	init()#place nodes
	
	T0 = K * m.sqrt(len(nodeList)) / 5.0
	for i in range(maxIters):
		adjust(cool(T0, i))
		if live:
			if not everyOtherFrame or i%2==0: draw()
			if checkQuit(): return
	
	if live: waitQuit()
	else: drawToFile(file)

do("runGraph.dot")