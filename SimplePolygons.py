import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import math

class MatrixNode:
    # cell is coordinates of cell as tuple
    def __init__(self, cell, next, direction):
        self.cell = cell 
        self.next = next
        self.direction = direction

class PolygonNode:
    def __init__(self, point, next, stitch):
        self.point = point 
        self.next = next
        self.stitch = stitch # '' if not stitch Node

# input is number of points
# returns random points in specified cell
# h_coord is horizontal coordinate of cell
# v_coord is vertical coordinate of cell
# dimension is dimension of cell, should be integer fraction of [0,1]X[0,1] box
def random_points(num_points, h_coord, v_coord, dimension):
    x_coords = [0]*(num_points)
    y_coords = [0]*(num_points)
    for i in range(num_points):
        x_coords[i] = random.random()*dimension + h_coord*dimension
        y_coords[i] = random.random()*dimension + v_coord*dimension

    return [x_coords, y_coords]

# input is cell coordinates, and matrix of visited cells
# returns array string of available cells to visit
def available_cells(h_coord, v_coord, visited):
    available=[]
    # left
    if h_coord > 0 and visited[h_coord - 1][v_coord]==0:
        available = available + ['left']
    # right
    if h_coord < visited.shape[0]-1 and visited[h_coord + 1][v_coord]==0:
        available = available + ['right']
    # down
    if v_coord > 0 and visited[h_coord][v_coord-1]==0:
        available = available + ['down']
    # up
    if v_coord < visited.shape[1]-1 and visited[h_coord][v_coord+1]==0:
        available = available + ['up']

    return available

# input is maximum number of cells in the path,
# and dimension of matrix
# returns the root of the linked list, and the number of nodes/cells
def matrix_path(max_num_cells, matrix_dim):
    visited = np.zeros((matrix_dim, matrix_dim))
    last = MatrixNode((0,0), None, None)
    visited[0,0]=1

    num_cells = 1
    deadEnd = False
    current = last
    while num_cells<=max_num_cells and not deadEnd:
        available= available_cells(current.cell[0], current.cell[1], visited)
        if len(available) == 0:
            deadEnd = True
        else:
            choice = random.choice(available)
            if choice == 'left':
                visited[current.cell[0]-1][current.cell[1]] = 1
                newNode = MatrixNode((current.cell[0]-1, current.cell[1]), current, 'right')
            elif choice == 'right':
                visited[current.cell[0]+1][current.cell[1]] = 1
                newNode = MatrixNode((current.cell[0]+1, current.cell[1]), current, 'left')
            elif choice == 'down':
                visited[current.cell[0]][current.cell[1]-1] = 1
                newNode = MatrixNode((current.cell[0], current.cell[1]-1), current, 'up')
            elif choice == 'up':
                visited[current.cell[0]][current.cell[1]+1] = 1
                newNode = MatrixNode((current.cell[0], current.cell[1]+1), current, 'down')
            num_cells+=1
            current = newNode
    test = current

    return current, num_cells

# returns booleanw whether c is above the line ab
# inputs are points as tuples
def aboveLine(a,b,c):
    slope = (b[1]-a[1])/(b[0]-a[0])
    residual = c[1]-(slope*(c[0]-a[0])+a[1])
    return residual>0

# computes polygon in cell and entry points for stitching
# input is number of points,
# stitching directions as strings in array
# constructs linked list of the polygon, and returns pointers to the node prior
# to the stitching nodes
def hull(n, h_coord, v_coord, dimension, stitches):

    points = random_points(n, h_coord, v_coord, dimension)
    x_coords = points[0]
    y_coords = points[1]
    stitch_points = ['']*n # marks which points are stitch points
    delta = dimension/(2*int(math.sqrt(n))+1)

    index = np.lexsort((y_coords, x_coords)) # sorts by x coordinate and then by y coordinate
    
    ordered_points = [(x_coords[i], y_coords[i], stitch_points[i]) for i in index]
    
    # add points on boundary of cell that will be stitched
    for direction in stitches:
        if direction == 'left':
            point1 = (h_coord*dimension,(v_coord+.5)*dimension-delta,'left')
            point2 = (h_coord*dimension,(v_coord+.5)*dimension+delta,'left')
            ordered_points = [point1] + [point2] + ordered_points
        elif direction == 'right':
            point1 = ((h_coord+1)*dimension,(v_coord+.5)*dimension-delta,'right')
            point2 = ((h_coord+1)*dimension,(v_coord+.5)*dimension+delta,'right')
            ordered_points = ordered_points + [point1] + [point2]
        elif direction == 'down' or direction == 'up':
            # binary search into points around halfway to find appropriate delta
            middle_ind = int(len(ordered_points)/2)
            found = False
            cellMiddle = (h_coord+.5)*dimension
            while not found and middle_ind >= 0 and middle_ind < len(ordered_points)-1:
                if ordered_points[middle_ind][0]<=cellMiddle and ordered_points[middle_ind+1][0]>=cellMiddle:
                    found = True
                else:
                    if ordered_points[middle_ind+1][0]<cellMiddle:
                        middle_ind+=1
                    elif ordered_points[middle_ind][0]>cellMiddle:
                        middle_ind-=1
            if middle_ind < 0: # if all points right of halfway x-coordinate
                newdelta = ordered_points[0][0]-cellMiddle
            elif middle_ind ==len(ordered_points)-1: # if all points below halfway x-coordinate
                newdelta = cellMiddle-ordered_points[0][0]
            else:
                newdelta = min([cellMiddle-ordered_points[middle_ind][0], ordered_points[middle_ind+1][0]-cellMiddle])
            
            if delta >= newdelta:
                delta = newdelta*.9

            if direction == 'down':
                point1 = ((h_coord+.5)*dimension-delta, v_coord*dimension, 'down')
                point2 = ((h_coord+.5)*dimension+delta, v_coord*dimension, 'down')
            elif direction == 'up':
                point1 = ((h_coord+.5)*dimension-delta,(v_coord+1)*dimension,'up')
                point2 = ((h_coord+.5)*dimension+delta,(v_coord+1)*dimension,'up')

            if middle_ind < 0:
                ordered_points = [point1] + [point2] + ordered_points
            elif middle_ind == len(ordered_points)-1:
                ordered_points = ordered_points + [point1] + [point2]
            else:
                ordered_points = ordered_points[:middle_ind+1] + [point1] + [point2] + ordered_points[middle_ind+1:]

    left = ordered_points[0]
    left_coords = (left[0],left[1])
    right = ordered_points[len(ordered_points)-1]
    right_coords = (right[0], right[1])

    # converting to linked list
    leftNode = PolygonNode(left_coords, None, left[2])
    rightNode = PolygonNode(right_coords, None, right[2])
    previousTop = leftNode
    previousBottom = leftNode
    prevStitchNodes = [] # array of nodes before stitch nodes
    for point in ordered_points[1:len(ordered_points)-1]:
        newNode = PolygonNode((point[0],point[1]), None, point[2])
        if aboveLine(left_coords, right_coords, (point[0], point[1])):
            previousTop.next = newNode
            if dimension!=1:
                if newNode.stitch != '' and previousTop.stitch != newNode.stitch:
                    prevStitchNodes = prevStitchNodes + [previousTop]
            previousTop = newNode
        else:
            newNode.next = previousBottom
            if dimension!=1:
                if previousBottom.stitch != '' and newNode.stitch != previousBottom.stitch:
                    prevStitchNodes = prevStitchNodes + [newNode]
            previousBottom = newNode
    

    if dimension!=1:
        if rightNode.stitch != '' and previousTop.stitch != rightNode.stitch:
            prevStitchNodes = prevStitchNodes + [previousTop]
    previousTop.next = rightNode
    if dimension!=1:
        if previousBottom.stitch != '' and rightNode.stitch != previousBottom.stitch:
            prevStitchNodes = prevStitchNodes + [rightNode]
    rightNode.next = previousBottom

    if dimension ==1:
        return leftNode

    return prevStitchNodes

def directionFlip(direction):
    if direction == 'left':
        return 'right'
    if direction == 'right':
        return 'left'
    if direction == 'up':
        return 'down'
    if direction == 'down':
        return 'up'

# stitches two polygons together
# returns the remaining unstitched part of postStitchNodes
# input is PolygonNode of polygons before and after
def stitch(prevStitchNode, postStitchNodes):
    prevDirection = prevStitchNode.next.stitch

    # matching up the corresponding stitches in the cells
    if prevDirection == 'left':
        for index, i in enumerate(postStitchNodes):
            if i.next.stitch == 'right':
                postNode = i
                retNode = postStitchNodes[(index+1)%len(postStitchNodes)]
                break
    elif prevDirection == 'right':
        for index, i in enumerate(postStitchNodes):
            if i.next.stitch == 'left':
                postNode = i
                retNode = postStitchNodes[(index+1)%len(postStitchNodes)]
                break
    elif prevDirection == 'down':
        for index, i in enumerate(postStitchNodes):
            if i.next.stitch == 'up':
                postNode = i
                retNode = postStitchNodes[(index+1)%len(postStitchNodes)]
                break
    elif prevDirection == 'up':
        for index, i in enumerate(postStitchNodes):
            if i.next.stitch == 'down':
                postNode = i
                retNode = postStitchNodes[(index+1)%len(postStitchNodes)]
                break
    # appropriately stitch for up-down scenarios by choosing the one with the smaller gap 
    if prevDirection == 'up' or prevDirection == 'down':
        if prevStitchNode.next.point[0]-prevStitchNode.next.next.point[0]>postNode.next.point[0]-postNode.next.next.point[0]:
            temp = prevStitchNode
            prevStitchNode = postNode
            postNode = temp
    b = prevStitchNode.next.next
    prevStitchNode.next.next=postNode.next.next.next
    postNode.next = b

    return retNode

def simple_polygon(n):
    # if n is too small, default to just return single cell hull
    if n<40:
        start = hull(n, 0, 0, 1, [])
        return listToArrays(start, n)
    max_cells = int(n/10)
    matrix_dimension = int(math.sqrt(max_cells))
    start, num_cells = matrix_path(max_cells, matrix_dimension)
    points_per_cell = int(n/num_cells)
    remainder = n - points_per_cell*num_cells
    if remainder>0:
        offset = 1
        remainder-=1
    else:
        offset = 0
    prevStitchNode = hull(points_per_cell-2+offset, start.cell[0], start.cell[1], 1/matrix_dimension, [start.direction])
    prevStitchNode = prevStitchNode[0]
    prevDirection = directionFlip(start.direction)
    current = start.next
    cell_num = 1
    while current.next is not None:
        if remainder>0:
            offset = 1
            remainder-=1
        else:
            offset = 0
        postStitchNodes = hull(points_per_cell-2+offset, current.cell[0], current.cell[1], 1/matrix_dimension, [prevDirection, current.direction])
        prevDirection = directionFlip(current.direction)
        prevStitchNode = stitch(prevStitchNode, postStitchNodes)
        cell_num+=1
        current = current.next

    postStitchNodes = hull(points_per_cell, current.cell[0], current.cell[1], 1/matrix_dimension, [prevDirection])
    prevStitchNode = stitch(prevStitchNode, postStitchNodes)
    cell_num+=1

    return listToArrays(prevStitchNode, n)


#converting linked list back to arrays
def listToArrays(start, n):
    x_coords = [0]*(n+1)
    y_coords = [0]*(n+1)
    x_coords[n]= start.point[0]
    y_coords[n]= start.point[1]
    current = start
    for i in range(n):
        x_coords[i]= current.point[0]
        y_coords[i]= current.point[1]
        current = current.next
    
    return (x_coords, y_coords)

# input is polygon coordinates
def polygon_plot(x_coords, y_coords):
    #plt.figure(figsize=(4, 4))
    # x_min = min(x_coords)
    # x_range = max(x_coords) - x_min
    # y_min = min(y_coords)
    # y_range = max(y_coords) - y_min
    # dimension = max(x_range, y_range)
    plt.plot(x_coords, y_coords)
    # plt.axis([x_min, x_min + dimension, y_min, y_min + dimension])
    plt.title(str(len(x_coords)-1) + ' vertices')
    plt.show()

def main():
    n = int(sys.argv[1])
    polygon = simple_polygon(n)
    polygon_plot(polygon[0], polygon[1])

if __name__ == '__main__':
    main()