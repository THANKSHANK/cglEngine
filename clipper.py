"""
This module contains the implementation of the Cohen-Sutherland line clipping algorithm
and the Sutherland-Hodgman polygon clipping algorithm.
"""
__author__ = "Zihan Wang"

from vertex import Vertex
from math import sqrt

INSIDE = 0  # 0000
LEFT = 1  # 0001
RIGHT = 2  # 0010
BOTTOM = 4  # 0100
TOP = 8  # 1000


def compute_outcode(x, y, top, bottom, right, left):
    """
    Compute the outcode of a vertex with respect to the clipping window.
    :param x:  x-coordinate
    :param y:  y-coordinate
    :param top:  top
    :param bottom:  bottom
    :param right:  right
    :param left:  left
    :return:  outcode
    """
    code = INSIDE
    if x < left:  # to the left of clipping window
        code |= LEFT
    elif x > right:  # to the right of clipping window
        code |= RIGHT
    if y < bottom:  # below the clipping window
        code |= BOTTOM
    elif y > top:  # above the clipping window
        code |= TOP
    return code


def clipLine(p0, p1, top, bottom, right, left):
    """
    Implement the Cohen-Sutherland line clipping algorithm.
    :param p0:  vertex p0
    :param p1:  vertex p1
    :param top:  top
    :param bottom:  bottom
    :param right:  right
    :param left:  left
    :return:  list of vertices after clipping
    """
    outcode0 = compute_outcode(p0.x, p0.y, top, bottom, right, left)
    outcode1 = compute_outcode(p1.x, p1.y, top, bottom, right, left)
    accept = False

    while True:
        if not (outcode0 | outcode1):
            # Trivial acceptance
            accept = True
            break
        elif outcode0 & outcode1:
            # Trivial rejection
            break
        else:
            # Subdivide the line and test the outcodes again
            x, y = 0.0, 0.0
            if outcode0:
                code_out = outcode0
            else:
                code_out = outcode1

            if code_out & TOP:
                x = p0.x + (p1.x - p0.x) * (top - p0.y) / (p1.y - p0.y)
                y = top
            elif code_out & BOTTOM:
                x = p0.x + (p1.x - p0.x) * (bottom - p0.y) / (p1.y - p0.y)
                y = bottom
            elif code_out & RIGHT:
                y = p0.y + (p1.y - p0.y) * (right - p0.x) / (p1.x - p0.x)
                x = right
            elif code_out & LEFT:
                y = p0.y + (p1.y - p0.y) * (left - p0.x) / (p1.x - p0.x)
                x = left

            if code_out == outcode0:
                p0.x, p0.y = x, y
                outcode0 = compute_outcode(p0.x, p0.y, top, bottom, right, left)
            else:
                p1.x, p1.y = x, y
                outcode1 = compute_outcode(p1.x, p1.y, top, bottom, right, left)

    if accept:
        return [Vertex(p0.x, p0.y, p0.r, p0.g, p0.b), Vertex(p1.x, p1.y, p1.r, p1.g, p1.b)]
    else:
        return []


def clipPoly(vertices: list, top: int, bottom: int, right: int, left: int, useNorm=False):
    """
    implement the Sutherland-Hodgman algorithm to clip a polygon.
    :param vertices:  list of vertices
    :param top:     top edge of the clip window
    :param bottom:  bottom edge of the clip window
    :param right:   right edge of the clip window
    :param left:    left edge of the clip window
    :param useNorm:     use normalized coordinates
    :return:  list of vertices after clipping
    """
    # clip along each edge at a time
    retPoly = shpc(vertices, 'top', top, useNorm)
    if len(retPoly) > 0:
        # Clip along the bottom edge
        retPoly = shpc(retPoly, 'bottom', bottom, useNorm)
    if len(retPoly) > 0:
        # Clip along the right edge
        retPoly = shpc(retPoly, 'right', right, useNorm)
    if len(retPoly) > 0:
        # Clip along the left edge
        retPoly = shpc(retPoly, 'left', left, useNorm)

    # Convert the clipped polygon vertices to triangles
    retPoly = polyToTriangles(retPoly)
    unzipped = []
    for i in range(len(retPoly)):
        for j in range(len(retPoly[i])):
            unzipped.append(retPoly[i][j])

    return unzipped


def shpc(vertices, edge, edgeVal, useNorm) -> list[Vertex]:
    """
    Sutherland-Hodgman polygon clipping algorithm.
    :param vertices:  list of vertices
    :param edge:  edge to clip against
    :param edgeVal:  value of the edge
    :param useNorm:  use normalized coordinates
    :return:  list of vertices after clipping
    """
    new_vertices = []  # List to store the new vertices after clipping
    prev_vertex = vertices[-1]  # Start with the last vertex
    prev_inside = is_inside(prev_vertex, edge, edgeVal)  # Check if the last vertex is inside the clip edge

    for current_vertex in vertices:
        current_inside = is_inside(current_vertex, edge,
                                   edgeVal)  # Check if the current vertex is inside the clip edge
        if not current_inside and not prev_inside:
            # Both this and the previous vertices are outside.
            # Do nothing.
            pass
        if current_inside and not prev_inside:
            # Current vertex is inside, but previous vertex is outside.
            # Add the intersection, then the current vertex.

            if useNorm:
                intersection = compute_intersection(prev_vertex, current_vertex, edge, edgeVal)
                u = sqrt((intersection.x - prev_vertex.x) ** 2 + (intersection.y - prev_vertex.y) ** 2) / sqrt(
                    (current_vertex.x - prev_vertex.x) ** 2 + (current_vertex.y - prev_vertex.y) ** 2)
                intersection.r = prev_vertex.r + (current_vertex.r - prev_vertex.r) * u
                intersection.g = prev_vertex.g + (current_vertex.g - prev_vertex.g) * u
                intersection.b = prev_vertex.b + (current_vertex.b - prev_vertex.b) * u
                new_vertices.append(intersection)

            else:
                new_vertices.append(
                    compute_intersection(prev_vertex, current_vertex, edge, edgeVal))  # Add the intersection
            new_vertices.append(current_vertex)  # Add the current vertex
        elif not current_inside and prev_inside:
            # Current vertex is outside, but previous vertex is inside.
            # Add only the intersection.

            if useNorm:
                intersection = compute_intersection(prev_vertex, current_vertex, edge, edgeVal)
                u = sqrt((intersection.x - prev_vertex.x) ** 2 + (intersection.y - prev_vertex.y) ** 2) / sqrt(
                    (current_vertex.x - prev_vertex.x) ** 2 + (current_vertex.y - prev_vertex.y) ** 2)
                intersection.r = prev_vertex.r + (current_vertex.r - prev_vertex.r) * u
                intersection.g = prev_vertex.g + (current_vertex.g - prev_vertex.g) * u
                intersection.b = prev_vertex.b + (current_vertex.b - prev_vertex.b) * u

                new_vertices.append(intersection)
            else:
                new_vertices.append(
                    compute_intersection(prev_vertex, current_vertex, edge, edgeVal))  # Add the intersection
        elif current_inside:
            # Both this and the previous vertices are inside.
            new_vertices.append(current_vertex)  # Add the current vertex

        prev_vertex = current_vertex
        prev_inside = current_inside

    return new_vertices


def is_inside(vertex, edge, edgeVal):
    """
    Check if a vertex is inside the clip edge.
    :param vertex:  vertex
    :param edge:  edge
    :param edgeVal:  value of the edge
    :return:  True if the vertex is inside the edge, False otherwise
    """
    if edge == 'top':
        return vertex.y <= edgeVal
    elif edge == 'bottom':
        return vertex.y >= edgeVal
    elif edge == 'right':
        return vertex.x <= edgeVal
    elif edge == 'left':
        return vertex.x >= edgeVal
    else:
        return False


def compute_intersection(vertex1, vertex2, edge, edgeVal) -> Vertex:
    """
    Compute the intersection of the edge with the clip window.
    :param vertex1:  vertex1
    :param vertex2:  vertex2
    :param edge:  edge
    :param edgeVal:  value of the edge
    :return:  intersection vertex
    """
    if edge == 'top':
        x = vertex1.x + (vertex2.x - vertex1.x) * (edgeVal - vertex1.y) / (vertex2.y - vertex1.y)
        y = edgeVal
        r = vertex1.r + (vertex2.r - vertex1.r) * (edgeVal - vertex1.y) / (vertex2.y - vertex1.y)
        g = vertex1.g + (vertex2.g - vertex1.g) * (edgeVal - vertex1.y) / (vertex2.y - vertex1.y)
        b = vertex1.b + (vertex2.b - vertex1.b) * (edgeVal - vertex1.y) / (vertex2.y - vertex1.y)
    elif edge == 'bottom':
        x = vertex1.x + (vertex2.x - vertex1.x) * (edgeVal - vertex1.y) / (vertex2.y - vertex1.y)
        y = edgeVal
        r = vertex1.r + (vertex2.r - vertex1.r) * (edgeVal - vertex1.y) / (vertex2.y - vertex1.y)
        g = vertex1.g + (vertex2.g - vertex1.g) * (edgeVal - vertex1.y) / (vertex2.y - vertex1.y)
        b = vertex1.b + (vertex2.b - vertex1.b) * (edgeVal - vertex1.y) / (vertex2.y - vertex1.y)
    elif edge == 'right':
        x = edgeVal
        y = vertex1.y + (vertex2.y - vertex1.y) * (edgeVal - vertex1.x) / (vertex2.x - vertex1.x)
        r = vertex1.r + (vertex2.r - vertex1.r) * (edgeVal - vertex1.x) / (vertex2.x - vertex1.x)
        g = vertex1.g + (vertex2.g - vertex1.g) * (edgeVal - vertex1.x) / (vertex2.x - vertex1.x)
        b = vertex1.b + (vertex2.b - vertex1.b) * (edgeVal - vertex1.x) / (vertex2.x - vertex1.x)
    elif edge == 'left':
        x = edgeVal
        y = vertex1.y + (vertex2.y - vertex1.y) * (edgeVal - vertex1.x) / (vertex2.x - vertex1.x)
        r = vertex1.r + (vertex2.r - vertex1.r) * (edgeVal - vertex1.x) / (vertex2.x - vertex1.x)
        g = vertex1.g + (vertex2.g - vertex1.g) * (edgeVal - vertex1.x) / (vertex2.x - vertex1.x)
        b = vertex1.b + (vertex2.b - vertex1.b) * (edgeVal - vertex1.x) / (vertex2.x - vertex1.x)
    else:
        x, y = 0, 0
        r, g, b = 0, 0, 0
    return Vertex(x, y, r, g, b)


def polyToTriangles(retPoly):
    """
    modify the list of vertices to a list of triangles vertices.
    :param retPoly: list of  Vertex objects – vertices of polygon.
    :return: list of triplets of Vertex objects – vertices of triangles in the type of list.
    """
    triangles = []
    if len(retPoly) < 3:
        return []
    for i in range(1, len(retPoly) - 1):
        triangles.append([retPoly[0], retPoly[i], retPoly[i + 1]])

    return triangles
