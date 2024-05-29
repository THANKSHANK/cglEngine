"""
This file contains the implementation of the CGIengine class
"""
__author__ = "Zihan Wang"
import glm
from rit_window import *
from vertex import *
import numpy as np
import math

epsilon = 1e-6
class CGIengine:
    def __init__(self, myWindow, defaction):
        self.w_width = myWindow.width
        self.w_height = myWindow.height
        self.win = myWindow
        self.keypressed = 1
        self.default_action = defaction
        self.viewWidth = 0
        self.viewHeight = 0
        self.viewLeft = 0
        self.viewBottom = 0
        self.scaleX = 0
        self.scaleY = 0
        self.offsetX = 0
        self.offsetY = 0
        self.z_buffer = np.full((self.w_width, self.w_height), np.inf)
        self.buffers = {}
        self.vertices = []

    def go(self):
        if self.keypressed == 1:
            # default scene
            self.default_action()

        if self.keypressed == 2:
            # clear the framebuffer
            self.win.clearFB(0, 0, 0)
        #self.downsample()
        # push the window's framebuffer to the window
        self.win.applyFB()


    # def downsample(self):
    #     for x in range(self.w_width):
    #         for y in range(self.w_height):
    #             r, g, b = 0, 0, 0
    #             for i in range(self.ssaa_factor):
    #                 for j in range(self.ssaa_factor):
    #                     r += self.framebuffer[x * self.ssaa_factor + i, y * self.ssaa_factor + j, 0]
    #                     g += self.framebuffer[x * self.ssaa_factor + i, y * self.ssaa_factor + j, 1]
    #                     b += self.framebuffer[x * self.ssaa_factor + i, y * self.ssaa_factor + j, 2]
    #             r /= (self.ssaa_factor * self.ssaa_factor)
    #             g /= (self.ssaa_factor * self.ssaa_factor)
    #             b /= (self.ssaa_factor * self.ssaa_factor)
    #             self.win.set_pixel(x, y, r, g, b)
    #
    # def set_ss_pixel(self, x, y,r,g,b):
    #     if 0 <= x < self.high_res_width and 0 <= y < self.high_res_height:
    #         self.framebuffer[x, y, 0] = r
    #         self.framebuffer[x, y, 1] = g
    #         self.framebuffer[x, y, 2] = b


    def addBuffer(self, n, data, n_per_vertex):
        """
        Add a buffer to the engine.
        :param n:  name of the buffer
        :param data:  data to be added
        :param n_per_vertex:  number of elements per vertex
        :return:  None
        """
        self.buffers[n] = []
        buffer = []
        for i in range(0, len(data), n_per_vertex):
            buffer.append(data[i:i + n_per_vertex])
        self.buffers[n] = buffer

    def getBuffer(self, n):
        """
        Get the buffer with the given name.
        :param n:  name of the buffer
        :return:  value of the buffer
        """
        return self.buffers[n]

    def edge_function(self, a: glm.vec2, b: glm.vec2, p: glm.vec2):
        """
        Calculate the edge function of the triangle abc.
        :param a:  vertex a
        :param b:  vertex b
        :param p:  point p
        :return:  edge function
        """
        return (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x)

    def barycentric_coordinates(self, a: glm.vec2, b: glm.vec2, c: glm.vec2, p: glm.vec2):
        """
        Calculate the barycentric coordinates of the point p with respect to the triangle abc.
        :param a:  vertex a
        :param b:  vertex b
        :param c:  vertex c
        :param p:  point p
        :return:  barycentric coordinates
        """
        # calculate the barycentric coordinates
        det = self.edge_function(a, b, c)
        if abs(det) < epsilon:
            return -1, -1, -1
        lambda0 = self.edge_function(b, c, p) / det
        lambda1 = self.edge_function(c, a, p) / det
        #lambda2 = self.edge_function(a, b, p) / det
        lambda2 = 1 - lambda0 - lambda1
        return lambda0, lambda1, lambda2

    def defineViewWindow(self, t, b, r, l):
        """
        Define the view window.
        :param t:  top
        :param b:  bottom
        :param r:  right
        :param l:  left
        :return:  None
        """
        self.viewWidth = r - l
        self.viewHeight = t - b
        self.viewLeft = l
        self.viewBottom = b
        # Calculate scale and offset to map [-1, 1] normalized coordinates to screen coordinates
        self.scaleX = self.viewWidth / 2
        self.scaleY = self.viewHeight / 2
        self.offsetX = self.viewLeft + self.viewWidth / 2
        self.offsetY = self.viewBottom + self.viewHeight / 2

    def identity3D(self):
        """
        Create an identity matrix.
        :return:  identity matrix
        """
        return glm.mat4(1)

    def translate3D(self, x, y, z):
        """
        Create a translation matrix.
        :param x:  x-coordinate
        :param y:  y-coordinate
        :param z:  z-coordinate
        :return:  translation matrix
        """
        return glm.translate(glm.mat4(1), glm.vec3(x, y, z))

    def scale3D(self, x, y, z):
        """
        Create a scale matrix.
        :param x:  x
        :param y:  y
        :param z:  z
        :return:  scale matrix
        """
        return glm.scale(glm.mat4(1), glm.vec3(x, y, z))

    def rotateX(self, angle):
        """
        Create a rotation matrix about the x-axis.
        :param angle:  angle in degrees
        :return:  rotation matrix
        """
        return glm.rotate(glm.mat4(1), glm.radians(angle), glm.vec3(1, 0, 0))

    def rotateY(self, angle):
        """
        Create a rotation matrix about the y-axis.
        :param angle:  angle in degrees
        :return:  rotation matrix
        """
        return glm.rotate(glm.mat4(1), glm.radians(angle), glm.vec3(0, 1, 0))

    def rotateZ(self, angle):
        """
        Create a rotation matrix about the z-axis.
        :param angle:  angle in degrees
        :return:  rotation matrix
        """
        return glm.rotate(glm.mat4(1), glm.radians(angle), glm.vec3(0, 0, 1))

    def ortho3D(self, l, r, b, t, n, f):
        """
        Create an orthographic projection matrix.
        :param r:  right
        :param l:  left
        :param t:  top
        :param b:  bottom
        :param f:  far
        :param n:  near
        :return:  orthographic projection matrix
        """
        return glm.orthoRH(l, r, b, t, n, f)

    def frustum3D(self, l, r, b, t, n, f):
        """
        Create a frustum projection matrix.
        :param r:  right
        :param l:  left
        :param t:  top
        :param b:  bottom
        :param f:  far
        :param n:  near
        :return:  frustum projection matrix
        """
        return glm.frustum(l, r, b, t, n, f)

    def perspective3D(self, fovy, width, height, near, far):
        """
        Create a perspective projection matrix.
        :param fovy:  field of view
        :param width:  width
        :param height:  height
        :param near:  near
        :param far:  far
        :return:  perspective projection matrix
        """
        return glm.perspectiveFov(fovy, width, height, near, far)


    def lookAt(self, eye, lookat, up) -> glm.mat4x4:
        eye = glm.vec3(eye[0], eye[1], eye[2])
        return glm.lookAtRH(eye, lookat, up)

    def drawTriangles(self, vertex_pos_buffer, vertex_index_buffer, vertex_normals_buffer, vertex_uv_buffer,
                      vertex_shader, fragment_shader, uniforms):
        """
        Draw triangles using the given buffers and shaders.
        :param vertex_pos_buffer:  vertex position buffer
        :param vertex_index_buffer:  vertex index buffer
        :param vertex_normals_buffer:  vertex normals buffer
        :param vertex_uv_buffer:  vertex uv buffer
        :param vertex_shader:  vertex shader
        :param fragment_shader:  fragment shader
        :param uniforms:  uniforms
        :return:  None
        """
        # Get the buffers
        vertex_pos = self.getBuffer(vertex_pos_buffer)
        indices = self.getBuffer(vertex_index_buffer)
        normals = self.getBuffer(vertex_normals_buffer)
        uvs = self.getBuffer(vertex_uv_buffer)

        for i in range(0, len(indices), 3): # Iterate over each triangle
            # create vertices
            v0 = Vertex(indices[i][0])
            v1 = Vertex(indices[i + 1][0])
            v2 = Vertex(indices[i + 2][0])

            # attach varying
            v0.attach_varying('position', vertex_pos[v0.get_id()])
            v1.attach_varying('position', vertex_pos[v1.get_id()])
            v2.attach_varying('position', vertex_pos[v2.get_id()])

            v0.attach_varying('normal', normals[v0.get_id()])
            v1.attach_varying('normal', normals[v1.get_id()])
            v2.attach_varying('normal', normals[v2.get_id()])

            v0.attach_varying('uvs', uvs[v0.get_id()])
            v1.attach_varying('uvs', uvs[v1.get_id()])
            v2.attach_varying('uvs', uvs[v2.get_id()])

            # vertex shader
            v0 = vertex_shader.vertex_shader(v0, uniforms)
            v1 = vertex_shader.vertex_shader(v1, uniforms)
            v2 = vertex_shader.vertex_shader(v2, uniforms)

            # rasterize
            p0 = v0.get_varying('position')
            v0.attach_varying('pos', p0) # world space position
            p0 = glm.vec3(p0.x, p0.y, p0.z)
            p0.x = int(p0.x * self.scaleX + self.offsetX)
            p0.y = int(p0.y * self.scaleY + self.offsetY)
            v0.attach_varying('position', p0) # screen space position

            p1 = v1.get_varying('position')
            v1.attach_varying('pos', p1)
            p1 = glm.vec3(p1.x, p1.y, p1.z)
            p1.x = int(p1.x * self.scaleX + self.offsetX)
            p1.y = int(p1.y * self.scaleY + self.offsetY)
            v1.attach_varying('position', p1)

            p2 = v2.get_varying('position')
            v2.attach_varying('pos', p2)
            p2 = glm.vec3(p2.x, p2.y, p2.z)
            p2.x = int(p2.x * self.scaleX + self.offsetX)
            p2.y = int(p2.y * self.scaleY + self.offsetY)
            v2.attach_varying('position', p2)

            self.rasterizeTriangle(v0, v1, v2, fragment_shader, uniforms)

    def rasterizeTriangle(self, p0, p1, p2, fragment_shader, uniforms):
        """
        Rasterize a triangle.
        :param p0:  vertex 0
        :param p1:  vertex 1
        :param p2:  vertex 2
        :param fragment_shader: fragment shader
        :param uniforms:  uniforms
        :return:
        """
        # Get the bounding box
        minX = int(min(p0.get_varying('position').x, p1.get_varying('position').x, p2.get_varying('position').x))
        minY = int(min(p0.get_varying('position').y, p1.get_varying('position').y, p2.get_varying('position').y))
        maxX = int(max(p0.get_varying('position').x, p1.get_varying('position').x, p2.get_varying('position').x))
        maxY = int(max(p0.get_varying('position').y, p1.get_varying('position').y, p2.get_varying('position').y))

        # Iterate over each pixel within the bounding box
        for x in range(minX, maxX):
            for y in range(minY, maxY):
                p = glm.vec2(x, y)
                a = glm.vec2(p0.get_varying('position').x, p0.get_varying('position').y)
                b = glm.vec2(p1.get_varying('position').x, p1.get_varying('position').y)
                c = glm.vec2(p2.get_varying('position').x, p2.get_varying('position').y)

                l0, l1, l2 = self.barycentric_coordinates(a, b, c, p)
                if l0 >= -epsilon and l1 >= -epsilon and l2 >= -epsilon: # Check if the point is inside the triangle
                    z = p0.get_varying('position').z * l0 + p1.get_varying('position').z * l1 + p2.get_varying(
                        'position').z * l2
                    if 0 <= x < self.w_width and 0 <= y < self.w_height:
                        if z < self.z_buffer[x, y]:
                            self.z_buffer[x, y] = z
                            color = fragment_shader.fragment_shader(p0, p1, p2, l0, l1, l2, uniforms)
                            self.win.set_pixel(x, y, color[0], color[1], color[2])
