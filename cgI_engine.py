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
        self.win.applyFB()

    def addBuffer(self, name, data, elements_per_vertex):
        """
        Add a buffer to the engine.

        :param name: Name of the buffer
        :param data: Data to be added (list or array)
        :param elements_per_vertex: Number of elements per vertex
        :raises ValueError: If the data length is not a multiple of elements_per_vertex
        """
        if not isinstance(name, str):
            raise TypeError("Buffer name must be a string")
        if not isinstance(data, (list, np.ndarray)):
            raise TypeError("Data must be a list or numpy array")
        if not isinstance(elements_per_vertex, int) or elements_per_vertex <= 0:
            raise ValueError("elements_per_vertex must be a positive integer")

        if len(data) % elements_per_vertex != 0:
            raise ValueError("Data length must be a multiple of elements_per_vertex")

        buffer = [data[i:i + elements_per_vertex] for i in range(0, len(data), elements_per_vertex)]
        self.buffers[name] = buffer

    def getBuffer(self, name):
        """
        Get the buffer with the given name.

        :param name: Name of the buffer
        :raises KeyError: If the buffer with the given name does not exist
        :return: Value of the buffer
        """
        if name not in self.buffers:
            raise KeyError(f"Buffer '{name}' does not exist")

        return self.buffers[name]

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

        if not callable(vertex_shader.vertex_shader):
            raise TypeError("vertex_shader must have a callable method 'vertex_shader'")
        if not callable(fragment_shader.fragment_shader):
            raise TypeError("fragment_shader must have a callable method 'fragment_shader'")

        required_uniforms = ['modelT', 'viewT', 'projectionT']
        for uniform in required_uniforms:
            if uniform not in uniforms:
                raise ValueError(f"Uniform '{uniform}' is required but not provided")

        for i in range(0, len(indices), 3): # Iterate over each triangle
            try:
                # create vertices
                v0 = Vertex(indices[i][0])
                v1 = Vertex(indices[i + 1][0])
                v2 = Vertex(indices[i + 2][0])

                # Attach varying attributes
                for v in [v0, v1, v2]:
                    v.attach_varying('position', vertex_pos[v.get_id()])
                    v.attach_varying('normal', normals[v.get_id()])
                    v.attach_varying('uvs', uvs[v.get_id()])

                # vertex shader
                v0 = vertex_shader.vertex_shader(v0, uniforms)
                v1 = vertex_shader.vertex_shader(v1, uniforms)
                v2 = vertex_shader.vertex_shader(v2, uniforms)

                # rasterize
                for v in [v0, v1, v2]:
                    p = v.get_varying('position')
                    v.attach_varying('pos', p)
                    p = glm.vec3(p.x, p.y, p.z)
                    p.x = int(p.x * self.scaleX + self.offsetX)
                    p.y = int(p.y * self.scaleY + self.offsetY)
                    v.attach_varying('position', p)

                self.rasterizeTriangle(v0, v1, v2, fragment_shader, uniforms)
            except Exception as e:
               print(f"Error processing triangle {i // 3}: {e}")

    def rasterizeTriangle(self, p0, p1, p2, fragment_shader, uniforms):
        """
        Rasterize a triangle.
        :param p0: Vertex 0
        :param p1: Vertex 1
        :param p2: Vertex 2
        :param fragment_shader: Fragment shader object
        :param uniforms: Uniforms dictionary
        :raises TypeError: If fragment_shader is not callable
        :return: None
        """
        if not callable(fragment_shader.fragment_shader):
            raise TypeError("fragment_shader must have a callable method 'fragment_shader'")

        try:
            # Get the bounding box
            minX = int(min(p0.get_varying('position').x, p1.get_varying('position').x, p2.get_varying('position').x))
            minY = int(min(p0.get_varying('position').y, p1.get_varying('position').y, p2.get_varying('position').y))
            maxX = int(max(p0.get_varying('position').x, p1.get_varying('position').x, p2.get_varying('position').x))
            maxY = int(max(p0.get_varying('position').y, p1.get_varying('position').y, p2.get_varying('position').y))

            samples = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]

            # Iterate over each pixel within the bounding box
            for x in range(minX, maxX):
                for y in range(minY, maxY):
                    color_samples = []
                    for sample in samples:
                        px, py = x + sample[0], y + sample[1]
                        p = glm.vec2(px, py)
                        a = glm.vec2(p0.get_varying('position').x, p0.get_varying('position').y)
                        b = glm.vec2(p1.get_varying('position').x, p1.get_varying('position').y)
                        c = glm.vec2(p2.get_varying('position').x, p2.get_varying('position').y)

                        l0, l1, l2 = self.barycentric_coordinates(a, b, c, p)
                        if l0 >= -epsilon and l1 >= -epsilon and l2 >= -epsilon:  # Check if the point is inside the triangle
                            z = p0.get_varying('position').z * l0 + p1.get_varying('position').z * l1 + p2.get_varying(
                                'position').z * l2
                            if 0 <= x < self.w_width and 0 <= y < self.w_height:
                                if z < self.z_buffer[x, y]:
                                    self.z_buffer[x, y] = z
                                    color = fragment_shader.fragment_shader(p0, p1, p2, l0, l1, l2, uniforms)
                                    color_samples.append(color)

                    if color_samples:
                        final_color = np.mean(color_samples, axis=0)
                        self.win.set_pixel(x, y, final_color[0], final_color[1], final_color[2])
        except Exception as e:
            print(f"Error rasterizing triangle: {e}")
