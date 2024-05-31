import numpy as np
import signal
from cgI_engine import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

class RitWindow:
    def __init__(self, width, height, title="Computer Graphics Engine"):
        # window-related info
        self.width = width
        self.height = height
        self.title = title
        self.framebuffer = np.array([0 for i in range(self.width*self.height*4)])
        
    def run (self, e):
        self.myEngine = e
        
        glutInit()
        glutInitDisplayMode(GLUT_RGBA)
        glutInitWindowSize(self.width, self.height)
        glutInitWindowPosition(0, 0)
        self.winid = glutCreateWindow(self.title)
        glutDisplayFunc(e.go)
        glutKeyboardUpFunc (self._keypress)

        self.t_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.t_id)
        glPixelStorei(GL_UNPACK_ALIGNMENT,1)
        glTexImage2D(
            GL_TEXTURE_2D, 0, 3, self.width, self.height, 0,
            GL_RGBA, GL_UNSIGNED_INT, self.framebuffer
        )
        glutMainLoop()
        
    def clearFB (self, r, g, b):
        for x in range(self.width):
            for y in range(self.height):
                self.set_pixel(x, y, r, g, b)
        
    def set_pixel(self, x, y, r, g, b):
        if (x <0 or y <0 or x >= self.width or y >= self.height):
            print ("set_pixel error:  pixel [", x, ",", y, "] is out of range")
        else:
            # sets the pixel at (x,y) to (r, g, b, 255), with 8-bit integer colors
            rr = int (r *255) if not math.isnan(r) else 0
            gg = int (g * 255) if not math.isnan(g) else 0
            bb = int (b * 255) if not math.isnan(b) else 0

            yy = self.height - y - 1
            self.framebuffer[x*4 + 0 + yy*self.width*4] = rr      # r
            self.framebuffer[x*4 + 1 + yy*self.width*4] = gg      # g
            self.framebuffer[x*4 + 2 + yy*self.width*4] = bb      # b
            self.framebuffer[x*4 + 3 + yy*self.width*4] = 255    # a
        
        
    def applyFB(self):
        glEnable(GL_TEXTURE_2D)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
        glTexImage2D(
            GL_TEXTURE_2D, 0, 3, self.width, self.height, 0,
            GL_RGBA, GL_UNSIGNED_BYTE, self.framebuffer
        )
        glBindTexture(GL_TEXTURE_2D, self.t_id)
        self._render()
        
        
    #
    # some private stuff you should be not concerned with
    #
    def _fullscreen_quad(self):
        # the quad!
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0)
        glVertex2f(0, 0)
        glTexCoord2f(1.0, 0.0)
        glVertex2f(self.width, 0)
        glTexCoord2f(1.0, 1.0)
        glVertex2f(self.width, self.height)
        glTexCoord2f(0.0, 1.0)
        glVertex2f(0, self.height)
        glEnd()
        
    def _render(self):
        # internal function that actually draws the scene
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glDisable(GL_LIGHTING) 
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0.0, self.width, self.height, 0.0, 0.0, 1.0)
        glMatrixMode (GL_MODELVIEW)
        glLoadIdentity()
        glColor3f(1.0, 1.0, 0.0)
        glEnable(GL_TEXTURE_2D)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
        glBindTexture(GL_TEXTURE_2D, self.t_id)
        self._fullscreen_quad()
        glutSwapBuffers()

    def _keypress (self, key, x, y):
        keychar = key.decode('UTF-8')
        self.myEngine.keyboard (keychar)
        if (keychar == 'q') :
            os.kill(os.getpid(), signal.SIGTERM)

    def drawRect (self, top, bottom, right, left, r, g, b):
        #top and bottom edges
        for i in range (left, right):
            self.set_pixel (i, top, r, g, b)
            self.set_pixel (i, bottom, r, g, b)
          
        # left and right
        for j in range (bottom, top):
            self.set_pixel (left, j, r, g, b)
            self.set_pixel (right, j, r, g, b)
        
