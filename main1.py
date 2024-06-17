from rit_window import *
from cgI_engine import *
from vertex import *
from clipper import *
from shapes import *
import numpy as np
from vertex_shader import *
from fragment_shader import *

class Sphere:
    def __init__(self, center, radius, color):
        self.center = center
        self.radius = radius
        self.color = color

    def intersect(self, ray):
        L = self.center - ray.origin
        tca = glm.dot(L, ray.direction)
        d2 = glm.dot(L, L) - tca * tca
        if d2 > self.radius * self.radius:
            return None
        thc = glm.sqrt(self.radius * self.radius - d2)
        t0 = tca - thc
        t1 = tca + thc

        if t0 < 0 and t1 < 0:
            return None
        t = t0 if t0 > 0 else t1
        hit_point = ray.origin + t * ray.direction
        normal = glm.normalize(hit_point - self.center)

        return {'t': t, 'hit_point': hit_point, 'normal': normal, 'color': self.color}

def default_action():
    myEngine.win.clearFB(.25, .25, .75)
    myEngine.defineViewWindow(800, 0, 800, 0)

    # Example scene setup
    myEngine.scene = {
        'objects': [
            Sphere(glm.vec3(0, 0, -5), 1, glm.vec3(1, 0, 0)),
            Sphere(glm.vec3(2, 0, -5), 1, glm.vec3(0, 1, 0))
        ],
        'light_pos': glm.vec3(0, 10, 0),
        'light_color': glm.vec3(1, 1, 1),
        'camera_pos': glm.vec3(0, 0, 0),
        'camera_target': glm.vec3(0, 0, -1),
        'camera_up': glm.vec3(0, 1, 0),
        'ambient_color': glm.vec3(1, 1, 1)
    }
window = RitWindow(800, 800)
myEngine = CGIengine(window, default_action)


def main():
    window.run(myEngine)


if __name__ == "__main__":
    main()
