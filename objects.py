import glm
import numpy as np

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
class Cube:
    def __init__(self, min_corner, max_corner, color):
        self.min_corner = min_corner
        self.max_corner = max_corner
        self.color = color

    def intersect(self, ray):
        epsilon = 1e-5  # Small value to avoid division by zero
        tmin = (self.min_corner.x - ray.origin.x) / (ray.direction.x + epsilon)
        tmax = (self.max_corner.x - ray.origin.x) / (ray.direction.x + epsilon)
        if tmin > tmax:
            tmin, tmax = tmax, tmin

        tymin = (self.min_corner.y - ray.origin.y) / (ray.direction.y + epsilon)
        tymax = (self.max_corner.y - ray.origin.y) / (ray.direction.y + epsilon)
        if tymin > tymax:
            tymin, tymax = tymax, tymin

        if (tmin > tymax) or (tymin > tmax):
            return None
        if tymin > tmin:
            tmin = tymin
        if tymax < tmax:
            tmax = tymax

        tzmin = (self.min_corner.z - ray.origin.z) / (ray.direction.z + epsilon)
        tzmax = (self.max_corner.z - ray.origin.z) / (ray.direction.z + epsilon)
        if tzmin > tzmax:
            tzmin, tzmax = tzmax, tzmin

        if (tmin > tzmax) or (tzmin > tmax):
            return None
        if tzmin > tmin:
            tmin = tzmin
        if tzmax < tmax:
            tmax = tzmax

        t = tmin if tmin > 0 else tmax
        if t < 0:
            return None

        hit_point = ray.origin + t * ray.direction
        normal = self.get_normal(hit_point)

        return {'t': t, 'hit_point': hit_point, 'normal': normal, 'color': self.color}

    def get_normal(self, hit_point):
        epsilon = 1e-5
        if abs(hit_point.x - self.min_corner.x) < epsilon:
            return glm.vec3(-1, 0, 0)
        if abs(hit_point.x - self.max_corner.x) < epsilon:
            return glm.vec3(1, 0, 0)
        if abs(hit_point.y - self.min_corner.y) < epsilon:
            return glm.vec3(0, -1, 0)
        if abs(hit_point.y - self.max_corner.y) < epsilon:
            return glm.vec3(0, 1, 0)
        if abs(hit_point.z - self.min_corner.z) < epsilon:
            return glm.vec3(0, 0, -1)
        if abs(hit_point.z - self.max_corner.z) < epsilon:
            return glm.vec3(0, 0, 1)
        return glm.vec3(0, 0, 0)
