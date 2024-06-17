import glm
import numpy as np
import matplotlib.pyplot as plt

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = glm.normalize(direction)

def generate_ray(x, y, width, height, fov, camera_pos, camera_target, camera_up):
    aspect_ratio = width / height
    scale = glm.tan(glm.radians(fov * 0.5))

    image_x = (2 * (x + 0.5) / width - 1) * aspect_ratio * scale
    image_y = (1 - 2 * (y + 0.5) / height) * scale

    camera_forward = glm.normalize(camera_target - camera_pos)
    camera_right = glm.normalize(glm.cross(camera_forward, camera_up))
    camera_up = glm.cross(camera_right, camera_forward)

    pixel_position = camera_pos + camera_forward + image_x * camera_right + image_y * camera_up
    direction = pixel_position - camera_pos

    return Ray(camera_pos, direction)

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

def phong_shading(hit_info, light_pos, light_color, camera_pos, ambient_color, ka, kd, ks, shininess):
    hit_point = hit_info['hit_point']
    normal = hit_info['normal']
    color = hit_info['color']

    to_light = glm.normalize(light_pos - hit_point)
    to_camera = glm.normalize(camera_pos - hit_point)

    ambient = ka * ambient_color * color
    diffuse = kd * max(glm.dot(normal, to_light), 0) * light_color * color
    reflection = glm.reflect(-to_light, normal)
    specular = ks * pow(max(glm.dot(reflection, to_camera), 0), shininess) * light_color

    return glm.clamp(ambient + diffuse + specular, 0, 1)

def render(scene, width, height, fov, camera_pos, camera_target, camera_up):
    framebuffer = np.zeros((height, width, 3))
    for y in range(height):
        for x in range(width):
            ray = generate_ray(x, y, width, height, fov, camera_pos, camera_target, camera_up)
            color = trace_ray(ray, scene)
            framebuffer[y, x] = color
    return framebuffer

def trace_ray(ray, scene):
    closest_hit = None
    for obj in scene['objects']:
        hit_info = obj.intersect(ray)
        if hit_info and (closest_hit is None or hit_info['t'] < closest_hit['t']):
            closest_hit = hit_info

    if closest_hit:
        return phong_shading(closest_hit, scene['light_pos'], scene['light_color'], scene['camera_pos'], scene['ambient_color'], 0.1, 0.7, 0.2, 10)
    else:
        return np.array([0, 0, 0])  # Background color

# Example usage
scene = {
    'objects': [
        Sphere(glm.vec3(-2, 0, -5), 4, glm.vec3(1, 0, 0)),
        Sphere(glm.vec3(4, 0, -5), 1, glm.vec3(0, 1, 0))
    ],
    'light_pos': glm.vec3(-5, 0, 0),
    'light_color': glm.vec3(1, 1, 1),
    'camera_pos': glm.vec3(0, 0, 0),
    'camera_target': glm.vec3(0, 0, -1),
    'camera_up': glm.vec3(0, 1, 0),
    'ambient_color': glm.vec3(1, 1, 1)
}

width, height, fov = 800, 800, 90
framebuffer = render(scene, width, height, fov, glm.vec3(0, 0, 0), glm.vec3(0, 0, -1), glm.vec3(0, 1, 0))

plt.imshow(framebuffer)
plt.show()
