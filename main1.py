from rit_window import *
from cgI_engine import *
from objects import *
def default_action():
    myEngine.win.clearFB(.25, .25, .75)
    myEngine.defineViewWindow(800, 0, 800, 0)

    # Define Cornell Box scene
    myEngine.scene = {
        'objects': [
            # Walls
            Cube(glm.vec3(-3, -3, -10), glm.vec3(3, -2.9, -4), glm.vec3(0.75, 0.75, 0.75)),  # Bottom wall
            Cube(glm.vec3(-3, 2.9, -10), glm.vec3(3, 3, -4), glm.vec3(0.75, 0.75, 0.75)),   # Top wall
            Cube(glm.vec3(-3, -3, -10), glm.vec3(-2.9, 3, -4), glm.vec3(0.75, 0.25, 0.25)),  # Left wall
            Cube(glm.vec3(2.9, -3, -10), glm.vec3(3, 3, -4), glm.vec3(0.25, 0.25, 0.75)),   # Right wall
            Cube(glm.vec3(-3, -3, -10), glm.vec3(3, 3, -9.9), glm.vec3(0.75, 0.75, 0.75)),  # Back wall

            # Light source
            Cube(glm.vec3(-0.5, 2.8, -7), glm.vec3(0.5, 3, -6), glm.vec3(1, 1, 1)),  # Ceiling light

            # Objects
            Sphere(glm.vec3(-1, -2, -7), 1, glm.vec3(0.75, 0.75, 0.75)),  # Sphere
            Cube(glm.vec3(0.5, -3, -8), glm.vec3(2, -1, -6), glm.vec3(0.5, 0.5, 0.5))  # Cube
        ],
        'light_pos': glm.vec3(0, 2.8, -6.5),
        'light_color': glm.vec3(1, 1, 1),
        'camera_pos': glm.vec3(0, 0, 0),
        'camera_target': glm.vec3(0, 0, -1),
        'camera_up': glm.vec3(0, 1, 0),
        'ambient_color': glm.vec3(0.2, 0.2, 0.2)
    }

window = RitWindow(800, 800)
myEngine = CGIengine(window, default_action)

def main():
    window.run(myEngine)

if __name__ == "__main__":
    main()


def main():
    window.run(myEngine)


if __name__ == "__main__":
    main()
