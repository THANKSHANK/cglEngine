"""
this program is to test Gouraud shading and Phong shading of a sphere
"""
from rit_window import *
from cgI_engine import *
from vertex import *
from clipper import *
from shapes import *
from vertex_shader import *
from fragment_shader import *

def default_action():
    myEngine.win.clearFB(.25, .25, .75)
    myEngine.defineViewWindow(800, 0, 800, 0)

    myEngine.addBuffer('sphere', sphere, 3)
    myEngine.addBuffer('sphere_idx', sphere_idx, 1)
    myEngine.addBuffer('sphere_normals', sphere_normals, 3)
    myEngine.addBuffer('sphere_uv', sphere_uv, 2)

    myEngine.addBuffer('cone', cone, 3)
    myEngine.addBuffer('cone_idx', cone_idx, 1)
    myEngine.addBuffer('cone_normals', cone_normals, 3)
    myEngine.addBuffer('cone_uv', cone_uv, 2)


    viewT = myEngine.lookAt([0.0, 0.0, 0.0], [0, 0, -20], [0, 1, 0])
    projectionT = myEngine.ortho3D(-3.0, 3.0, -3.0, 3.0, 0, 30.0)
    #projectionT = myEngine.perspective3D(90.0,6,6,0.1,30.0)
    modelT = myEngine.translate3D(-2.0, 0.0, -5.0) * myEngine.scale3D(2.0, 2.0, 2.0)

    uniforms = {}
    uniforms['viewT'] = viewT
    uniforms['projectionT'] = projectionT
    uniforms['modelT'] = modelT
    uniforms['ocolor'] = [1.0, 0.0, 0.0]
    uniforms['scolor'] = [1.0, 1.0, 1.0]
    uniforms['k'] = [0.2, 0.4, 0.4]
    uniforms['exponent'] = 10.0
    uniforms['lightpos'] = [-2.0, 3.0, 2.0]
    uniforms['lightcolor'] = [1.0, 1.0, 1.0]
    uniforms['amb_color'] = [1.0, 1.0, 1.0]

    v_shader = vertex_shader_Gouraud()
    f_shader = fragment_shader_Gouraud()

    myEngine.drawTriangles('sphere', 'sphere_idx', 'sphere_normals', 'sphere_uv', v_shader, f_shader, uniforms)

    modelT = myEngine.translate3D(2.0, 0.0, -5.0) * myEngine.scale3D(2.0, 2.0, 2.0)
    uniforms['modelT'] = modelT
    uniforms['ocolor'] = [0.0, 1.0, 0.0]
    v_shader = vertex_shader_Phong()
    f_shader = fragment_shader_Phong()

    myEngine.drawTriangles('sphere', 'sphere_idx', 'sphere_normals', 'sphere_uv', v_shader, f_shader, uniforms)

window = RitWindow(800, 800)
myEngine = CGIengine(window, default_action)


def main():
    window.run(myEngine)


if __name__ == "__main__":
    main()
