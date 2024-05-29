from rit_window import *
from cgI_engine import *
from vertex import *
from clipper import *
from shapes import *
from vertex_shader import *
from fragment_shader import *
from stack import *
from  PIL import Image

def default_action():
    myEngine.win.clearFB(.15, .15, .45)
    myEngine.defineViewWindow(800, 0, 800, 0)

    viewT = myEngine.lookAt([0, 2.0, 0], [-15, -1, 12], [0, 1, 0])
    projectionT = myEngine.ortho3D(-8.0, 8.0, -6.0, 6.0, 0, 10.0)

    # draw chess board
    myEngine.addBuffer('cube', cube, 3)
    myEngine.addBuffer('cube_idx', cube_idx, 1)
    myEngine.addBuffer('cube_normals', cube_normals, 3)
    myEngine.addBuffer('cube_uv', cube_uv, 2)

    v_shader = vertex_shader_checkerboard()
    f_shader = fragment_shader_checkerboard()
    uniforms = {}
    uniforms['viewT'] = viewT
    uniforms['projectionT'] = projectionT
    uniforms['color1'] = [0.0, 0.0, 0.0]
    uniforms['color2'] = [1.0, 1.0, 1.0]
    uniforms['checksize'] = 0.25
    modelT =myEngine.scale3D(8.0, 1.0, 8.0)
    uniforms['modelT'] = modelT
    myEngine.drawTriangles('cube', 'cube_idx', 'cube_normals', 'cube_uv', v_shader, f_shader, uniforms)

    #pawn
    im = Image.open("texture_wood.jpg")
    tex_width, tex_height = im.size
    uniforms['texture'] = im
    uniforms['tex_width'] = tex_width
    uniforms['tex_height'] = tex_height
    v_shader = vertex_shader_texture()
    f_shader = fragment_shader_texture()
    modelT = myEngine.translate3D(-2.7, 1, 3) * myEngine.scale3D(1.0, 1.0, 1.0)
    uniforms['modelT'] = modelT
    myEngine.drawTriangles('cube', 'cube_idx', 'cube_normals', 'cube_uv', v_shader, f_shader, uniforms)

    # Phong settings
    uniforms['ocolor'] = [1.0, 0.0, 0.0]
    uniforms['scolor'] = [1.0, 1.0, 1.0]
    uniforms['k'] = [0.2, 0.4, 0.4]
    uniforms['exponent'] = 10.0
    uniforms['lightpos'] = [4,3,-20]
    uniforms['lightcolor'] = [1.0, 1.0, 1.0]
    uniforms['amb_color'] = [1.0, 1.0, 1.0]

    # king ( small cube on top of big cube) with phong shading
    stack = Stack()
    coordT = myEngine.translate3D(-2.5, 1, -1.5)
    stack.push(coordT)
    partT = myEngine.scale3D(1, 1, 1)
    stack.push(partT)
    uniforms['modelT'] = stack.pop()
    v_shader = vertex_shader_texture()
    f_shader = fragment_shader_texture()
    myEngine.drawTriangles('cube', 'cube_idx', 'cube_normals', 'cube_uv', v_shader, f_shader, uniforms)
    coordT = myEngine.translate3D(0, .6,0)
    stack.push(coordT)
    partT = myEngine.rotateY(40) * myEngine.scale3D(.6,.6,.6)
    stack.push(partT)
    uniforms['modelT'] = stack.pop()
    v_shader = vertex_shader_Phong()
    f_shader = fragment_shader_Phong()
    myEngine.drawTriangles('cube', 'cube_idx', 'cube_normals', 'cube_uv', v_shader, f_shader, uniforms)

    # rook (half of a small sphere on top of big cube) with phong shading
    uniforms['ocolor'] = [0.0, 1.0, 0.0]
    stack = Stack()
    coordT = myEngine.translate3D(2.8, 1, 3)
    partT = myEngine.scale3D(1, 1, 1)
    stack.push(coordT)
    stack.push(partT)
    uniforms['modelT'] = stack.pop()
    myEngine.addBuffer('sphere', sphere, 3)
    myEngine.addBuffer('sphere_idx', sphere_idx, 1)
    myEngine.addBuffer('sphere_normals', sphere_normals, 3)
    myEngine.addBuffer('sphere_uv', sphere_uv, 2)
    v_shader = vertex_shader_texture()
    f_shader = fragment_shader_texture()
    myEngine.drawTriangles('cube', 'cube_idx', 'cube_normals', 'cube_uv', v_shader, f_shader, uniforms)
    coordT = myEngine.translate3D(0, 0.5, 0)
    stack.push(coordT)
    partT = myEngine.scale3D(.8, .8, .8)
    stack.push(partT)
    uniforms['modelT'] = stack.pop()
    v_shader = vertex_shader_Phong()
    f_shader = fragment_shader_Phong()
    myEngine.drawTriangles('sphere', 'sphere_idx', 'sphere_normals', 'sphere_uv', v_shader, f_shader, uniforms)

    # rook (half of a small sphere on top of big cube) with gouraud shading
    uniforms['ocolor'] = [0.0, 1.0, 0.0]
    stack = Stack()
    coordT = myEngine.translate3D(2.8, 1, -2.7)
    partT = myEngine.scale3D(1, 1, 1)
    stack.push(coordT)
    stack.push(partT)
    uniforms['modelT'] = stack.pop()
    myEngine.addBuffer('sphere', sphere, 3)
    myEngine.addBuffer('sphere_idx', sphere_idx, 1)
    myEngine.addBuffer('sphere_normals', sphere_normals, 3)
    myEngine.addBuffer('sphere_uv', sphere_uv, 2)
    v_shader = vertex_shader_texture()
    f_shader = fragment_shader_texture()
    myEngine.drawTriangles('cube', 'cube_idx', 'cube_normals', 'cube_uv', v_shader, f_shader, uniforms)
    coordT = myEngine.translate3D(0, 0.5, 0)
    stack.push(coordT)
    partT = myEngine.scale3D(.8, .8, .8)
    stack.push(partT)
    uniforms['modelT'] = stack.pop()
    v_shader = vertex_shader_Gouraud()
    f_shader = fragment_shader_Gouraud()
    myEngine.drawTriangles('sphere', 'sphere_idx', 'sphere_normals', 'sphere_uv', v_shader, f_shader, uniforms)

    # queen ( small cube on top of big cylinder) with phong shading
    uniforms['ocolor'] = [0.0, 0.0, 1.0]
    stack = Stack()
    coordT = myEngine.translate3D(-2.5, 1, 1)
    partT = myEngine.scale3D(1, 1, 1)
    stack.push(coordT)
    stack.push(partT)
    uniforms['modelT'] = stack.pop()
    myEngine.addBuffer('cylinder', cylinder, 3)
    myEngine.addBuffer('cylinder_idx', cylinder_idx, 1)
    myEngine.addBuffer('cylinder_normals', cylinder_normals, 3)
    myEngine.addBuffer('cylinder_uv', cylinder_uv, 2)
    v_shader = vertex_shader_texture()
    f_shader = fragment_shader_texture()
    myEngine.drawTriangles('cylinder', 'cylinder_idx', 'cylinder_normals', 'cylinder_uv', v_shader, f_shader, uniforms)
    coordT = myEngine.translate3D(0, .6, 0)
    partT = myEngine.scale3D(0.4, 0.4, 0.4)
    stack.push(coordT)
    stack.push(partT)
    uniforms['modelT'] = stack.pop()
    v_shader = vertex_shader_Phong()
    f_shader = fragment_shader_Phong()
    myEngine.drawTriangles('cube', 'cube_idx', 'cube_normals', 'cube_uv', v_shader, f_shader, uniforms)


window = RitWindow(800, 800)
myEngine = CGIengine(window, default_action)


def main():
    window.run(myEngine)


if __name__ == "__main__":
    main()
