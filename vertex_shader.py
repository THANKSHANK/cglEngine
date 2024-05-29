"""
This file contains the vertex shader classes for the different shading models.
"""
import glm
class vertex_shader_solid:
    def __init__(self):
        pass

    def vertex_shader(self, V, uniforms):
        modelT = uniforms['modelT']
        viewT = uniforms['viewT']
        projectionT = uniforms['projectionT']
        pos = projectionT * viewT * modelT * glm.vec4(V.get_varying('position'), 1.0)
        V.attach_varying('position', glm.vec3(pos) / pos.w)
        V.attach_varying('color', uniforms['color'])
        return V
class vertex_shader_solid_with_edge:
    def __init__(self):
        pass

    def vertex_shader(self, V, uniforms):
        modelT = uniforms['modelT']
        viewT = uniforms['viewT']
        projectionT = uniforms['projectionT']
        pos = projectionT * viewT * modelT * glm.vec4(V.get_varying('position'), 1.0)
        V.attach_varying('position', glm.vec3(pos) / pos.w)
        V.attach_varying('color', uniforms['color'])
        return V
class vertex_shader_Gouraud:
    def __init__(self):
        pass

    def vertex_shader(self, V, uniforms):
        modelT = uniforms['modelT']
        viewT = uniforms['viewT']
        projectionT = uniforms['projectionT']
        pos = projectionT * viewT * modelT * glm.vec4(V.get_varying('position'), 1.0)
        V.attach_varying('position', glm.vec3(pos) / pos.w)

        n = glm.vec4(V.get_varying('normal'), 0.0)
        tn = glm.transpose(glm.inverse(modelT) )* n
        tn = glm.normalize(glm.vec3(tn.x, tn.y, tn.z))
        V.attach_varying('normal', tn)

        ka, kd, ks = uniforms['k']
        ambient_color = glm.vec3(uniforms['amb_color'])
        object_color = glm.vec3(uniforms['ocolor'])
        specular_color = glm.vec3(uniforms['scolor'])
        light_color = glm.vec3(uniforms['lightcolor'])
        light_pos = glm.vec3(uniforms['lightpos'])

        ambient = ka * ambient_color * object_color

        L = glm.normalize(glm.vec3(light_pos) - V.get_varying('position'))
        dotLN = max(glm.dot(L, tn), 0.0)
        diffuse = kd * light_color * object_color * dotLN

        Vp = glm.normalize(V.get_varying('position')* -1.0)
        R = glm.reflect(-L, tn)
        dotRV = max(glm.dot(R, Vp), 0.0)
        specular = ks * pow(dotRV, uniforms['exponent']) * specular_color * light_color

        V.attach_varying('color', ambient + diffuse + specular)
        return V

class vertex_shader_Phong:
    def __init__(self):
        pass

    def vertex_shader(self, V, uniforms):
        modelT = uniforms['modelT']
        viewT = uniforms['viewT']
        projectionT = uniforms['projectionT']
        pos = projectionT * viewT * modelT * glm.vec4(V.get_varying('position'), 1.0)
        V.attach_varying('position', glm.vec3(pos) / pos.w)

        n = glm.vec4(V.get_varying('normal'), 0.0)
        tn = glm.transpose(glm.inverse(modelT)) * n
        tn = glm.normalize(glm.vec3(tn.x, tn.y, tn.z))
        V.attach_varying('normal', tn)

        return V
class vertex_shader_texture:
    def __init__(self):
        pass

    def vertex_shader(self, V, uniforms):
        modelT = uniforms['modelT']
        viewT = uniforms['viewT']
        projectionT = uniforms['projectionT']
        pos = projectionT * viewT * modelT * glm.vec4(V.get_varying('position'), 1.0)
        V.attach_varying('position', glm.vec3(pos) / pos.w)

        return V
class vertex_shader_checkerboard:
    def __init__(self):
        pass

    def vertex_shader(self, V, uniforms):
        modelT = uniforms['modelT']
        viewT = uniforms['viewT']
        projectionT = uniforms['projectionT']
        pos = projectionT * viewT * modelT * glm.vec4(V.get_varying('position'),1.0)
        V.attach_varying('position',glm.vec3(pos)/pos.w)

        return V
