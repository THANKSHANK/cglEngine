"""
This file contains the fragment shader classes for the different types of shading
"""
import glm
class fragment_shader_solid:
    def __init__(self):
        pass
    def fragment_shader(self,p0,p1,p2,alpha,beta,gamma,uniforms):
        r = p0.get_varying('color')[0]*alpha + p1.get_varying('color')[0]*beta + p2.get_varying('color')[0]*gamma
        g = p0.get_varying('color')[1]*alpha + p1.get_varying('color')[1]*beta + p2.get_varying('color')[1]*gamma
        b = p0.get_varying('color')[2]*alpha + p1.get_varying('color')[2]*beta + p2.get_varying('color')[2]*gamma
        return [r,g,b]

class fragment_shader_solid_with_edge:
    def __init__(self):
        pass
    def fragment_shader(self,p0,p1,p2,alpha,beta,gamma,uniforms):
        if alpha <= 0.05 or beta <= 0.05 or gamma <= 0.05:
            return[uniforms['edge_color'][0],uniforms['edge_color'][1],uniforms['edge_color'][2]]
        else:
            r = p0.get_varying('color')[0]*alpha + p1.get_varying('color')[0]*beta + p2.get_varying('color')[0]*gamma
            g = p0.get_varying('color')[1]*alpha + p1.get_varying('color')[1]*beta + p2.get_varying('color')[1]*gamma
            b = p0.get_varying('color')[2]*alpha + p1.get_varying('color')[2]*beta + p2.get_varying('color')[2]*gamma
            return [r,g,b]



class fragment_shader_Gouraud:
    def __init__(self):
        pass
    def fragment_shader(self,p0,p1,p2,alpha,beta,gamma,uniforms):
        r = p0.get_varying('color')[0]*alpha + p1.get_varying('color')[0]*beta + p2.get_varying('color')[0]*gamma
        g = p0.get_varying('color')[1]*alpha + p1.get_varying('color')[1]*beta + p2.get_varying('color')[1]*gamma
        b = p0.get_varying('color')[2]*alpha + p1.get_varying('color')[2]*beta + p2.get_varying('color')[2]*gamma
        return [r,g,b]


class fragment_shader_Phong:
    def __init__(self):
        pass
    def fragment_shader(self,p0,p1,p2,alpha,beta,gamma,uniforms):
        # interpolate the normal

        n = p0.get_varying('normal')*alpha + p1.get_varying('normal')*beta + p2.get_varying('normal')*gamma
        n = glm.normalize(n)
        vp = glm.vec3(p0.get_varying('pos')*alpha + p1.get_varying('pos')*beta + p2.get_varying('pos')*gamma)

        ka, kd, ks = uniforms['k']
        ambient_color = glm.vec3(uniforms['amb_color'])
        object_color = glm.vec3(uniforms['ocolor'])
        specular_color = glm.vec3(uniforms['scolor'])
        light_color = glm.vec3(uniforms['lightcolor'])
        light_pos = glm.vec3(uniforms['lightpos'])

        ambient = ka * ambient_color * object_color

        L = glm.normalize(light_pos - vp)

        dotLN = max(glm.dot(L, n), 0.0)
        diffuse = kd * dotLN * object_color * light_color

        Vp = glm.normalize(-vp)
        R = glm.reflect(-L, n)
        dotRV = max(glm.dot(R, Vp), 0.0)
        specular = ks * pow(dotRV, uniforms['exponent']) * specular_color * light_color

        color = ambient + diffuse + specular
        color = glm.clamp(color, 0.0, 1.0)

        return color




class fragment_shader_texture:
    def __init__(self):
        pass
    def fragment_shader(self,p0,p1,p2,alpha,beta,gamma,uniforms):
        u = alpha*p0.get_varying('uvs')[0] + beta*p1.get_varying('uvs')[0] + gamma*p2.get_varying('uvs')[0]
        v = alpha*p0.get_varying('uvs')[1] + beta*p1.get_varying('uvs')[1] + gamma*p2.get_varying('uvs')[1]
        tex_width = uniforms['tex_width']
        tex_height = uniforms['tex_height']
        u = int(u*tex_width)
        v = int(v*tex_height)
        u = min(max(u, 0), tex_width-1)
        v = min(max(v, 0), tex_height-1)
        color = uniforms['texture'].getpixel((u,v))
        return [color[0]/255.0, color[1]/255.0, color[2]/255.0]




class fragment_shader_checkerboard:
    def __init__(self):
        pass
    def fragment_shader(self,p0,p1,p2,alpha,beta,gamma,uniforms):

        color1 = uniforms['color1']
        color2 = uniforms['color2']
        checksize = uniforms['checksize']
        u = alpha*p0.get_varying('uvs')[0] + beta*p1.get_varying('uvs')[0] + gamma*p2.get_varying('uvs')[0]
        v = alpha*p0.get_varying('uvs')[1] + beta*p1.get_varying('uvs')[1] + gamma*p2.get_varying('uvs')[1]
        u = int(u/checksize)
        v = int(v/checksize)
        if (u+v) %2 == 0:
            return color1
        else:
            return color2



