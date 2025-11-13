'''
# Sigma3D
- Cool
- Amazing
- Cool
- Amazing
'''

from math import atan2
from os import system as _system
import time
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

try: 
    import pygame
except ImportError:
    print("Installing pygame... Press Ctrl + C to cancel.")
    _system('python3 -m pip install pygame')
    import pygame
from pygame.locals import *

try: 
    import numpy as np
except ImportError:
    print("Installing numpy... Press Ctrl + C to cancel.")
    _system('python3 -m pip install numpy')
    import numpy as np

try: 
    import moderngl
except ImportError:
    print("Installing moderngl... Press Ctrl + C to cancel.")
    _system('python3 -m pip install moderngl')
    import moderngl

try: 
    from glm import *
except ImportError:
    print("Installing pyglm... Press Ctrl + C to cancel.")
    _system('python3 -m pip install pyglm')
    from glm import *

try: length(0) # type: ignore
except NameError:
    length = lambda x:0
    rotate = lambda x, y, z:0

print("Sigma3D welcomes you.")

_ctx = None
_phong_shader = None
_flat_shader = None
_default_poly_shader = None
_default_circle_shader = None
_enviroment_shader = None
default_shaders = {}

_global_window = None
_active_camera = None
_active_ui_camera = None
_active_enviroment = None

class Physics:
    '''
    Provides some basic methods to help in physics calculations.\n
    NOTE: Incomplete, please don't use this yet.
    '''

    class Point:
        '''
        A point object.
        '''
        def __init__(self, position: vec3):
            self.position = vec3(position)
            self.velocity = vec3(0)
            self.force = vec3(0)
            self.mass = 1.0
        
        def step(self, delta_time = 1 / 60) -> None:
            '''
            Simulate dynamics.
            '''

            self.velocity += (self.force / self.mass) * delta_time
            self.position += self.velocity * delta_time
            self.force = vec3(0)
        
        def accelerate(self, acceleration: vec3) -> None:
            '''
            Accelerate the object.
            '''

            self.force += acceleration * self.mass
        
        def apply_force(self, force: vec3) -> None:
            '''
            Apply force to the object.
            '''

            self.force += force

class Shader:
    '''
    Provides GLSL shader support.
    '''

    def __init__(self):
        self.program = None

    @staticmethod
    def load(path: str):
        '''
        Load a shader file.
        '''

        with open(path) as file:
            return Shader.load_from_buffer(file.read())
    
    @staticmethod 
    def load_from_buffer(buffer: str):
        '''
        Load shader from buffer(string)
        '''

        vs_source = ""
        fs_source = ""
        source_type = 0

        for line in buffer.split('\n'):
            if line.lstrip().startswith("#vertex"):
                source_type = 1
                continue
            elif line.lstrip().startswith("#fragment"):
                source_type = 2
                continue
            
            if source_type == 1: vs_source += line + '\n'
            if source_type == 2: fs_source += line + '\n'

        shader = Shader()
        shader.program = _ctx.program(
            vertex_shader=vs_source,
            fragment_shader=fs_source
        )
        return shader

    @staticmethod 
    def load_from_buffer2(vertex:str, fragment:str):
        '''
        Load shader from vertex shader & fragment shader buffers.
        '''
        
        return Shader.load_from_buffer(
            "#vertex\n" + vertex + '\n#fragment\n' + fragment
        )

    def set_uniform(self, name: str, value) -> None:
        '''
        Set the value of a shader uniform. (vectors and matrices must be provided as a tuple or list)
        '''

        try:self.program[name] = value
        except KeyError: pass

    def get_uniform(self, name: str) -> any:
        '''
        Get the value of a shader uniform.
        '''

        try:return self.program[name]
        except KeyError: pass

class Vertex:
    '''
    Class for storing a vertex in three dimensional geometry.
    '''

    def __init__(self, position: vec3, normal: vec3 = vec3(0)):
        self.position = vec3(position)
        self.normal = vec3(normal)

class Vertex2D:
    '''
    Class for storing a vertex in two dimensional geometry.
    '''
    
    def __init__(self, position: vec2, color: vec4 = vec4(1)):
        self.position = vec2(position)
        self.color = vec4(color)

class Polygon2D:
    '''
    A 2D mesh.
    '''

    def __init__(
                self, position: vec2 = vec2(0), rotation: float = 0, scale: vec2 = vec2(1), color: vec4 = vec4(1),
                 vertices: list[Vertex2D] = [], shader: Shader = None
            ):
            self.position = vec2(position)
            self.rotation = rotation
            self.scale = vec2(scale)
            self.color = vec4(color)
            self.vertices = vertices
            self._vbo = None
            self._vao = None
            self.shader = _default_poly_shader if shader is None else shader

            self.refresh()
        
    def refresh(self) -> None:
        '''
        Update the shader (shoul
        d be called after modifying the polygon).
        '''
        if len(self.vertices) == 0: return

        verts = []
        for i in range(len(self.vertices)):
            verts.append(self.vertices[i].position.x)
            verts.append(self.vertices[i].position.y)
            verts.append(self.vertices[i].color.x)
            verts.append(self.vertices[i].color.y)
            verts.append(self.vertices[i].color.z)
            verts.append(self.vertices[i].color.w)

        self._vert = np.array(verts, dtype='f4')
        self._vbo = _ctx.buffer(self._vert.tobytes())
        self._vao = _ctx.vertex_array(self.shader.program, [(self._vbo, '2f 4f', 'in_vert', 'in_col')])
    
    def render(self) -> None:
        '''
        Renders the polygon.
        '''

        if _global_window is None: 
            _raise_error("Inactive window: No active window to render to.", RuntimeError)
        if _active_ui_camera == None:
            _raise_error("Inactive UI camera: Cannot render an object when no camera is active.", RuntimeError)

        ws = 2.0 / vec2(_global_window.get_size())
        ws *= vec2(1, -1)
        wo = vec2(-1, 1)

        transformed_pos = rotate2d(self.position - _active_ui_camera.position - _global_window.get_size() * 0.5, _active_ui_camera.rotation)\
                                * _active_ui_camera.zoom + _global_window.get_size() * 0.5
        
        self.shader.set_uniform('uposition', (transformed_pos * ws + wo).to_tuple())
        self.shader.set_uniform('uscale', (self.scale * _active_ui_camera.zoom * ws * vec2(1, 1.0 / _global_window.aspect_ratio)).to_tuple())
        self.shader.set_uniform('urotation', self.rotation - _active_ui_camera.rotation)
        self.shader.set_uniform('ucolor', self.color.to_tuple())
        self.shader.set_uniform('uaspect_ratio', _global_window.aspect_ratio)

        if _active_camera is not None:
            if self.shader.get_uniform('uMVP'):
                tp = _active_camera.get_vp_matrix()
                tup = []
                for t in tp:
                    tup.append(t[0])
                    tup.append(t[1])
                    tup.append(t[2])
                    tup.append(t[3])
                
                self.shader.set_uniform('uMVP', tup)
            self.shader.set_uniform('uview_pos', _active_camera.get_transform().position.to_tuple())

        self._vao.render(moderngl.TRIANGLES)
    
    @staticmethod
    def create_rectangle(position: vec2 = vec2(0), scale:vec2 = vec2(1), rotation: float = 0, color: vec4 = vec4(1)):
        '''
        Create a rectangle.
        '''

        return Polygon2D(
            position=position, scale=scale, rotation=rotation, color=color,
            vertices=[
                Vertex2D(0.5 * vec2(-1, -1)),
                Vertex2D(0.5 * vec2(-1, 1)),
                Vertex2D(0.5 * vec2(1, -1)),
                Vertex2D(0.5 * vec2(1, 1)),
                Vertex2D(0.5 * vec2(1, -1)),
                Vertex2D(0.5 * vec2(-1, 1)),
            ]
        )
    
    '''
    Create a circle.
    '''
    @staticmethod
    def create_circle(position: vec2 = vec2(0), radius:float = 1, color: vec4 = vec4(1)):
        return Polygon2D(
            position=position, scale=vec2(radius), rotation=0, color=color,
            shader=_default_circle_shader,
            vertices=[
                Vertex2D(vec2(-1, -1)),
                Vertex2D(vec2(-1, 1)),
                Vertex2D(vec2(1, -1)),
                Vertex2D(vec2(1, 1)),
                Vertex2D(vec2(1, -1)),
                Vertex2D(vec2(-1, 1)),
            ],
        )
        
class Mesh:
    '''
    A 3D Triangle Mesh.
    '''

    def __init__(self, vertices: list[Vertex] = [], preserve_normals: bool = False):
        self.vertices = vertices
        self._vbo = None
        self._vao = None

        if not preserve_normals: self.regenerate_normals()
        self.refresh()
    
    
    def regenerate_normals(self) -> None:
        '''
        Recalculate the normals for each triangle.
        '''

        for i in range(len(self.vertices) // 3):
            norm = normalize(cross(
                self.vertices[i * 3].position - self.vertices[i * 3 + 1].position,
                self.vertices[i * 3 + 1].position - self.vertices[i * 3 + 2].position
            ))

            cen = normalize(self.vertices[i * 3].position + self.vertices[i * 3 + 1].position +
                    self.vertices[i * 3 + 2].position)

            if dot(norm, cen) < 0: norm *= -1

            self.vertices[i * 3].normal = norm
            self.vertices[i * 3 + 2].normal = norm
            self.vertices[i * 3 + 1].normal = norm

    def refresh(self) -> None:
        '''
        Update the shader (should be called after modifying the mesh).
        '''

        if len(self.vertices) == 0: return

        verts = []
        for i in range(len(self.vertices)):
            verts.append(self.vertices[i].position.x)
            verts.append(self.vertices[i].position.y)
            verts.append(self.vertices[i].position.z)
            verts.append(self.vertices[i].normal.x)
            verts.append(self.vertices[i].normal.y)
            verts.append(self.vertices[i].normal.z)

        self._vert = np.array(verts, dtype='f4')
        self._vbo = _ctx.buffer(self._vert.tobytes())
    
    '''
    Binds the mesh to prepare it for rendering.
    '''
    def bind(self, shader: Shader) -> None:
        if _active_camera == None: return
        
        tp = _active_camera.get_vp_matrix()
        tup = []
        for t in tp:
            tup.append(t[0])
            tup.append(t[1])
            tup.append(t[2])
            tup.append(t[3])
        
        shader.set_uniform('uMVP', tup)
        self._vao = _ctx.vertex_array(shader.program, [(self._vbo, '3f 3f', 'in_vert', 'in_norm')])
    
    def _render_vao(self):
        '''
        Actually renders the object, assuming all the neccessary data has been sent to the shader.
        '''

        self._vao.render(moderngl.TRIANGLES)
    
    @staticmethod
    def add_double_sided(verts: list[Vertex]):
        '''
        Takes a list of vertices, copies it again with inverted normals to
        create a "back side".
        '''

        for v in range(len(verts) // 3):
            verts.append(Vertex(vec3(*verts[v * 3].position.to_tuple())))
            verts.append(Vertex(vec3(*verts[v * 3 + 2].position.to_tuple())))
            verts.append(Vertex(vec3(*verts[v * 3 + 1].position.to_tuple())))

    @staticmethod
    def create_box(double_sided: bool = False):
        '''
        Create a unit cube.
        '''

        loc = [
            vec3(-1, -1, -1),
            vec3(-1, -1,  1),
            vec3(-1,  1, -1),
            vec3(-1,  1,  1),

            vec3( 1, -1, -1),
            vec3( 1, -1,  1),
            vec3( 1,  1, -1),
            vec3( 1,  1,  1),
        ]

        v=[
            0, 1, 2, 1, 3, 2,
            4, 6, 5, 5, 6, 7,
            0, 4, 1, 5, 1, 4,
            2, 3, 6, 7, 6, 3,
            0, 2, 4, 4, 2, 6,
            1, 5, 3, 3, 5, 7,
        ]

        verts = [Vertex(loc[x]) for x in v]
        if double_sided: Mesh.add_double_sided(verts)

        return Mesh(vertices=verts)
    
    @staticmethod
    def create_quad(double_sided: bool = True):
        '''
        Create a unit quad.
        '''

        verts = [
            Vertex(vec3(-1, -1, 0)),
            Vertex(vec3(-1,  1, 0)),
            Vertex(vec3( 1,  1, 0)),

            Vertex(vec3(-1, -1, 0)),
            Vertex(vec3( 1,  1, 0)),
            Vertex(vec3( 1, -1, 0)),
        ]

        if double_sided: Mesh.add_double_sided(verts)

        return Mesh(vertices=verts)

    @staticmethod
    def create_disc(divisions: int = 32, double_sided: bool = True):
        '''
        Create a unit disc (circle).
        '''

        verts = []
        angle = 0
        p_angle = (2 * pi()) / divisions
        i = 0
        while i < divisions:
            verts.append(Vertex(vec3(cos(angle), sin(angle), 0)))
            verts.append(Vertex(vec3(0)))
            angle += p_angle
            verts.append(Vertex(vec3(cos(angle), sin(angle), 0)))
            i += 1
        
        if double_sided: Mesh.add_double_sided(verts)
        
        return Mesh(vertices=verts)
    
    @staticmethod
    def create_sphere(divisions: int = 32, double_sided: bool = False):
        '''
        Create a unit sphere.
        '''

        verts = []
        d = (1 / divisions) * 2

        for i in range(divisions):
            x = (i / divisions - 0.5) * 2.0
            for j in range(divisions):
                y = (j / divisions - 0.5) * 2.0

                verts.append(Vertex(vec3(x, y, 1)))
                verts.append(Vertex(vec3(x + d, y, 1)))
                verts.append(Vertex(vec3(x, y + d, 1)))
                verts.append(Vertex(vec3(x + d, y, 1)))
                verts.append(Vertex(vec3(x + d, y + d, 1)))
                verts.append(Vertex(vec3(x, y + d, 1)))
                
                verts.append(Vertex(vec3(x + d, y, -1)))
                verts.append(Vertex(vec3(x, y, -1)))
                verts.append(Vertex(vec3(x, y + d, -1)))
                verts.append(Vertex(vec3(x + d, y, -1)))
                verts.append(Vertex(vec3(x, y + d, -1)))
                verts.append(Vertex(vec3(x + d, y + d, -1)))
                
                verts.append(Vertex(vec3(-1, x + d, y)))
                verts.append(Vertex(vec3(-1, x, y)))
                verts.append(Vertex(vec3(-1, x, y + d)))
                verts.append(Vertex(vec3(-1, x + d, y)))
                verts.append(Vertex(vec3(-1, x, y + d)))
                verts.append(Vertex(vec3(-1, x + d, y + d)))

                verts.append(Vertex(vec3(1, x, y)))
                verts.append(Vertex(vec3(1, x + d, y)))
                verts.append(Vertex(vec3(1, x, y + d)))
                verts.append(Vertex(vec3(1, x + d, y)))
                verts.append(Vertex(vec3(1, x + d, y + d)))
                verts.append(Vertex(vec3(1, x, y + d)))

                verts.append(Vertex(vec3(x + d, 1, y)))
                verts.append(Vertex(vec3(x, 1, y)))
                verts.append(Vertex(vec3(x, 1, y + d)))
                verts.append(Vertex(vec3(x + d, 1, y)))
                verts.append(Vertex(vec3(x, 1, y + d)))
                verts.append(Vertex(vec3(x + d, 1, y + d)))

                verts.append(Vertex(vec3(x, -1, y)))
                verts.append(Vertex(vec3(x + d, -1, y)))
                verts.append(Vertex(vec3(x, -1, y + d)))
                verts.append(Vertex(vec3(x + d, -1, y)))
                verts.append(Vertex(vec3(x + d, -1, y + d)))
                verts.append(Vertex(vec3(x, -1, y + d)))
        
        for v in verts:
        
            v.position = normalize(v.position)
        if double_sided: Mesh.add_double_sided(verts)
        
        return Mesh(vertices=verts)


class Component:
    '''
    Base class for all components.
    '''

    def __init__(self): 
        self.entity_id = 0

class Transform(Component):
    '''
    Defines position, orientation and scale of an Entity in 3D space.
    '''

    def __init__(self, position: vec3 = vec3(0), rotation: vec3 = vec3(0), scale: vec3 = vec3(1)):
        super().__init__()
        self.position = vec3(position)
        self.rotation = vec3(rotation)
        self.scale = vec3(scale)
    
    def get_forward(self) -> vec3:
        '''
        Get the forward direction.
        '''

        vec = rotate(vec3(0, 0, 1), self.rotation.x, vec3(-1, 0, 0))
        vec = rotate(vec, self.rotation.y, vec3(0, -1, 0))
        vec = rotate(vec, self.rotation.z, vec3(0, 0, -1))
        return vec

    def get_right(self) -> vec3:
        '''
        Get the right direction.
        '''
        
        return cross(self.get_forward(), vec3(0, 1, 0))
    
    def get_up(self) -> vec3:
        '''
        Get the up direction.
        '''

        return cross(self.get_right(), self.get_forward())
    
    def get_yaw_pitch(self) -> vec2:
        '''
        Get the Yaw (rotation in y-axis) and Pitch (rotation in x-axis).
        '''

        forward = self.get_forward()
        return vec2(asin(forward.y), atan2(forward.x, forward.z))
    
class Material:
    '''
    Base class for all Materials.
    '''

    def __init__(self, shader: Shader = None, enable_lighting: bool = False):
        self.shader = shader
        self.enable_lighting = enable_lighting
    
    '''
    Bind the material with the shader.
    '''
    def bind(self): ...

class FlatMaterial(Material):
    '''
    A flat material with no shading.
    '''

    def __init__(self, color: vec3 = vec3(1)):
        super().__init__(_flat_shader, False)
        self.color = vec3(color)
    
    def bind(self):
        '''
        Bind the material with the shader.
        '''
            
        self.shader.set_uniform('ucolor', self.color.to_tuple())
        self.shader.set_uniform('ucolor', self.color.to_tuple())

class PhongMaterial(Material):
    '''
    A phong material, adds ambient, diffuse and specular lighting.
    '''

    def __init__(
            self,
            color: vec3 = vec3(1),
            ambient_strength: float = 0.15,
            diffuse_strength: float = 0.75,
            specular_strength: float = 0.5,
            specular_exponent: float = 32
        ):
        super().__init__(_phong_shader, True)
        self.color = vec3(color)
        self.ambient_strength = ambient_strength
        self.diffuse_strength = diffuse_strength
        self.specular_strength = specular_strength
        self.specular_exponent = specular_exponent
    
    def bind(self):
        '''
        Bind the material with the shader.
        '''

        self.shader.set_uniform('ucolor', self.color.to_tuple())
        self.shader.set_uniform('uambient_strength', self.ambient_strength)
        self.shader.set_uniform('udiffuse_strength', self.diffuse_strength)
        self.shader.set_uniform('uspecular_strength', self.specular_strength)
        self.shader.set_uniform('uspecular_exponent', self.specular_exponent)

class MeshRenderer(Component):
    '''
    Component used for rendering mesh.
    '''

    def __init__(self, mesh: Mesh = None, material: Material = None):
        super().__init__()
        self.mesh = mesh
        self.material = material

class Light(Component):
    '''
    Base class for light.
    '''

    def __init__(self, color: vec3 = vec3(1), intensity: float = 1.0):
        super().__init__()
        self.intensity = intensity
        self.color = vec3(color)
        self.active = True
        _lights.append(self)

class DirectionalLight(Light):
    '''
    Directional Light, can be used to simulate sunlight.
    '''

    def __init__(
            self, color: vec3 = vec3(1), intensity: float = 1.0, sun_color: vec3 = vec3(1),
            sun_strength: float = 2, sun_exponent: float = 5, sun_tint = vec3(0.95, 0.8, 1), is_sun = True
        ):
        super().__init__(color, intensity)
        self.sun_color = sun_color
        self.sun_strength = sun_strength
        self.sun_exponent = sun_exponent
        self.sun_tint = sun_tint
        self.is_sun = is_sun

class PointLight(Light):
    '''
    Point Light, can be used to simulate objects like lanters.
    '''

    def __init__(self, color: vec3 = vec3(1), intensity: float = 1.0):
        super().__init__(color, intensity)

class _DirectionalLightShaderBuffer:
    '''
    For handling DirectionalLight shader buffer.
    '''

    def __init__(self):
       self.ssbo = _ctx.buffer(dynamic=True,reserve=1)
       self.buffer = []
       self.buffer_stride = 16 # in floats, 16 * 4 = 64 bytes
    
    def clear(self) -> None:
        '''
        Clear the buffer.
        '''

        self.buffer.clear()
    
    def append(self, transform: Transform, light: DirectionalLight) -> None:
        '''
        Append to the buffer.
        '''

        direction = transform.get_forward()
        self.buffer += [
            light.intensity,
            light.color.x, light.color.y, light.color.z,
            direction.x, direction.y, direction.z,
            light.sun_color.x, light.sun_color.y, light.sun_color.z,
            light.sun_tint.x, light.sun_tint.y, light.sun_tint.z,
            light.sun_strength, light.sun_exponent, 1.0 if light.is_sun else 0.0
        ]

    # kindly ask the class to help us. (also you may want to watch your tone!)
    def plz_do_something(self, shader) -> None: # rename this to bind()...? no.
        '''
        Bind buffer with a shader.
        '''

        if(len(self.buffer) >= self.buffer_stride):
            self.ssbo.orphan(len(self.buffer) * 4)
            self.ssbo.write(np.array(self.buffer, dtype='f4'))
            self.ssbo.bind_to_storage_buffer(0)

        shader.set_uniform('udlcount', int(len(self.buffer) / self.buffer_stride))

class _PointLightShaderBuffer:
    '''
    For handling PointLight shader buffer.
    '''

    def __init__(self):
       self.ssbo = _ctx.buffer(dynamic=True,reserve=1)
       self.buffer = []
       self.buffer_stride = 7 # in floats, 7 * 4 = 28 bytes
    
    def clear(self) -> None:
        '''
        Clear the buffer.
        '''

        self.buffer.clear()
    
    def append(self, transform: Transform, light: PointLight) -> None:
        '''
        Append to the buffer.
        '''

        self.buffer += [
            light.intensity,
            light.color.x, light.color.y, light.color.z,
            transform.position.x, transform.position.y, transform.position.z
        ]

    # kindly ask the class to help us. (also you may want to watch your tone!)
    def plz_do_something(self, shader) -> None: # rename this to bind()...? no.
        '''
        Bind buffer with a shader.
        '''

        if(len(self.buffer) >= self.buffer_stride):
            self.ssbo.orphan(len(self.buffer) * 4)
            self.ssbo.write(np.array(self.buffer, dtype='f4'))
            self.ssbo.bind_to_storage_buffer(1)

        shader.set_uniform('uplcount', int(len(self.buffer) / self.buffer_stride))

class System:
    '''
    Base class for all Systems.
    '''

    def __init__(self): ...
    def update(self) -> None: ...

class LightingSystem(System):
    '''
    For lighting stuff.
    '''

    def __init__(self):
        super().__init__()
        self.dlsb = _DirectionalLightShaderBuffer()
        self.plsb = _PointLightShaderBuffer()
    
    def update(self) -> None:
        '''
        Update the lighting.
        '''

        self.dlsb.clear()
        self.plsb.clear()

        for entity in _entities:
            dl = entity.get_component(DirectionalLight)
            pl = entity.get_component(PointLight)
            tr = entity.get_component(Transform)
            if tr is None: continue
            if dl is not None and dl.active: self.dlsb.append(tr, dl)
            if pl is not None and dl.active: self.plsb.append(tr, pl)
    
    def bind_with_shader(self, shader: Shader):
        '''
        Bind lighting buffers with a shader.
        '''

        self.dlsb.plz_do_something(shader)
        self.plsb.plz_do_something(shader)

class RenderSystem(System):
    '''
    For rendering stuff.
    '''

    def __init__(self):
        super().__init__()
        self.clear_color = vec4(0.05)
        self._dlsb = _DirectionalLightShaderBuffer()
        self._plsb = _PointLightShaderBuffer()
        _ctx.depth_func = "<="

    def draw_mesh(self, transform: Transform, mesh: Mesh, material: Material) -> None:
        '''
        Draw a specified mesh.
        '''
        
        if _active_camera is None: return
        if _global_window is None: return
        if material is None: material = FlatMaterial()
        if material.enable_lighting: _global_window.lighting_system.bind_with_shader(material.shader)

        camera_transform = _active_camera.get_transform()
        if camera_transform is None: return
        
        material.shader.set_uniform('uview_pos', camera_transform.position.to_tuple())
        material.shader.set_uniform('uposition', transform.position.to_tuple())
        material.shader.set_uniform('urotation', transform.rotation.to_tuple())
        material.shader.set_uniform('uscale', transform.scale.to_tuple())
        material.bind()
        mesh.bind(material.shader)
        mesh._render_vao()
    
    def draw_enviroment(self):
        '''
        Draws the active enviroment.
        '''

        global _active_enviroment
        global _active_camera
        if _active_enviroment is None: return
        if _active_camera is None: return

        _active_enviroment.render()

    def update(self) -> None:
        '''
        Draws all the active scene objects, called at the end of the callback loop.
        '''
         
        for entity in _entities:
            tr: MeshRenderer = entity.get_component(Transform)
            mr: MeshRenderer = entity.get_component(MeshRenderer)
            if tr is None or mr is None: continue
            self.draw_mesh(tr, mr.mesh, mr.material)
            
    def clear(self) -> None:
        '''
        Clear the screen.
        '''

        _ctx.clear(
            self.clear_color.x,
            self.clear_color.y,
            self.clear_color.z,
            self.clear_color.w,
        )
    
    def _set_gl_flag(self, name, flag: bool) -> None:
        '''
        Set OpenGL Flags.
        '''

        if flag: _ctx.enable(name)
        else: _ctx.disable(name)
    
    def set_cull_face(self, flag: bool) -> None:
        '''
        Enable/Disable face culling.
        '''

        self._set_gl_flag(_ctx.CULL_FACE, flag)

    def set_depth_test(self, flag: bool) -> None:
        '''
        Enable/Disable depth test.
        '''

        self._set_gl_flag(_ctx.DEPTH_TEST, flag)

    def set_alpha_blending(self, flag: bool) -> None:
        '''
        Enable/Disable alpha blending.
        '''

        self._set_gl_flag(_ctx.BLEND, flag)
    
class Camera(Component):
    '''
    A 3D camera class.
    '''

    def __init__(self, FOV : float = radians(60), near: float = 0.2, far: float = 1000, up: vec3 = vec3(0, 1, 0)):
        super().__init__()
        self.FOV = FOV
        self.near = near
        self.far = far
        self.up = vec3(up)
    
    def get_transform(self) -> Transform | None:
        '''
        Returns transform of the Camera (Transform of the entity Camera belongs to).
        '''
        
        entity = get_entity_from_id(self.entity_id)
        if entity is None:
            _raise_error("A Camera component must be attached to an entity.")
        return entity.get_component(Transform)

    def get_vp_matrix(self) -> mat4x4:
        '''
        Get the Viewport-Projection matrix for the camera.
        '''

        transform = self.get_transform()
        if transform is None: return

        view = lookAt(transform.position, transform.position + transform.get_forward(), self.up)
        proj = perspective(self.FOV, _global_window.aspect_ratio, self.near, self.far)
        return proj * view

    def use(self) -> None:
        '''
        Activate the camera.
        '''
    
        global _active_camera
        _active_camera = self
    
    @staticmethod
    def use_none() -> None:
        '''
        Deactivate the currently active camera.
        '''
        
        global _active_camera
        _active_camera = None
    
    def control(
            self, active_key = K_ESCAPE, move_keys = (K_w, K_a, K_s, K_d, K_e, K_q),
            movement_speed: float = 10.0, camera_sensitivity: float | vec2 = 0.002
        ) -> None:
        '''
        Control the camera.
        #### Intended for debugging.
        '''
        
        if _global_window is None: return
        if _global_window.is_key_pressed(active_key):
            _global_window.unlock_mouse() if _global_window.is_mouse_locked() else _global_window.lock_mouse()
        if not _global_window.is_mouse_locked(): return

        transform = self.get_transform()
        if transform is None: return

        if type(camera_sensitivity) == vec2:
            transform.rotation.y += _global_window.mouse_delta.x * camera_sensitivity.x
            transform.rotation.x -= _global_window.mouse_delta.y * camera_sensitivity.y
        else:
            transform.rotation.y += _global_window.mouse_delta.x * camera_sensitivity
            transform.rotation.x -= _global_window.mouse_delta.y * camera_sensitivity
        
        if transform.rotation.x > pi() / 2: transform.rotation.x = pi() / 2 - 0.0001
        if transform.rotation.x < -pi() / 2: transform.rotation.x = -pi() / 2 + 0.0001

        movement = vec3(0)
        if _global_window.is_key_down(move_keys[0]): movement += transform.get_forward()
        if _global_window.is_key_down(move_keys[1]): movement -= transform.get_right()
        if _global_window.is_key_down(move_keys[2]): movement -= transform.get_forward()
        if _global_window.is_key_down(move_keys[3]): movement += transform.get_right()
        if _global_window.is_key_down(move_keys[4]): movement += transform.get_up()
        if _global_window.is_key_down(move_keys[5]): movement -= transform.get_up()

        if movement == vec3(0): return

        movement = normalize(movement)
        transform.position += movement * _global_window.delta_time * movement_speed

class Enviroment(Component):
    '''
    The enviroment or the background.
    '''

    Sky = 1
    Solid = 2
    __MAX__ = 3

    def __init__(self, mode: int = Sky, sky_color: vec3 = vec3(0.5, 0.7, 1.0), solid_color: vec3 = vec3(0.0)):
        super().__init__()
        self._camera = Camera2D()
        self.polygon = Polygon2D.create_rectangle()
        self.polygon.shader = None
        self._mode: int | None = None
        self.set_mode(mode)

        self.sky_color = sky_color
        self.solid_color = solid_color

    def set_mode(self, mode: int) -> None:
        '''
        Set the enviroment mode.
        '''
        
        if mode >= self.__MAX__ or mode < 0:
            _raise_error(f"Invalid argument: Enviroment.set_mode({mode}) -> Unexpected value.")
        
        self._mode = mode
        if   mode == self.Sky:   self.polygon.shader = _enviroment_shader
        elif mode == self.Solid: self.polygon.shader = _default_poly_shader

        self.polygon.refresh()

    def use(self) -> None:
        '''
        Activate the Enviroment.
        '''
        
        global _active_enviroment
        _active_enviroment = self
        
    @staticmethod
    def use_none():
        '''
        Deactivate the currently active Enviroment.
        '''
        
        global _active_enviroment
        _active_enviroment = None
    
    def _bind(self):
        global _active_camera
        global _global_window
        if _active_camera is None: return
        if _global_window is None: return

        if self._mode != self.Sky:
            self.polygon.scale = _global_window.get_size()
            self.polygon.position = self.polygon.scale * 0.5
            self.polygon.color = vec4(self.solid_color, 1.0)
            return

        yaw_pitch = _active_camera.get_transform().get_yaw_pitch();
        self.polygon.shader.set_uniform("yaw", yaw_pitch.x)
        self.polygon.shader.set_uniform("pitch", yaw_pitch.y)
        self.polygon.shader.set_uniform("sky_color", self.sky_color.to_tuple())
        _global_window.lighting_system.bind_with_shader(self.polygon.shader)
    
    def render(self):
        '''
        Render the Enviroment.\n
        #### NOTE: The active Enviroment is rendered just before rendering The scene,
        #### there is no need to call this function unless you want to.
        '''

        global _active_ui_camera
        self._bind()
        active_cam = _active_ui_camera
        self._camera.use()
        self.polygon.render()
        _active_ui_camera = active_cam

class Camera2D:
    '''
    Camera class for 2D rendering.
    '''

    def __init__(self, position: vec2 = vec2(0), zoom: float = 1, rotation: float = 0):
        self.position = vec2(position)
        self.zoom = zoom
        self.rotation = rotation
    
    def use(self):
        '''
        Activate the Camera2D
        '''
        
        global _active_ui_camera
        _active_ui_camera = self
    
    @staticmethod
    def use_none():
        '''
        Deactivate the currently active Camera2D
        '''
        
        global _active_ui_camera
        _active_ui_camera = None
    
class Window:
    '''
    Creates a window and OpenGL context.
    '''

    def __init__(self, width: int, height: int, title: str = "Sigma3D Window", max_fps: int = None):
        global _ctx

        # Context
        pygame.init()
        self._width = width
        self._height = height
        self._title = str(title)
        self._window = pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
        self.aspect_ratio = 1
        pygame.display.set_caption(self._title)
        _ctx = moderngl.create_context()

        # Input
        self._key_state = {}
        self._p_key_state = {}
        self._c_key_state = {}
        self.mouse_pos = vec2(0)
        self.mouse_delta = vec2(0)
        self._mouse_state = [False,] * 10
        self._p_mouse_state = [False,] * 10
        self._c_mouse_state = [False,] * 10

        # Render
        self.render_system = RenderSystem()
        self.lighting_system = LightingSystem()
        self._maxFPS = max_fps
        self._clock = pygame.time.Clock()
        self.delta_time = 1
        self.time_elapsed = 0.0
        self._start_point = time.perf_counter()

        # Flags
        self.render_system.set_cull_face(True)
        self.render_system.set_depth_test(True)
        self.render_system.set_alpha_blending(True)

        _init_utils(self)

    def set_max_fps(self, fps: int) -> None:
        '''
        Set the maximum framerate.
        '''
        
        self._maxFPS = fps

    def set_size(self, width: int, height: int) -> None:
        '''
        Set the size of the window.
        '''
        
        self._width = width
        self._height = height
        pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)

    def set_title(self, title: str) -> None:
        '''
        Set the title of the window.
        '''
        
        self._title = str(title)
        pygame.display.set_caption(title)
    
    def get_size(self) -> vec2:
        '''
        Returns the size eof the window.
        '''
        
        return vec2(self._width, self._height)

    def get_title(self) -> str:
        '''
        Returns the title of the window.
        '''
        
        return self._title
    
    def lock_mouse(self) -> None:
        '''
        Lock the mouse pointer.
        '''
        
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
    
    def unlock_mouse(self) -> None:
        '''
        Unlock the mouse pointer.
        '''
        
        pygame.event.set_grab(False)
        pygame.mouse.set_visible(True)
    
    def is_mouse_locked(self) -> bool:
        '''
        Returns whether the mouse is locked or not.
        '''
        
        return pygame.event.get_grab()
    
    def is_key_down(self, key) -> bool:
        '''
        Returns whether the given key is being pressed.
        '''
        
        try: return self._key_state[key]
        except KeyError: return False
    
    def is_key_pressed(self, key) -> bool:
        '''
        Returns whether the given key is just pressed in the current frame.
        '''
        
        s1 = False
        s2 = False
        try: s1 = self._c_key_state[key]
        except: s1 = False
        try: s2 = self._p_key_state[key]
        except: s2 = False
        return s1 == True and s2 == False
    
    def is_key_released(self, key) -> bool:
        '''
        Returns whether the given key is just released in the current frame.
        '''
        
        s1 = False
        s2 = False
        try: s1 = self._c_key_state[key]
        except: s1 = False
        try: s2 = self._p_key_state[key]
        except: s2 = False
        return s1 == False and s2 == True
    
    def is_mouse_down(self, button) -> bool:
        '''
        Returns whether the given mouse button is being pressed.
        '''
        
        try: return self._mouse_state[button]
        except KeyError: return False
    
    def is_mouse_pressed(self, button) -> bool:
        '''
        Returns whether the given mouse button is just pressed in the current frame.
        '''
        
        s1 = False
        s2 = False
        try: s1 = self._c_mouse_state[button]
        except: s1 = False
        try: s2 = self._p_mouse_state[button]
        except: s2 = False
        return s1 == True and s2 == False
    
    def is_mouse_released(self, button) -> bool:
        '''
        Returns whether the given mouse button is just released in the current frame.
        '''
        
        s1 = False
        s2 = False
        try: s1 = self._c_mouse_state[button]
        except: s1 = False
        try: s2 = self._p_mouse_state[button]
        except: s2 = False
        return s1 == False and s2 == True
    
    def get_fps(self) -> float:
        '''
        Returns the framerate of the window.
        '''
        
        if self.delta_time == 0: return 1000.0
        return 1.0 / self.delta_time
    
    def start_loop(self, callback: callable = lambda: None, draw_ui: callable = lambda: None) -> None:
        '''
        Starts the window loop.
        '''
        
        while True:
            start_time_ms = int(round(time.time() * 1000))
            for event in pygame.event.get():
                if event.type == pygame.QUIT: exit()
                if event.type == pygame.VIDEORESIZE:
                    self._width, self._height = event.size
                    self.set_size(self._width, self._height)
                    
                elif event.type == pygame.KEYDOWN:
                    self._key_state[event.key] = True
                elif event.type == pygame.KEYUP:
                    self._key_state[event.key] = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._mouse_state[event.button] = True
                elif event.type == pygame.MOUSEBUTTONUP:
                    self._mouse_state[event.button] = False

            self._update()
            callback()
            self.render_system.clear()
            self.lighting_system.update()

            # Background
            self.render_system.set_depth_test(False)
            self.render_system.set_cull_face(False)
            self.render_system.draw_enviroment()
            self.render_system.set_depth_test(True)
            self.render_system.set_cull_face(True)

            # Scene
            self.render_system.update()

            # UI / Overlay
            self.render_system.set_depth_test(False)
            self.render_system.set_cull_face(False)
            draw_ui()
            self.render_system.set_depth_test(True)
            self.render_system.set_cull_face(True)

            pygame.display.flip()

            if self._maxFPS is not None:
                self._clock.tick(self._maxFPS)
            end_time_ms = int(round(time.time() * 1000))
            self.delta_time = (end_time_ms - start_time_ms) / 1000
            self.time_elapsed = time.perf_counter() - self._start_point

    def _update(self) -> None:
        self.aspect_ratio = self.get_size().x / self.get_size().y
        _ctx.viewport = 0, 0, self.get_size().x, self.get_size().y

        self._p_key_state = self._c_key_state.copy()
        self._c_key_state = self._key_state.copy()
        self._p_mouse_state = self._c_mouse_state.copy()
        self._c_mouse_state = self._mouse_state.copy()
        self.mouse_pos = vec2(*pygame.mouse.get_pos())
        self.mouse_delta = vec2(*pygame.mouse.get_rel())

        if pygame.event.get_grab():
            pygame.mouse.set_pos(self.get_size().x / 2, self.get_size().y / 2)

class Entity:
    '''
    An Entity.
    '''

    def __init__(self, components: list[Component] = []):
        global _entities
        self.id = _gen_entity_id()
        self.components = []
        for component in components: self.add_component(component)
        _entities.append(self)
    
    def get_component(self, component_type) -> Component | None:
        '''
        Returns a given component. If it doesn't exists, returns None.
        '''
        
        for x in self.components:
            if component_type != x.__class__: continue
            return x
        return None
    
    def has_component(self, component_type) -> bool:
        '''
        Return whether the entity has a given component.
        '''
        
        for x in  self.components:
            if component_type != x.__class__: continue
            return True
        return False

    def add_component(self, component_or_type) -> Component:
        '''
        Creates a new component of given type and returns it.\n
        Raises TypeError if it already exists.
        '''
    
        if component_or_type.__class__ != type:
            if self.has_component(component_or_type.__class__):
                _raise_error(f"Entity.add_component({component_or_type}) -> Component already exists.")
            self.components.append(component_or_type)
            component_or_type.entity_id = self.id
        else:
            if self.has_component(component_or_type):
                _raise_error(f"Entity.add_component({component_or_type}) -> Component already exists.")
            self.components.append(component_or_type())
            self.components[-1].entity_id = self.id

        name = to_snake_case(self.components[-1].__class__.__name__)
        try:getattr(self, name)
        except AttributeError: setattr(self, name, self.components[-1])
        return self.components[-1]

    def remove_component(self, component_type) -> None:
        '''
        Removes a given component.\n
        Raises TypeError if it doesn't exists.
        '''
        
        for i in range(len(self.components)):
            if component_type != self.components[i].__class__: continue
            self.components.pop(i)
            return
            
        _raise_error(f"Entity.remove_component({component_type}) -> Component doesn't exists.")

def _init_utils(window: Window) -> None:
    '''
    Initialize shaders and stuff.
    '''

    global default_shaders, _phong_shader, _global_window, _active_ui_camera,\
            _default_poly_shader, _default_circle_shader, _flat_shader, _enviroment_shader

    default_shaders['mesh.vertex'] = '''
    #version 430 core
    layout(location = 0) in vec3 in_vert;
    layout(location = 1) in vec3 in_norm;

    uniform mat4 uMVP;
    uniform vec3 uposition;
    uniform vec3 urotation;
    uniform vec3 uscale;
    out vec3 normal;
    out vec3 frag_pos;

    mat4 rotationMatrix(vec3 axis, float angle) {
        axis = normalize(axis);
        float s = sin(angle);
        float c = cos(angle);
        float oc = 1.0 - c;
        
        return mat4(
        oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
        oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
        oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
        0.0,                                0.0,                                0.0,                                1.0);
    }

    vec3 rotate(vec3 v, vec3 axis, float angle) {
        mat4 m = rotationMatrix(axis, angle);
        return (m * vec4(v, 1.0)).xyz;
    }

    void main()
    {
        vec3 pos = uscale * in_vert;
        pos = rotate(pos, vec3(1, 0, 0), urotation.x);
        pos = rotate(pos, vec3(0, 1, 0), urotation.y);
        pos = rotate(pos, vec3(0, 0, 1), urotation.z);
        pos += uposition;
        
        normal = rotate(in_norm, vec3(1, 0, 0), urotation.x);
        normal = rotate(normal, vec3(0, 1, 0), urotation.y);
        normal = rotate(normal, vec3(0, 0, 1), urotation.z);
        
        frag_pos = pos;
        gl_Position = uMVP * vec4(pos, 1.0);
    }
    '''

    default_shaders['phong.fragment'] = '''
    #version 430 core
    layout(location = 0) out vec4 f_color;
    layout(std430, binding = 0) /*kaota*/ readonly buffer _
    {
        float dldata[];
    };
    layout(std430, binding = 1) /*kaota*/ readonly buffer _2
    {
        float pldata[];
    };

    uniform int udlcount;
    uniform int uplcount;

    uniform vec3 uview_pos;
    uniform vec3 ucolor;
    uniform float uambient_strength;
    uniform float udiffuse_strength;
    uniform float uspecular_strength;
    uniform float uspecular_exponent;

    const int dl_stride = 16;
    const int pl_stride = 7;
    
    in vec3 normal;
    in vec3 frag_pos;

    void main()
    {
        vec3 view_dir = normalize(frag_pos - uview_pos);
        vec3 light = vec3(0);
        for(int i = 0; i < udlcount; i ++)
        {
            float light_intensity = dldata[i * dl_stride];
            vec3 light_color = vec3(
                dldata[i * dl_stride + 1],
                dldata[i * dl_stride + 2],
                dldata[i * dl_stride + 3]
            );
            vec3 light_dir = normalize(vec3(
                dldata[i * dl_stride + 4],
                dldata[i * dl_stride + 5],
                dldata[i * dl_stride + 6]
            ));

            vec3 reflect_dir = reflect(-light_dir, normal);
            float specular = pow(max(dot(view_dir, reflect_dir), 0), uspecular_exponent);

            light += light_color * light_intensity * uambient_strength;
            light += light_color * light_intensity * udiffuse_strength * max(dot(normal, -light_dir), 0);
            light += light_color * light_intensity * uspecular_strength * specular;
        }

        for(int i = 0; i < uplcount; i ++)
        {
            float light_intensity = pldata[i * pl_stride];
            vec3 light_color = vec3(
                pldata[i * pl_stride + 1],
                pldata[i * pl_stride + 2],
                pldata[i * pl_stride + 3]
            );
            vec3 light_pos_rel = frag_pos - vec3(
                pldata[i * pl_stride + 4],
                pldata[i * pl_stride + 5],
                pldata[i * pl_stride + 6]
            );
            vec3 light_dir = normalize(light_pos_rel);
            float dist_sqr = dot(light_pos_rel, light_pos_rel);

            vec3 reflect_dir = reflect(-light_dir, normal);
            float specular = pow(max(dot(view_dir, reflect_dir), 0), uspecular_exponent);

            light += (light_color * light_intensity * uambient_strength +
                      light_color * light_intensity * udiffuse_strength * max(dot(normal, -light_dir), 0) +
                      light_color * light_intensity * uspecular_strength * specular) * (1.0 / dist_sqr);
        }

        f_color = vec4(light * ucolor, 1);
    }
    '''
    
    default_shaders['flat.fragment'] = '''
    #version 430 core
    layout(location = 0) out vec4 f_color;

    uniform vec3 ucolor;

    in vec3 normal;
    in vec3 frag_pos;

    void main()
    {
        f_color = vec4(ucolor + normal * 0.0000001, 1);
    }
    '''

    default_shaders['polygon.vertex'] = '''
    #version 430 core
    layout(location = 0) in vec2 in_vert;
    layout(location = 1) in vec4 in_col;

    uniform vec2 uposition;
    uniform float urotation;
    uniform vec2 uscale;
    uniform vec4 ucolor;
    uniform float uaspect_ratio;

    out vec4 color;
    out vec2 norm_pos;

    vec2 rotate(vec2 v, float a) {
        float s = sin(a);
        float c = cos(a);
        mat2 m = mat2(c, s, -s, c);
        return m * v;
    }

    void main()
    {
        vec2 pos = in_vert * uscale;
        pos = rotate(pos, urotation);
        pos.y *= uaspect_ratio;
        pos += uposition;
        
        color = in_col * ucolor;
        norm_pos = in_vert;
        gl_Position = vec4(pos, 0.0, 1.0);
    }
    '''

    default_shaders['polygon.fragment'] = '''
    #version 430 core
    layout(location = 0) out vec4 f_color;

    in vec4 color;    

    void main()
    {
        f_color = color;
    }
    '''

    default_shaders['polygon.circle.fragment'] = '''
    #version 430 core
    layout(location = 0) out vec4 f_color;

    in vec4 color;
    in vec2 norm_pos;

    void main()
    {
        f_color = mix(color, vec4(0), pow(dot(norm_pos, norm_pos), 75));
    }
    '''

    default_shaders['enviroment.vertex'] = '''
    #version 430 core
    layout(location = 0) in vec2 in_vert;
    layout(location = 1) in vec4 in_col;

    out vec3 norm_pos;
    out vec4 color;

    uniform mat4 uMVP;
    uniform vec3 uview_pos;
    uniform vec3 uview_dir;
    uniform float yaw;
    uniform float pitch;
    uniform float uaspect_ratio;

    const vec3 vertices[6] = {
        vec3(-1, -1, 0),
        vec3(-1, 1, 0),
        vec3(1, -1, 0),
        vec3(1, 1, 0),
        vec3(1, -1, 0),
        vec3(-1, 1, 0)
    };

    mat4 rotationMatrix(vec3 axis, float angle) {
        axis = normalize(axis);
        float s = sin(angle);
        float c = cos(angle);
        float oc = 1.0 - c;
        
        return mat4(
        oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
        oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
        oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
        0.0,                                0.0,                                0.0,                                1.0);
    }

    vec3 rotate(vec3 v, vec3 axis, float angle) {
        mat4 m = rotationMatrix(axis, angle);
        return (m * vec4(v, 1.0)).xyz;
    }

    void main()
    {
        vec3 pos = 2 * vertices[gl_VertexID] * vec3(uaspect_ratio, 1, 1);
        pos += vec3(0, 0, 1);
        pos = rotate(pos, vec3(1, 0, 0), yaw);
        pos = rotate(pos, vec3(0, 1, 0), -pitch);

        norm_pos = pos + vec3(in_vert.x) * 0.000001;
        color = in_col;
        
        pos += uview_pos;
        gl_Position = uMVP * vec4(pos, 1.0);
    }
    '''

    default_shaders['enviroment.fragment'] = '''
    #version 430 core
    layout(location = 0) out vec4 f_color;
    layout(std430, binding = 1) readonly buffer _
    {
        float dldata[];
    };

    uniform int udlcount;
    const int dl_stride = 16;

    uniform float yaw;
    uniform float pitch;
    uniform float uaspect_ratio;
    uniform vec3 sky_color;

    in vec3 norm_pos;
    in vec4 color;

    void main()
    {
        vec3 look = normalize(norm_pos);
        float sun_count = 0.0;
        for(int i = 0; i < udlcount; i ++)
        {
            if(dldata[i * dl_stride + 15] < 0.5) continue;
            sun_count ++;
        }

        vec3 final_color = vec3(0);
        for(int i = 0; i < udlcount; i ++)
        {
            if(dldata[i * dl_stride + 15] < 0.5) continue;

            vec3 sun_direction = vec3(
                dldata[i * dl_stride + 4],
                dldata[i * dl_stride + 5],
                dldata[i * dl_stride + 6]
            );

            vec3 sun_color = vec3(
                dldata[i * dl_stride + 7],
                dldata[i * dl_stride + 8],
                dldata[i * dl_stride + 9]
            );

            vec3 sun_tint = vec3(
                dldata[i * dl_stride + 10],
                dldata[i * dl_stride + 11],
                dldata[i * dl_stride + 12]
            );

            float sun_strength = dldata[i * dl_stride + 13];
            float sun_exponent = exp(dldata[i * dl_stride + 14]);

            float angle_to_sun = max(dot(-sun_direction, look), 0);
            angle_to_sun = pow(angle_to_sun, sun_exponent);
            angle_to_sun *= sun_strength;
            
            final_color += mix(sky_color, sun_color, vec3(angle_to_sun) * sun_tint * sun_count);
        }
        
        final_color /= sun_count;

        f_color = color * 0.000001 + vec4(final_color, 1.0);
    }
    '''

    _phong_shader = Shader.load_from_buffer2(default_shaders['mesh.vertex'], default_shaders['phong.fragment'])
    _flat_shader = Shader.load_from_buffer2(default_shaders['mesh.vertex'], default_shaders['flat.fragment'])
    _default_poly_shader = Shader.load_from_buffer2(default_shaders['polygon.vertex'], default_shaders['polygon.fragment'])
    _default_circle_shader = Shader.load_from_buffer2(default_shaders['polygon.vertex'], default_shaders['polygon.circle.fragment'])
    _enviroment_shader = Shader.load_from_buffer2(default_shaders['enviroment.vertex'], default_shaders['enviroment.fragment'])

    _global_window = window
    _active_ui_camera = Camera2D()


def _gen_entity_id():
    '''
    Generate a unique Entity ID.
    '''
    global _entity_id_last
    _entity_id_last += 1
    return _entity_id_last

def get_entity_from_id(entity_id: int) -> Entity | None:
    '''
    Get an entity from its ID.
    '''

    for entity in _entities:
        if entity.id != entity_id: continue
        return entity
    return None

def rotate2d(vector: vec2, angle: float) -> vec2:
    '''
    Rotate a vec2 by specified angle.
    '''
    angle = atan2(vector.y, vector.x) + angle
    return vec2(cos(angle), sin(angle)) * length(vector)

def _raise_error(error, type=TypeError):
    '''
    Helper for raising an error.
    '''
    raise type('\033[31m' + error + '\033[0m')

def to_snake_case(name: str):
    '''
    Convert Camel case to snake case.
    '''
    result = name[0].lower()
    for c in name[1:]:
        if c.isupper(): result += '_'
        result += c.lower()
    return result

_entities: list[Entity] = []
_lights: list[Entity] = []
_entity_id_last = 0
