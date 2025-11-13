# Sigma3D
* Sigma3D is a basic 3D renderer in made in Python with possible support for physics in future (hopefully).

## Installation

Open `Terminal` or `Command Prompt` in your project directory and run the following command:

```cmd
git clone https://github.com/Luludotc/Sigma3D.git ./sigma3d
```

Create a python file by the name `main.py` in the same folder as `sigma3d`


├── `sigma3d`\
│   ├── `__init__.py`\
│   ├── `LICENSE`\
│   └── `README.md`\
└── `main.py`

Paste the following code in `main.py` and play around!

```py
from sigma3d import *

win = Window(720, 480, max_fps=60) # Create a window of 720x480 pixels.

# Create a camera entity.
camera_entity = Entity() # Create an entity.
camera_entity.add_component(Transform) # Add a Transform component, for position, rotation & scale.
camera_entity.add_component(Camera)    # Camera component, to see stuff.
camera_entity.get_component(Camera).use() # Enable the camera


# Alternative way to add components.
light = Entity([
    Transform(rotation=vec3(radians(-70), 0, radians(45))), # Rotation in XYZ axis
    DirectionalLight(color=vec3(1.0, 1.0, 0.8))  # To simulate sunlight, RGB colors in [0 -> 1] range.
])

# A sphere mesh.
sphere = Entity([
    Transform(position=vec3(0, 0, 5)),
    MeshRenderer()
])

# Alternative to Entity.get_component()
sphere.mesh_renderer.mesh = Mesh.create_sphere(divisions=64) # Higher divisions = high quality sphere, takes time to load.
sphere.mesh_renderer.material = PhongMaterial(color=vec3(1)) # A material describes how a mesh should be rendered.

# Called once every frame
def my_loop():
    camera_entity.camera.control() # A first-person controller (Press Escape to toggle focus)

# Start the window loop.
win.start_loop(my_loop)
```

## Documentation

### Window
```py
constructor(width: int, height: int, title?: str, max_fps?: int) # title and max_fps is optional.
```

* Properties

    ```py
    aspect_ratio: float # The ratio of width by height of the window

    mouse_pos: vec2 # Position of the mouse pointer relative to the window.

    mouse_delta: vec2 # The difference between current and last frame's mouse position.

    render_system: RenderSystem # To draw stuff (see RenderSystem class).

    lighting_system: LightingSystem # To handle scene lighting (see LightingSystem class).

    delta_time: float # Time elapsed (in seconds) since last frame.

    time_elapsed: float # Time elapsed (in seconds) since the creation of the window.
    ```

* Methods

    ```py
    # Sets the maximum framerate.
    set_max_fps(fps: int) -> None
    
    # Sets the size of the window.
    set_size(width: int, height: int) -> None
    
    # Sets the title of the window.
    set_title(title: str) -> None
    
    # Get the size of the window.
    get_size() -> vec2
    
    # Get the title of the window.
    get_title() -> str
    
    # Get the framerate of the window.
    get_fps() -> float
    
    # Lock mouse pointer.
    lock_mouse() -> None
    
    # Unlock (release) mouse pointer.
    unlock_mouse() -> None
    
    # Returns whether the mouse pointer is locked or not.
    is_mouse_locked() -> bool
    
    # Returns whether the given key is being pressed.
    is_key_down(key) -> bool
    
    # Returns whether the given key is pressed in the current frame.
    is_key_pressed(key) -> bool
    
    # Returns whether the given key is released in the current frame.
    is_key_released(key) -> bool
    
    # Return whether the given mouse button is being pressed.
    is_mouse_down(button) -> bool
    
    # Return whether the given mouse button is pressed in the current frame.
    is_mouse_pressed(button) -> bool
    
    # Return whether the given mouse button is released in the current frame.
    is_mouse_released(button) -> bool
    
    # Start the window loop, calls `callback` function once every frame, calls `draw_ui` function after 3D rendering for drawing UI objects.
    start_loop(callback?: callable, draw_ui?: callable) -> None
    ```

### Component
The base class for all components

* Properties
    ```py
    entity_id: int # The id of the entity this component belongs to.
    ```

### Transform (Component)
Stores position, rotation and scale in 3D space.

```py
constructor(position?: vec3, rotation?: vec3, scale?: vec3)
```

* Properties

    ```py
    position: vec3 # Position in 3D space

    rotation: vec3 # Rotation in X Y Z axes

    scale: vec3 # Scale in X Y Z axes
    ```
    
* Methods

    ```py
    # Get the forward direction.
    def get_forward() -> vec3:

    # Get the right direction.
    def get_right() -> vec3:

    # Get the up direction.
    def get_up() -> vec3:

    # Get the Yaw (rotation in y-axis) and Pitch (rotation in x-axis).
    def get_yaw_pitch() -> vec2:
    ```

### Camera (Component)
```py
constructor(FOV?: float, near?: flaot, far?: float, up?: vec3)
```

* Properties

    ```py
    FOV: float # Field of view.

    near: float # The near clipping plane (closest distance camera can see.)

    far: float # The far clipping plane (furthest distance camera can see.)

    up: vec3 # The "up" direction of the camera.
    ```

* Methods

    ```py
    # Get the viewport-projection matrix of the camera.
    get_vp_matrix() -> mat4

    # Activate the camera
    use() -> None

    # (static method) Deactivate the current active camera.
    Camera.use_none() -> None

    # A basic first-person controller, Intended for debugging.
    control(active_key?, move_keys?: tuple, movement_speed?: float, camera_sensitivity?: float) -> None

        active_key = K_ESCAPE # The key to lock/unlock mouse.

        move_keys = (K_w, K_a, K_s, K_d, K_e, K_q) # The key used to move the camera.

        movement_speed = 10 # The movement speed.

        camera_sensitivity = 0.002 # Sensitivity of the camera (can also be vec2(0.002, -0.002) to invert y-axis).
    ```

### Light (Component)
Base class for all kind of Lights.

```py
constructor(color?: vec3, intensity?: float)
```

* Properties

    ```py
    intensity: float # The intensity of the light.

    color: vec3 # The color of the light.

    active: bool # Whether the light is turned on or off.
    ```

### DirectionalLight (Component)
```py
# Inherits from Light.
constructor(color?: vec3, intensity?: float, sun_color?: vec3, sun_strength?: float, sun_exponent?: float, sun_tint?: float, is_sun?: bool)
```

* Properties

    ```py
    intensity: float # The intensity of the light.

    color: vec3 # The color of the light.

    active: bool # Whether the light is turned on or off.

    # NOTE: These "sun" values are used in while rendering Enviroment (see Enviroment class), can be disabled by DirectionalLight.is_sun = False.
    sun_color: vec3 # Color of the sun.

    sun_strength: float # Intensity of the sun.

    sun_exponent: float # Inverse size of the sun.

    sun_tint: float # Tint color for the sun, can be used to make outline around the sun.

    is_sun: bool # Whether an Enviroment should render the DirectionalLight as a sun.
    ```

### PointLight (Component)
```py
# Inherits from Light.
constructor(color?: vec3, intensity?: float)
```

* Properties

    ```py
    intensity: float # The intensity of the light.

    color: vec3 # The color of the light.

    active: bool # Whether the light is turned on or off.
    ```

### Enviroment (Component)
```py
constructor(mode?: Enviroment.Sky | Enviroment.Solid, sky_color?: vec3, solid_color?: vec3)
```

* Properties

    ```py
    polygon: Polygon2D # The Polygon where Enviroment is rendered to.
    
    sky_color: vec3 # The color of the sky, this color is used if mode is Enviroment.Sky
    
    solid_color: vec3 # The color of the solid background, this color is used if mode is Enviroment.Solid
    ```

* Methods

    ```py
    # Set the enviroment mode.
    set_mode(mode: Enviroment.Sky | Enviroment.Solid) -> None

    # Activate the Enviroment.
    use() -> None

    # (static method) Deactivate the currently active Enviroment.
    Enviroment.use_none() -> None

    # Render the Enviroment.
    # NOTE: The active Enviroment is rendered just before rendering The scene, there is no need to call this function unless you want to.
    render() -> None
    ```


### Material
Base class for all Materials.
A Material defines color, lighting and shading of a Mesh.

```py
constructor(shader:? Shader, enable_lighting?: bool)
```

* Properties

    ```py
    shader: Shader # The shader of the material.
    enable_lighting: bool # Whether shader of this material should input Light buffers.
    ```

* Methods

    ```py
    # Bind data to the shader.
    bind() -> None
    ```


### FlatMaterial (Material)
For flat shading.

```py
constructor(color?: vec3)
```

* Properties

    ```py
    color: vec3 # Albedo of the material.
    ```

### PhongMaterial (Material)
For phong shading.

```py
constructor(color?: vec3, ambient_strength?: float, diffuse_strength?: float, specular_strength?: float, specular_exponent?: float)
```

* Properties

    ```py
    color: vec3 # Albedo of the material.
    ambient_strength: float # Strength of ambience lighting.
    diffuse_strength: float # Strength of diffuse lighting.
    specular_strength: float # Strength of specular lighting.
    specular_exponent: float # Exponent of specular lighting.
    ```

### System
A base class for all systems.\
NOTE: A system class is used by the engine, It should not be tweaked with directly.

### RenderSystem (System)
* Properties

    ```py
    clear_color: vec4 # When clearing the screen, this color is used.
    ```

* Methods

    ```py
    # Clear the screen.
    clear() -> None

    # Draws the active enviroment.
    draw_enviroment() -> None

    # Draw a mesh.
    draw_mesh(transform: Transform, mesh: Mesh, material: Material) -> None

    # Renders the scene.
    update() -> None

    # Enable/Disable face culling.
    set_cull_face(flag: bool) -> None

    # Enable/Disable depth test.
    set_depth_test(flag: bool) -> None

    # Enable/Disable alpha blending.
    set_alpha_blending(flag: bool) -> None
    ```

### LightingSystem (System)
* Properties

    ```py
    dlsb: _DirectionalLightShaderBuffer # DirectionalLight Shader Buffer
    plsb: _PointLightShaderBuffer # PointLight Shader Buffer
    ```

* Methods

    ```py
    # Clear the screen.

    # Update the lighting.
    update() -> None

    # Bind lighting buffer(s) with a shader.
    bind_with_shader(shader: Shader)
    ```

### Vertex
```py
constructor(position: vec3, normal?: vec3)
```

* Propeties

    ```py
       position: vec3 # The position of the vertex.

       normal: vec3 # The normal direction of the triangle this vertex belongs to.
    ```

### Shader
* Propeties

    ```py
    program ## OpenGL shader program.
    ```

* Methods

    ```py
    # (static method) load a shader program from a file.
    Shader.load(path: str) -> Shader

    # (static method) load a shader program from a string buffer.
    Shader.load_from_buffer(buffer: str) -> Shader

    # (static method) load a shader program from a vertex shader and a fragment shader.
    Shader.load_from_buffer2(vertex: str, fragment: str) -> Shader

    # Set the value of a shader uniform. (vectors and matrices must be provided as a tuple or list)
    set_uniform(name, value) -> None

    # Get the value of a shader uniform.
    get_uniform(name) -> any
    ```

### Mesh
```py
constructor(vertices?: list[Vertex], preserve_normals?: bool)
```

* Properties

    ```py
    vertices: list[Vertex] # Vertices of the mesh.
    ```

* Methods

    ```py
    # Recalculate the normals for each triangle.
    regenerate_normals() -> None

    # Update the shader (should be called after modifying the mesh).
    refresh() -> None

    # Bind the mesh with the shader to ready it for rendering.
    bind(shader: Shader) -> None

    # (static method) Takes a list of vertices, copies it again with inverted normals to create a "back side".
    Mesh.add_double_sided(verts: list[Vertex]) -> list[Vertex]

    # (static method) Creates a unit cube.
    Mesh.create_box(double_sided?: bool) -> Mesh

    # (static method) Creates a unit quad.
    Mesh.create_quad(double_sided?: bool) -> Mesh

    # (static method) Creates a unit disc (circle).
    Mesh.create_disc(double_sided?: bool) -> Mesh

    # (static method) Creates a unit sphere.
    Mesh.create_sphere(double_sided?: bool) -> Mesh
    ```




### Vertex2D
```py
constructor(position: vec2, color?: vec4)
```

* Propeties

    ```py
       position: vec2 # The position of the vertex.

       color: vec4 # Color of the vertex.
    ```

### Polygon2D
```py
constructor(position?: vec2, rotation?: float, scale?: vec2, color?: vec4, vertices: list[Vertex2D], shader?: Shader)
```

* Properties

    ```py
    position: vec2 # Position of the polygon.
    rotation: float # rotation of the polygon.
    scale: vec2 # Scale of the polygon.
    color: vec4 # Color of the polygon.
    vertices: list[Vertex2D] # List of vertices of the polygon.
    shader: Shader # Shader program to use for rendering of the polygon.
    ```

* Methods

    ```py
    # Update the shader (should be called after modifying the polygon).
    refresh() -> None

    # Render the polygon.
    render() -> None

    # (static method) Create a rectangle polygon.
    Polygon2D.create_rectangle(position?: vec2, scale?: vec2, rotation?: float, color?: vec4) -> Polygon2D

    # (static method) Create a circle polygon.
    Polygon2D.create_circle(position?: vec2, radius?: float, color?: vec4) -> Polygon2D
    ```

### Camera2D
```py
constructor(position?: vec2, zoom?: float, rotation?: float)
```

* Properties

    ```py
    position: vec2 # Position of the Camera2D.
    zoom: float # Zoom-in Scale.
    rotation: float # Rotation of the camera.
    ```

* Members

    ```py
    # Activate the Camera2D
    use() -> None

    # (static method) Deactivate the currently active Camera2D
    Camera2D.use_none() -> None
    ```



### Physics
Physics engine is not completed yet it will will take quite some time.\
I won't recommend using the incomplete physics engine.

### Entity
```py
constructor(components?: list[Component])
```

* Properties

    ```py
    id: int # Unique integer to identify an entity.
    components: list[Component] # List of Components of the entity.
    ```

* Methods

    ```py
    # Returns a given component. If it doesn't exists, returns None.
    get_component(component_type: type[Component]) -> Component | None

    # Return whether the entity has a given component.
    has_component(component_type: type[Component]) -> bool

    # Creates a new component of given type and returns it. Raises TypeError if it already exists.
    add_component(component_or_type: Component | type[Component]) -> Component

    # Removes a given component. Raises TypeError if it doesn't exists.
    remove_component(component_type: type[Component]) -> None
    ```

## Miscellaneous stuff

```py
# Rotate a vec2 by specified angle.
rotate2d(vector: vec2, angle: float) -> vec2

# Get an entity by it's ID, returns None if it doesn't exist.
get_entity_from_id(entity_id: int) -> Entity | None

# Convert Camel-Case to Snake-Case
to_snake_case(name: str) -> str
```
