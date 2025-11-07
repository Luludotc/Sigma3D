# Sigma3D
* Sigma3D is a basic 3D renderer in made in Python with possible support for physics (hopefully).

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

```py
from sigma3d import *

win = Window(700, 500, max_fps = 60) # Create the 700x500 window.

cam = Camera() # Create a camera
cam.use() # Activate the camera

light = DirectionalLight() # Create a directional light
light.intensity = 1.5

cube = Mesh.create_box() # Create a box
cube.position = vec3(0, 0, 2) # Set the position
cube.color = vec3(1, 0.1, 0.1) # Red color

def loop():
    cam.control() # First person controller (Press Escape to toggle)

win.start_loop(loop) # Start the loop
```

## Documentation
### Window
```py
constructor(width: int, height: int, title: str, max_fps?: int) # max_fps is optional.
```

* Properties

    ```py
    Window.aspect_ratio: float # The ratio of width by height of the window

    Window.mouse_pos: vec2 # Position of the mouse pointer relative to the window.

    Window.mouse_delta: vec2 # The difference between current and last frame's mouse position.

    Window.draw: Drawer # Can be used to draw things (see Drawer class).

    Window.delta_time: float # Time elapsed (in seconds) since last frame.

    Window.time_elapsed: float # Time elapsed (in seconds) since the creation of the window.
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
    get_size() -> tuple[int, int]
    
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
    
    # Start the loop, calls the given callback function once every frame.
    start_loop(callback: callable) -> None
    ```

### Camera
```py
constructor(position?: vec3, yaw?: float, pitch?: float, FOV?: float, near?: flaot, far?: float, up?: vec3)
```

* Properties

    ```py
    position: vec3 # The position of the Camera.

    yaw: float # The yaw (rotation in Y-axis).

    pitch: float # The pitch (rotation in X-axis).

    FOV: float # Field of view.

    near: float # The near clipping plane (closest distance camera can see.)

    far: float # The far clipping plane (furthest distance camera can see.)

    up: vec3 # The "up" direction of the camera.
    ```

* Methods

    ```py
    # Get the direction camera is looking at.
    get_forward() -> vec3

    # Get the right direction of the camera.
    get_right() -> vec3

    # Get the up direction of the camera.
    # NOTE: This is different from Camera.up
    get_up() -> vec3

    # Get the viewport-projection matrix of the camera.
    get_vp_matrix() -> mat4

    # Activate the camera
    use() -> None

    # (static method) Deactivate the current active camera.
    Camera.use_none() -> None

    # A basic first-person controller
    control(active_key?, move_keys?: tuple, movement_speed?: float, camera_sensitivity?: float) -> None

        active_key = K_ESCAPE # The key to lock/unlock mouse.

        move_keys = (K_w, K_a, K_s, K_d, K_e, K_q) # The key used to move the camera.

        movement_speed = 10 # The movement speed.

        camera_sensitivity = 0.002 # Sensitivity of the camera (can also be vec2(0.002, -0.002) to invert y-axis).
    ```

### Drawer
* Properties

    ```py
    clear_color: vec4 # When clearing the screen, the color is used.
    ```

* Methods

    ```py
    # Clear the screen.
    clear() -> None

    # Draw a mesh. (Meshes are drawn at the end of the frame, there is no reason to call this method explicitly)
    draw_mesh(mesh: Mesh) -> None

    # Enable/Disable face culling.
    set_cull_face(flag: bool) -> None

    # Enable/Disable depth test.
    set_depth_test(flag: bool) -> None
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
    load_from_buffer(buffer: str) -> Shader
    
    # Get value of a shader uniform.
    get_uniform(name: str)

    # Set value of a shader uniform.
    set_uniform(name: str, value)
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

### Mesh
```py
constructor(position?: vec3, rotation?: vec3, scale?: vec3, color?: vec4, vertices?: list[Vertex], shader?: Shader, preserve_normals?: bool)
```

* Properties

    ```py
    position: vec3 # Position of the mesh.

    rotation: vec3 # Rotation of the mesh.

    scale: vec3 # Scale of the mesh.

    color: vec4 # Color of the mesh.

    vertices: list[Vertex] # Vertices of the mesh.

    shader: Shader # Shader program to use for rendering of the mesh.
    ```

* Methods

    ```py
    # Recalculate the normals for each triangle.
    regenerate_normals() -> None

    # Update the shader (should be called after modifying the mesh).
    refresh() -> None

    # Delete the mesh.
    delete() -> None

    # Renders the mesh (on lighting!).
    # NOTE: Don't call this function by yourself, unless you know what you're doing.
    render() -> None

    # (static method) Takes a list of vertices, copies it again with inverted normals to create a "back side".
    Mesh.add_double_sided(verts: list[Vertex])

    # (static method) Creates a unit cube.
    Mesh.create_box(double_sided?: bool)

    # (static method) Creates a unit quad.
    Mesh.create_quad(double_sided?: bool)

    # (static method) Creates a unit disc (circle).
    Mesh.create_disc(double_sided?: bool)

    # (static method) Creates a unit sphere.
    Mesh.create_sphere(double_sided?: bool)
    ```

### Light
NOTE: This is a abstract class. Don't instantiate it.

```py
constructor(color: vec3)
```

* Properties

    ```py
        intensity: float # The intensity of the light.

        ambient_strength: float # The ambient light strength.

        diffuse_strength: float # The diffuse light strength.

        specular_strength: float # The specular light strength.

        specular_exp: float # The specular exponent.

        color: vec3 # The color of the light.
    ```

* Methods

    ```py
    # Activate the light.
    activate() -> None

    # Deactivate the light.
    deactivate() -> None
    ```

### DirectionalLight
```py
# Inherits from Light.
constructor(color: vec3, direction?: vec3)
```

* Properties (+ all from light)

    ```py
    direction: vec3 # The direction of the light.
    ```

### PointLight
```py
# Inherits from Light.
constructor(color: vec3, position?: vec3)
```

* Properties (+ all from Light)

    ```py
    position: vec3 # The position of the light.
    ```

### Physics.Point
```py
constructor(position: vec3)
```

* Properties

    ```py
    position: vec3 # Position of the point.

    acceleration: vec3 # Acceleration of the point.
    ```

* Methods

    ```py
    # Calculate position for the next frame.
    step(delta_time?: float) -> None

    # Set the position (resets velocity!).
    set_position(position: vec3) -> None

    # Set the velocity.
    set_velocity(velocity: vec3) -> None

    # Get the velocity.
    get_velocity() -> vec3

    # Set the acceleration.
    set_acceleration(acceleration: vec3) -> None

    # Add acceleration.
    accelerate(acceleration: vec3) -> None
    ```

### Physics.Sphere
```py
# Inherits from Physics.Point.
constructor(position: vec3, radius: float)
```

* Properties (+ all from Physics.Point)

    ```py
    radius: vec3 # Radius of the sphere
    ```

### Physics

* Methods

    ```py
    # Returns whether the two given sphere collide or not.
    check_collision_sphere(sphere_1: Sphere, sphere_2: Sphere) -> bool

    # Resolves the collision between two spheres.
    resolve_collision_sphere(sphere_1: Sphere, sphere_2: Sphere) -> None
    ```

