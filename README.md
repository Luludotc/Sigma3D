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
cube.position = vec3(0, 0, 3) # Set the position
cube.color = vec3(1, 0.1, 0.1) # Red color

def loop():
    cam.control() # First person controller (Press Escape to toggle)

win.start_loop(loop) # Start the loop
```

## Documentation
To be made...


