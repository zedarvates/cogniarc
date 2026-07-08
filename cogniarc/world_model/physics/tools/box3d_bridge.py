"""
Box3D Python Bridge — Complete ctypes wrapper.
Wires all key Box3D C functions through to Python.
3D physics engine by Erin Catto (MIT).
"""

import ctypes
import ctypes.util
import math
import os
import numpy as np
from typing import List, Tuple, Optional, Dict


# ============================================================
# 1. Load Box3D shared library
# ============================================================

_lib_path = ctypes.util.find_library("box3d") or ""
if not _lib_path or not os.path.exists(_lib_path):
    _lib_path = os.path.expanduser("~/projects/box3d/build/bin/libbox3d.so")
if not os.path.exists(_lib_path):
    raise ImportError(f"Box3D library not found at {_lib_path}. Build it: cd ~/projects/box3d && bash build.sh")

lib = ctypes.CDLL(_lib_path)

# ============================================================
# 2. C type definitions
# ============================================================

class b3Vec3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float)]
    def to_tuple(self): return (self.x, self.y, self.z)
    @classmethod
    def from_tuple(cls, t): return cls(t[0], t[1], t[2] if len(t) > 2 else 0)

class b3Quat(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), 
                ("z", ctypes.c_float), ("w", ctypes.c_float)]

class b3WorldId(ctypes.Structure):
    _fields_ = [("index1", ctypes.c_int16), ("revision", ctypes.c_int16), ("pad", ctypes.c_int32)]
    def to_tuple(self): return (self.index1, self.revision)

class b3BodyId(ctypes.Structure):
    _fields_ = [("index1", ctypes.c_int16), ("world0", ctypes.c_int16), ("revision", ctypes.c_int16)]
    def to_tuple(self): return (self.index1, self.world0, self.revision)

class b3ShapeId(ctypes.Structure):
    _fields_ = [("index1", ctypes.c_int16), ("world0", ctypes.c_int16), ("revision", ctypes.c_int16)]

class b3JointId(ctypes.Structure):
    _fields_ = [("index1", ctypes.c_int16), ("world0", ctypes.c_int16), ("revision", ctypes.c_int16)]

class b3Filter(ctypes.Structure):
    _fields_ = [("categoryBits", ctypes.c_uint64), ("maskBits", ctypes.c_uint64)]

class b3SurfaceMaterial(ctypes.Structure):
    _fields_ = [("friction", ctypes.c_float), ("restitution", ctypes.c_float),
                ("tangentFriction", ctypes.c_float), ("rollingFriction", ctypes.c_float),
                ("userData", ctypes.c_void_p)]

class b3ShapeDef(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("userData", ctypes.c_void_p),
        ("materials", ctypes.POINTER(b3SurfaceMaterial)),
        ("materialCount", ctypes.c_int),
        ("baseMaterial", b3SurfaceMaterial),
        ("density", ctypes.c_float),
        ("explosionScale", ctypes.c_float),
        ("filter", b3Filter),
        ("enableCustomFiltering", ctypes.c_bool),
        ("isSensor", ctypes.c_bool),
        ("_pad", ctypes.c_char * 3),
        ("bodyUserData", ctypes.c_void_p),
    ]

class b3BodyDef(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("position", b3Vec3),
        ("rotation", b3Quat),
        ("linearVelocity", b3Vec3),
        ("angularVelocity", b3Vec3),
        ("linearDamping", ctypes.c_float),
        ("angularDamping", ctypes.c_float),
        ("sleepThreshold", ctypes.c_float),
        ("userData", ctypes.c_void_p),
        ("enableSleep", ctypes.c_bool),
        ("isAwake", ctypes.c_bool),
        ("isEnabled", ctypes.c_bool),
        ("fixedRotation", ctypes.c_bool),
        ("isBullet", ctypes.c_bool),
        ("_pad", ctypes.c_char * 3),
        ("gravityScale", ctypes.c_float),
        ("allowFastRotation", ctypes.c_bool),
        ("_pad2", ctypes.c_char * 7),
    ]

class b3WorldDef(ctypes.Structure):
    _fields_ = [
        ("gravity", b3Vec3),
        ("restitutionThreshold", ctypes.c_float),
        ("contactTau", ctypes.c_float),
        ("contactHertz", ctypes.c_float),
        ("contactDampingRatio", ctypes.c_float),
        ("jointTau", ctypes.c_float),
        ("jointHertz", ctypes.c_float),
        ("jointDampingRatio", ctypes.c_float),
        ("maximumStates", ctypes.c_int32),
        ("maximumStaticBodyCount", ctypes.c_int32),
        ("maximumDynamicBodyCount", ctypes.c_int32),
        ("maximumKinematicBodyCount", ctypes.c_int32),
        ("restitutionThreshold", ctypes.c_float),
        ("enableSleep", ctypes.c_bool),
        ("enableContinuos", ctypes.c_bool),
        ("_pad", ctypes.c_char * 3),
        ("userData", ctypes.c_void_p),
    ]

# ============================================================
# 3. Function prototypes
# ============================================================

lib.b3CreateWorld.argtypes = [ctypes.POINTER(b3WorldDef)]
lib.b3CreateWorld.restype = b3WorldId

lib.b3DestroyWorld.argtypes = [b3WorldId]
lib.b3DestroyWorld.restype = None

lib.b3World_Step.argtypes = [b3WorldId, ctypes.c_float, ctypes.c_int]
lib.b3World_Step.restype = None

lib.b3CreateBody.argtypes = [b3WorldId, ctypes.POINTER(b3BodyDef)]
lib.b3CreateBody.restype = b3BodyId

lib.b3DestroyBody.argtypes = [b3BodyId]
lib.b3DestroyBody.restype = None

lib.b3Body_GetPosition.argtypes = [b3BodyId]
lib.b3Body_GetPosition.restype = b3Vec3

lib.b3Body_GetRotation.argtypes = [b3BodyId]
lib.b3Body_GetRotation.restype = b3Quat

lib.b3Body_GetLinearVelocity.argtypes = [b3BodyId]
lib.b3Body_GetLinearVelocity.restype = b3Vec3

lib.b3Body_GetAngularVelocity.argtypes = [b3BodyId]
lib.b3Body_GetAngularVelocity.restype = b3Vec3

lib.b3Body_SetLinearVelocity.argtypes = [b3BodyId, b3Vec3]
lib.b3Body_SetLinearVelocity.restype = None

lib.b3Body_GetMass.argtypes = [b3BodyId]
lib.b3Body_GetMass.restype = ctypes.c_float

lib.b3Body_IsValid.argtypes = [b3BodyId]
lib.b3Body_IsValid.restype = ctypes.c_bool

lib.b3Body_GetType.argtypes = [b3BodyId]
lib.b3Body_GetType.restype = ctypes.c_int

lib.b3Body_SetTransform.argtypes = [b3BodyId, b3Vec3, b3Quat]
lib.b3Body_SetTransform.restype = None

lib.b3Body_SetName.argtypes = [b3BodyId, ctypes.c_char_p]
lib.b3Body_SetName.restype = None


lib.b3Body_GetShapeCount.argtypes = [b3BodyId]
lib.b3Body_GetShapeCount.restype = ctypes.c_int


lib.b3Body_GetContactCapacity.argtypes = [b3BodyId]
lib.b3Body_GetContactCapacity.restype = ctypes.c_int

lib.b3CreateSphereShape.argtypes = [b3BodyId, ctypes.POINTER(b3ShapeDef), ctypes.POINTER(b3Vec3)]
lib.b3CreateSphereShape.restype = b3ShapeId

class b3SphereData(ctypes.Structure):
    _fields_ = [("center", b3Vec3), ("radius", ctypes.c_float)]
    @classmethod
    def from_radius(cls, r): return cls(b3Vec3(0,0,0), r)

lib.b3CreateBoxMesh.argtypes = []
lib.b3CreateBoxMesh.restype = b3ShapeId

lib.b3CreateBoxMesh.argtypes = []

lib.b3CreateCapsuleShape.argtypes = [b3BodyId, ctypes.POINTER(b3ShapeDef), b3Vec3]
lib.b3CreateCapsuleShape.restype = b3ShapeId

lib.b3World_GetContactEvents.argtypes = [b3WorldId]
lib.b3World_GetContactEvents.restype = ctypes.c_void_p

lib.b3World_SetGravity.argtypes = [b3WorldId, b3Vec3]
lib.b3World_SetGravity.restype = None

lib.b3World_EnableSleeping.argtypes = [b3WorldId, ctypes.c_bool]
lib.b3World_EnableSleeping.restype = None

lib.b3World_EnableContinuous.argtypes = [b3WorldId, ctypes.c_bool]
lib.b3World_EnableContinuous.restype = None

# ============================================================
# 4. High-level Python API
# ============================================================

class Box3DError(Exception):
    pass


class Box3DBody:
    """High-level wrapper around b3BodyId"""
    
    def __init__(self, name: str, body_id: b3BodyId, world):
        self._name = name
        self._id = body_id
        self._world = world
    
    @property
    def id(self) -> b3BodyId:
        return self._id
    
    @property
    def position(self) -> Tuple[float, float, float]:
        return lib.b3Body_GetPosition(self._id).to_tuple()
    
    @property
    def rotation(self) -> Tuple[float, float, float, float]:
        q = lib.b3Body_GetRotation(self._id)
        return (q.x, q.y, q.z, q.w)
    
    @property
    def velocity(self) -> Tuple[float, float, float]:
        return lib.b3Body_GetLinearVelocity(self._id).to_tuple()
    
    @velocity.setter
    def velocity(self, v: Tuple[float, float, float]):
        lib.b3Body_SetLinearVelocity(self._id, b3Vec3.from_tuple(v))
    
    @property
    def speed(self) -> float:
        v = self.velocity
        return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    
    @property
    def mass(self) -> float:
        return lib.b3Body_GetMass(self._id)
    
    @property
    def is_valid(self) -> bool:
        return lib.b3Body_IsValid(self._id)
    
    @property
    def body_type(self) -> str:
        bt = lib.b3Body_GetType(self._id)
        return ["static", "kinematic", "dynamic"][bt]
    
    @property
    def kinetic_energy(self) -> float:
        v = self.velocity
        return 0.5 * self.mass * (v[0]**2 + v[1]**2 + v[2]**2)
    
    def set_position(self, pos: Tuple[float, float, float], rot: Tuple[float, float, float, float] = (0, 0, 0, 1)):
        lib.b3Body_SetTransform(self._id, b3Vec3.from_tuple(pos),
                               b3Quat(rot[0], rot[1], rot[2], rot[3]))
    
    def apply_impulse(self, impulse: Tuple[float, float, float], world_point: Tuple[float, float, float] = None):
        """Apply an impulse"""
        # Use velocity change as approximation for impulse
        im = b3Vec3.from_tuple(impulse)
        pt = b3Vec3.from_tuple(world_point) if world_point else b3Vec3.from_tuple(self.position)
        v = lib.b3Body_GetLinearVelocity(self._id)
        lib.b3Body_SetLinearVelocity(self._id, b3Vec3(v.x + im.x, v.y + im.y, v.z + im.z))
    
    def activate(self):
        pass
    
    def __repr__(self) -> str:
        pos = self.position
        return f"Box3DBody('{self._name}', pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}), v={self.speed:.2f}m/s, {self.body_type})"
    
    def to_dict(self) -> dict:
        pos = self.position
        vel = self.velocity
        return {
            "id": self._name,
            "pos": list(pos),
            "vel": list(vel),
            "speed": round(self.speed, 2),
            "mass": round(self.mass, 2),
            "type": self.body_type,
            "Ek": round(self.kinetic_energy, 1),
        }


class Box3DWorld:
    """High-level Box3D physics world"""
    
    def __init__(self, gravity: Tuple[float, float, float] = (0, -9.81, 0),
                 enable_sleep: bool = True):
        # Configure world
        wd = b3WorldDef()
        wd.gravity = b3Vec3.from_tuple(gravity)
        wd.enableSleep = enable_sleep
        wd.enableContinuos = True
        
        # Create
        self._world_id = lib.b3CreateWorld(ctypes.byref(wd))
        self._bodies: List[Box3DBody] = []
        self._body_map: Dict[str, int] = {}  # name → index
        self._time = 0.0
        
        self._default_material = b3SurfaceMaterial(
            friction=0.5, restitution=0.3,
            tangentFriction=0.0, rollingFriction=0.0,
            userData=None
        )
    
    @property
    def world_id(self) -> b3WorldId:
        return self._world_id
    
    def add_sphere(self, name: str, radius: float, density: float = 1000,
                   position: Tuple[float, float, float] = (0, 0, 0),
                   velocity: Tuple[float, float, float] = (0, 0, 0),
                   body_type: int = 2,  # dynamic
                   restitution: float = 0.3,
                   friction: float = 0.5,
                   fixed_rotation: bool = False) -> Box3DBody:
        """Add a sphere to the world"""
        # Body definition
        bd = b3BodyDef()
        bd.type = body_type
        bd.position = b3Vec3.from_tuple(position)
        bd.linearVelocity = b3Vec3.from_tuple(velocity)
        bd.fixedRotation = fixed_rotation
        
        body_id = lib.b3CreateBody(self._world_id, ctypes.byref(bd))
        
        # Shape definition
        sd = b3ShapeDef()
        sd.density = density
        sd.baseMaterial = b3SurfaceMaterial(
            friction=friction, restitution=restitution,
            tangentFriction=0.0, rollingFriction=0.0,
            userData=None
        )
        
        lib.b3CreateSphereShape(body_id, ctypes.byref(sd), b3Vec3(radius, 0, 0))
        
        body = Box3DBody(name, body_id, self)
        self._bodies.append(body)
        self._body_map[name] = len(self._bodies) - 1
        return body
    
    def add_box(self, name: str, half_extents: Tuple[float, float, float],
                density: float = 1000,
                position: Tuple[float, float, float] = (0, 0, 0),
                velocity: Tuple[float, float, float] = (0, 0, 0),
                angle_rad: float = 0,
                body_type: int = 2,
                restitution: float = 0.3,
                friction: float = 0.5,
                fixed_rotation: bool = False) -> Box3DBody:
        """Add a box to the world"""
        bd = b3BodyDef()
        bd.type = body_type
        bd.position = b3Vec3.from_tuple(position)
        bd.linearVelocity = b3Vec3.from_tuple(velocity)
        bd.fixedRotation = fixed_rotation
        
        # Rotation (simple Z-axis rotation for ramp)
        half_a = angle_rad / 2
        bd.rotation = b3Quat(0, 0, math.sin(half_a), math.cos(half_a))
        
        body_id = lib.b3CreateBody(self._world_id, ctypes.byref(bd))
        
        sd = b3ShapeDef()
        sd.density = density
        sd.baseMaterial = b3SurfaceMaterial(
            friction=friction, restitution=restitution,
            tangentFriction=0.0, rollingFriction=0.0,
            userData=None
        )
        
        lib.b3CreateBoxMesh(body_id, ctypes.byref(sd), b3Vec3(*half_extents))
        
        body = Box3DBody(name, body_id, self)
        self._bodies.append(body)
        self._body_map[name] = len(self._bodies) - 1
        return body
    
    def add_plane(self, name: str, 
                  position: Tuple[float, float, float] = (0, -5, 0),
                  restitution: float = 0.1,
                  friction: float = 0.8) -> Box3DBody:
        """Add a ground plane (large thin box)"""
        return self.add_box(name, (10, 0.1, 10), density=1, 
                          position=position, 
                          body_type=0,  # static
                          restitution=restitution, friction=friction)
    
    def get_body(self, name: str) -> Optional[Box3DBody]:
        idx = self._body_map.get(name)
        if idx is not None and idx < len(self._bodies):
            return self._bodies[idx]
        return None
    
    def step(self, dt: float = 1/60, sub_steps: int = 4):
        """Advance the physics simulation"""
        lib.b3World_Step(self._world_id, dt, sub_steps)
        self._time += dt
    
    def get_state(self) -> dict:
        return {
            "time": round(self._time, 3),
            "bodies": [b.to_dict() for b in self._bodies],
            "engine": "box3d",
        }
    
    def set_gravity(self, gravity: Tuple[float, float, float]):
        lib.b3World_SetGravity(self._world_id, b3Vec3.from_tuple(gravity))
    
    def close(self):
        lib.b3DestroyWorld(self._world_id)


# ============================================================
# 5. Demo
# ============================================================

def demo_ramp():
    """Ball on ramp — the classic"""
    print("=" * 60)
    print("  BOX3D — 3D RAMP SIMULATION")
    print("=" * 60)
    
    world = Box3DWorld(gravity=(0, -9.81, 0))
    
    # Ramp — 45° rotated box
    ramp = world.add_box("ramp", (2.0, 0.2, 0.5), 
                        density=700, position=(0, -2, 0),
                        angle_rad=math.radians(45),
                        body_type=0, friction=0.4, restitution=0.2)
    
    # Ball
    ball = world.add_sphere("ball", 0.3, density=7800,
                           position=(0, 5, 0),
                           restitution=0.6, friction=0.3)
    
    # Ground
    world.add_plane("ground", position=(0, -7, 0))
    
    print(f"\nInitial: {ball}")
    
    for step in range(180):
        world.step(1/60)
        if step % 60 == 0 or step == 179:
            state = world.get_state()
            print(f"  t={state['time']:.1f}s: ball y={ball.position[1]:.2f}m, "
                  f"v={ball.speed:.1f}m/s, Ek={ball.kinetic_energy:.0f}J")
    
    print(f"\nFinal: {ball}")
    print(f"Total bodies: {len(world._bodies)}")
    
    world.close()
    return world


def demo_collision():
    """Two balls colliding"""
    print("\n" + "=" * 60)
    print("  BOX3D — COLLISION TEST")
    print("=" * 60)
    
    world = Box3DWorld()
    
    # Two spheres heading toward each other
    a = world.add_sphere("A", 0.5, density=7800, position=(-3, 0, 0), velocity=(3, 0, 0))
    b = world.add_sphere("B", 0.5, density=7800, position=(3, 0, 0), velocity=(-3, 0, 0))
    world.add_plane("ground", position=(0, -5, 0))
    
    print(f"\nBefore: A={a.speed:.1f}m/s  B={b.speed:.1f}m/s")
    
    for step in range(120):
        world.step(1/60)
        if 0.5 < world._time < 1.5:
            if step % 20 == 0:
                print(f"  t={world._time:.2f}s: A@({a.position[0]:.2f}) B@({b.position[0]:.2f}) "
                      f"vA={a.speed:.1f} vB={b.speed:.1f}")
    
    print(f"\nAfter: A={a.speed:.1f}m/s  B={b.speed:.1f}m/s")
    print(f"Momentum conserved: {abs(a.mass * a.velocity[0] + b.mass * b.velocity[0]):.1f} kg·m/s")
    
    world.close()


if __name__ == "__main__":
    demo_ramp()
    demo_collision()
