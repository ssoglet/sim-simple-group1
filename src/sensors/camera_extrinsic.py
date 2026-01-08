import numpy as np
from typing import TYPE_CHECKING
from src.common.math_utils import rot_z, rot_x, rot_y, euler_zyx

if TYPE_CHECKING:
    from src.dynamics.vehicle import VehicleState

def make_T_cam_world():
    """
    Define camera frame explicitly:
    - Camera +Z : forward
    - Camera +X : right
    - Camera +Y : down
    """

    # Camera position in world (x forward, y left, z up)
    t = np.array([1.5, 0.0, 1.3])

    # World -> Camera rotation
    # World axes: X forward, Y left, Z up
    # Camera axes: Z forward, X right, Y down
    R_world_cam = np.array([
        [ 0, -1,  0],   # world Y(left) -> camera X(right)
        [ 0,  0, -1],   # world Z(up)   -> camera Y(down)
        [ 1,  0,  0],   # world X(fwd)  -> camera Z(fwd)
    ], dtype=float)

    # Optional: pitch down
    pitch = np.deg2rad(-5.0)
    R_pitch = rot_z(pitch)
    # yaw = np.deg2rad(20)
    # roll = np.deg2rad(5)
    # R_euler = euler_zyx(yaw,pitch,roll)
    R = R_pitch @ R_world_cam

    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = t
    return T