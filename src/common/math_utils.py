import numpy as np

def rot_x(roll):
    c, s = np.cos(roll), np.sin(roll)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s,  c],
    ])

def rot_y(pitch):
    c, s = np.cos(pitch), np.sin(pitch)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c],
    ])

def rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ])

def euler_zyx(yaw, pitch, roll):
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)

# R = euler_zyx(yaw, pitch, roll)
# R = 3 -> by 3 rotation matrix, pts(3d point) 곱해서 transformation matrix
def apply_transform(T, pts):
    """
    T: (4,4), pts: (N,3)
    """
    N = pts.shape[0]
    pts_h = np.hstack([pts, np.ones((N,1))])
    out = (T @ pts_h.T).T
    return out[:, :3]
