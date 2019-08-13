import pandas as pd
import numpy as np
import json
from rdp import rdp


x_ = np.linspace(-10., 10., 7)

points = []
for x in x_:
    for y in x_:
        for z in x_:
            points.append((x,y,z))
points = np.stack(points)
points.shape
points.tolist()


normal = np.array([0,1,0.])
pt_plane = np.array([0,0,0.])
pt_forward = np.array([0,1,0.])
correct_side_sign = np.dot(pt_forward - pt_plane, normal)

pt_test = np.array([0,1,0])
np.dot(pt_test - pt_plane, normal) * correct_side_sign > 0

mask_plane1 = (np.dot(points - pt_plane, normal) * correct_side_sign >= 0)
points[mask_plane1].tolist()

normal = np.array([-1,-1,-1.])
pt_plane = np.array([0,3,0.])
pt_forward = np.array([-1,2,-1.])
correct_side_sign = np.dot(pt_forward - pt_plane, normal)

pt_test = np.array([0,1,0])
np.dot(pt_test - pt_plane, normal) * correct_side_sign > 0

mask_plane2 = (np.dot(points - pt_plane, normal) * correct_side_sign >= 0)
points[mask_plane2].tolist()
points[mask_plane2 & mask_plane1].tolist()

# ####################
normals = np.array([ # normals that define the plane
    [0,1,0],
    [-1,-1,-1.],
])
pts_plane = np.array([ # pts fixed on plane
    [0,0,0],
    [0,3,0.],
])
pts_forward = np.array([ # point on the forward side of the plane
    [0,1,0],
    [-1,2,-1.],
])

def np_dot(x, y, axis=1):
    return np.sum(x * y, axis=axis)
correct_side_signs = np_dot(pts_forward - pts_plane, normals, axis=1)


diff = points.reshape(-1, 1, 3) - pts_plane.reshape(1, -1, 3)
masks = np_dot(diff, normals.reshape(1, -1, 3), axis=2) * correct_side_signs.reshape(1, -1) >= 0
points[masks.all(axis=1)].tolist()
0


def closest_pt(p, v, w):
    d2 = np.linalg.norm(v - w)
    if d2 <= 0:
        return v
    t = np.dot(p - v, w - v) / d2
    if t < 0:
        return v
    elif t > 1.0:
        return w
    project = v + t * (w - v)
    return project

closest_pt(
    v = np.array([0,0,0]),
    w = np.array([0,1,0]),
    p = np.array([0,-1,0]),
)
closest_pt(
    v = np.array([0,0,0]),
    w = np.array([0,1,0]),
    p = np.array([0,-1,220]),
)
closest_pt(
    v = np.array([0,0,0]),
    w = np.array([0,1,0]),
    p = np.array([0,11,220]),
)



def closest_pt(p, v, w):
    d2 = np.linalg.norm(v - w)
    if d2 <= 0:
        return v
    t = np.dot(p - v, w - v) / d2
    if t < 0:
        return v
    elif t > 1.0:
        return w
    project = v + t * (w - v)
    return project

closest_pt(
    v = np.array([0,0,0]),
    w = np.array([0,1,0]),
    p = np.array([0,-1,0]),
)
