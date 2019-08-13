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


0




pt_test = np.array([0,-1,0])
np.dot(pt_test - pt_plane, normal) * correct_side_sign > 0

pt_test = np.array([320,-1,0])
np.dot(pt_test - pt_plane, normal) * correct_side_sign > 0

pt_test = np.array([320,-1,232330])
np.dot(pt_test - pt_plane, normal) * correct_side_sign > 0

pt_test = np.array([320,-1,232330])
np.dot(pt_test - pt_plane, normal) * correct_side_sign > 0

pt_test = np.array([320,1,232330])
np.dot(pt_test - pt_plane, normal) * correct_side_sign > 0

pt_test = np.array([320,23321,232330])
np.dot(pt_test - pt_plane, normal) * correct_side_sign > 0


normal = np.array([0,1,0.])
pt_plane = np.array([0,100,0.])

pt_forward = np.array([0,101,0.])
correct_side_sign = np.dot(pt_forward - pt_plane, normal)

pt_test = np.array([0,1,0])
np.dot(pt_test - pt_plane, normal) * correct_side_sign > 0

pt_test = np.array([0,-1,0])
np.dot(pt_test - pt_plane, normal) * correct_side_sign > 0

pt_test = np.array([320,-1,0])
np.dot(pt_test - pt_plane, normal) * correct_side_sign > 0

pt_test = np.array([320,-1,232330])
np.dot(pt_test - pt_plane, normal) * correct_side_sign > 0

pt_test = np.array([320,-1,232330])
np.dot(pt_test - pt_plane, normal) * correct_side_sign > 0

pt_test = np.array([320,1,232330])
np.dot(pt_test - pt_plane, normal) * correct_side_sign > 0

pt_test = np.array([320,23321,232330])
np.dot(pt_test - pt_plane, normal) * correct_side_sign > 0


normal = np.array([1,0,0.])
pt_plane = np.array([100,100,100.])

pt_forward = np.array([101,100,-90.])
correct_side_sign = np.dot(pt_forward - pt_plane, normal)

pt_test = np.array([101,100,100])
np.dot(pt_test - pt_plane, normal) * correct_side_sign > 0

pt_test = np.array([0,-1,0])
np.dot(pt_test - pt_plane, normal) * correct_side_sign > 0

pt_test = np.array([320,-1,0])
np.dot(pt_test - pt_plane, normal) * correct_side_sign > 0

pt_test = np.array([320,-1,232330])
np.dot(pt_test - pt_plane, normal) * correct_side_sign > 0

pt_test = np.array([320,-1,232330])
np.dot(pt_test - pt_plane, normal) * correct_side_sign > 0

pt_test = np.array([320,1,232330])
np.dot(pt_test - pt_plane, normal) * correct_side_sign > 0

pt_test = np.array([320,23321,232330])
np.dot(pt_test - pt_plane, normal) * correct_side_sign > 0




xyz = np.array([
    [0,0,0],
    [0,1,0],
    [0,2,0],
])

point_head = xyz[:-1]
point_tail = xyz[1:]


def create_planes_signs(normal, point_on_plane, point_on_plane_side):
    """
    normal is the normal vector for the plane
    point_on_plane is a point that is on the plane
    point_on_plane_side a point on the desireable (positive) side of the plane

    Returns
    -------
    tuple(np.array, np.array)
        planes: np.array (n, 4) floats - plane representations
        signs: np.array(n,) floats - proper signage for the correct size
    """
    planes = np.concatenate(
        [
            normal,
            np.sum(normal * point_on_plane, axis=1).reshape(-1, 1),
        ],
        axis=1
    )
    point4_other = np.concatenate(
        [
            point_on_plane_side,
            np.ones(point_on_plane_side.shape[0]).reshape(-1, 1),
        ],
        axis=1
    )
    signs_correct = np.sum(point4_other * planes, axis=1)
    return planes, signs_correct

def test_points_vs_planes_signs(points, planes, signs):
    """
    points: np.array (p, 3) floats - test points
    planes: np.array (n, 4) floats - plane representations
    signs: np.array(n,) floats - proper signage for the correct size

    Returns
    -------
    np.array
        results: np.array (n, p) bools - point satifies plane
    """
    point_count = points.shape[0]
    point4_test = np.concatenate(
        [
            points,
            np.ones(point_count).reshape(-1, 1),
        ],
        axis=1
    )
    result = np.sum(
        point4_test.reshape(1, -1, 4) * planes.reshape(-1, 1, 4),
        axis=2
    ) * signs.reshape(-1, 1) >= 0
    assert result.shape == (planes.shape[0], point_count)
    # dim 0 == plane
    # dim 1 == test point
    # so 1,2 boolean True means
    # point 2 is in front of or on plane 1
    return result


planes_head, signs_head = create_planes_signs(point_tail - point_head, point_head, point_tail)
planes_tail, signs_tail = create_planes_signs(point_head - point_tail, point_tail, point_head)
point_test = np.array([
    [0,-1,0],
    [0,0,0],
    [0,1,0],
    [0,2,0],
    [0,3,0],
])
test_points_vs_planes_signs(point_test, planes_head, signs_head)
test_points_vs_planes_signs(point_test, planes_tail, signs_tail)
