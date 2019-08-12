import pandas as pd
import numpy as np
import json
from rdp import rdp

xyz = np.array([
    [0,0,0],
    [0,1,0],
    [0,2,0],
])
xyz.shape

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
            np.ones(normal.shape[1]).reshape(-1, 1),
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

np.concatenate(
    [
        (point_head - point_tail),
        np.sum((point_head - point_tail) * point_head, axis=1).reshape(-1, 1),
    ],
    axis=1
)

planes_head, signs_head = create_planes_signs(point_head - point_tail, point_head, point_tail)
planes_tail, signs_tail = create_planes_signs(point_tail - point_head, point_tail, point_head)
point_test = np.array([
    [60,-9,0],
    [0,-9,0],
    [0,-2,0],
    [0,-1,0],
    [0,0,0],
    [0,1,0],
    [0,5,0],
])
test_points_vs_planes_signs(point_test, planes_head, signs_head)
test_points_vs_planes_signs(point_test, planes_tail, signs_tail)
test_points_vs_planes_signs(point_test, planes_ahead, signs_ahead) & test_points_vs_planes_signs(point_test, planes_behind, signs_behind)
planes.shape, signs.shape




point_test = point_lead

point4_test = np.concatenate(
    [
        point_test,
        np.ones(point_test.shape[0]).reshape(-1, 1),
    ],
    axis=1
)
# point4_test.shape
# plane_ahead.shape
# point4_test.reshape(1, -1, 4)
# plane_ahead.reshape(-1, 1, 4)
# point4_test.reshape(1, -1, 4) * plane_ahead.reshape(-1, 1, 4)
# np.sum(point4_test.reshape(1, -1, 4) * plane_ahead.reshape(-1, 1, 4), axis=2)
# np.sum(point4_test.reshape(1, -1, 4) * plane_ahead.reshape(-1, 1, 4), axis=2).shape

np.sum(point4_test.reshape(1, -1, 4) * plane_ahead.reshape(-1, 1, 4), axis=2) * signs_correct.reshape(-1, 1) >= 0
# dim 0 == plane
# dim 1 == test point
# so 1,2 boolean True means
# point 2 is in front of or on plane 1

signs_test = np.sum(point4_test * plane_ahead, axis=1)
signs_test
signs_test * signs_correct >= 0



pt_bisect_base = xyz[1:-1, :]
pt_bisect_ahead = xyz[2:, :]
pt_bisect_behind = xyz[:-2, :]
pt_bisect_base
pt_bisect_ahead
pt_bisect_behind


vector_ahead = pt_bisect_ahead - pt_bisect_base
vector_behind = pt_bisect_behind - pt_bisect_base

a = vector_behind
ap = -vector_behind
ab = vector_ahead - vector_behind
pt_middle = a + (np.sum(ap * ab, axis=1) / np.sum(ab * ab, axis=1)).unsqueeze(1) * ab
