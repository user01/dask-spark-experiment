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


# normal = np.array([0,1,0.])
# pt_plane = np.array([0,0,0.])
# pt_forward = np.array([0,1,0.])
# correct_side_sign = np.dot(pt_forward - pt_plane, normal)
#
# pt_test = np.array([0,1,0])
# np.dot(pt_test - pt_plane, normal) * correct_side_sign > 0
#
# mask_plane1 = (np.dot(points - pt_plane, normal) * correct_side_sign >= 0)
# points[mask_plane1].tolist()
#
# normal = np.array([-1,-1,-1.])
# pt_plane = np.array([0,3,0.])
# pt_forward = np.array([-1,2,-1.])
# correct_side_sign = np.dot(pt_forward - pt_plane, normal)
#
# pt_test = np.array([0,1,0])
# np.dot(pt_test - pt_plane, normal) * correct_side_sign > 0
#
# mask_plane2 = (np.dot(points - pt_plane, normal) * correct_side_sign >= 0)
# points[mask_plane2].tolist()
# points[mask_plane2 & mask_plane1].tolist()
#
# # ####################
# normals = np.array([ # normals that define the plane
#     [0,1,0],
#     [-1,-1,-1.],
# ])
# pts_plane = np.array([ # pts fixed on plane
#     [0,0,0],
#     [0,3,0.],
# ])
# pts_forward = np.array([ # point on the forward side of the plane
#     [0,1,0],
#     [-1,2,-1.],
# ])
#
def np_dot(x, y, axis=1):
    return np.sum(x * y, axis=axis)
# correct_side_signs = np_dot(pts_forward - pts_plane, normals, axis=1)
#
#
# diff = points.reshape(-1, 1, 3) - pts_plane.reshape(1, -1, 3)
# masks = np_dot(diff, normals.reshape(1, -1, 3), axis=2) * correct_side_signs.reshape(1, -1) >= 0
# for segment_mask in masks.T:
#     break
# points[masks.all(axis=1)].tolist()

# ########################################################

def plane_masks(normals, pts_plane, pts_forward, pts_test):
    correct_side_signs = np_dot(pts_forward - pts_plane, normals, axis=1)
    diff = pts_test.reshape(-1, 1, 3) - pts_plane.reshape(1, -1, 3)
    masks = np_dot(diff, normals.reshape(1, -1, 3), axis=2) * correct_side_signs.reshape(1, -1) >= 0
    return masks

def _plane_masks(pts_plane, pts_forward, pts_test):
    normals = pts_forward - pts_plane
    correct_side_signs = np_dot(pts_forward - pts_plane, normals, axis=1)
    diff = pts_test.reshape(-1, 1, 3) - pts_plane.reshape(1, -1, 3)
    masks = np_dot(diff, normals.reshape(1, -1, 3), axis=2) * correct_side_signs.reshape(1, -1) >= 0
    return masks

# need to compute ahead masks and behind masks
# for each there's the orthogonal and the bisection
# orthogonal exist for every segment, while bisections for all the first and last
# so bisections automatically are false at the edges

sequence = np.array([
    [0,0,0],
    [0,1,0],
    [0,2,1],
    [0,3,4],
    [1,4,5],
])

# pts_forward = sequence[1:]
# pts_plane = sequence[:-1]
# normal_forward = pts_forward - pts_plane
# normal_backward = -normal_forward

mask_ahead_ortho = _plane_masks(pts_plane=sequence[:-1], pts_forward=sequence[1:], pts_test=points)
mask_behind_ortho = _plane_masks(pts_plane=sequence[1:], pts_forward=sequence[:-1], pts_test=points)
mask_ahead_ortho.shape
mask_behind_ortho.shape
f"There are {mask_behind_ortho.shape[1]} segments in this sequence"

pts_plane = sequence[:-2] + (0.5 * (sequence[2:] - sequence[:-2]))
mask_ahead_bisect = np.concatenate(
    [
        np.array([False] * mask_ahead_ortho.shape[0]).reshape(-1, 1),
        _plane_masks(pts_plane=pts_plane, pts_forward=sequence[:-2], pts_test=points),
    ],
    axis=1,
)
mask_behind_bisect = np.concatenate(
    [
        _plane_masks(pts_plane=pts_plane, pts_forward=sequence[2:], pts_test=points),
        np.array([False] * mask_ahead_ortho.shape[0]).reshape(-1, 1),
    ],
    axis=1,
)
mask_ahead_bisect.shape
mask_behind_bisect.shape
# bisection masks at the extreme fail automatically - they have nothing to compare against

mask_points_per_segment = (mask_ahead_ortho | mask_ahead_bisect) & (mask_behind_ortho | mask_behind_bisect)

mask_points_per_segment.shape


# def closest_pt(p, v, w):
#     d2 = np.linalg.norm(v - w) # this may be slow even under numba
#     if d2 <= 0:
#         return v
#     t = np.dot(p - v, w - v) / d2
#     if t < 0:
#         return v
#     elif t > 1.0:
#         return w
#     project = v + t * (w - v)
#     return project
#
# closest_pt(
#     v = np.array([0,0,0]),
#     w = np.array([0,1,0]),
#     p = np.array([0,-1,0]),
# )
# closest_pt(
#     v = np.array([0,0,0]),
#     w = np.array([0,1,0]),
#     p = np.array([0,-1,220]),
# )
# closest_pt(
#     v = np.array([0,0,0]),
#     w = np.array([0,1,0]),
#     p = np.array([0,11,220]),
# )
# closest_pt(
#     v = np.array([0,0,0]),
#     w = np.array([0,1,0]),
#     p = np.array([0,-1,0]),
# )


def closest_pts(p, v, w):
    assert v.shape == (3,)
    assert w.shape == (3,)
    assert p.shape[1] == 3
    d2 = np.linalg.norm(v - w) # this may be slow even under numba

    t = np.dot(p - v, w - v) / d2
    assert t.shape[0] == p.shape[0]

    closest_pts = np.where(
        (t < 0).reshape(-1, 1),
        v.reshape(1, -1),
        np.where(
            (t > 1).reshape(-1, 1),
            w.reshape(1, -1),
            v + t.reshape(-1, 1) * (w - v).reshape(1, 3),
        )
    )
    closest_pts.shape
    assert closest_pts.shape == p.shape
    return closest_pts

# given multiple segments, pick the closest point on any to the point p

vs = np.array([
    [0,0,0],
    [0,1,0],
])
ws = np.array([
    [0,1,0],
    [0,1,1],
])

def closests_pt(p, vs, ws):
    """Find the closest point on segments to the point"""
    assert vs.shape == ws.shape
    d2 = np.linalg.norm(vs - ws, axis=1) # this may be slow even under numba

    ts = np_dot(
        p.reshape(1, 3) - vs,
        ws - vs,
        axis=1
    ) / d2
    assert ts.shape[0] == vs.shape[0]

    smallest_distance = -1.0
    for idx in range(ts.shape[0]):
        t = ts[idx]
        if t < 0:
            # pick vs for this one
            picked = vs[idx]
        elif t > 1:
            # pick ws for this one
            picked = ws[idx]
        else:
            # pick along segment
            picked = vs[idx] + t * (ws[idx] - vs[idx])
        distance = np.linalg.norm(picked - p)

        if smallest_distance < 0 or smallest_distance > distance:
            smallest_distance = distance
            best_point = picked

    return smallest_distance, best_point

vs = np.array([
    [0,0,0],
    [0,1,0],
])
ws = np.array([
    [0,1,0],
    [0,1,1],
])
closests_pt(np.array([0,0,0]), vs, ws)
closests_pt(np.array([0,1,0]), vs, ws)
closests_pt(np.array([1,1,1]), vs, ws)
closests_pt(np.array([0,1,1]), vs, ws)



for mask_for_segments in mask_points_per_segment:
    break
%timeit mask_points_per_segment.T[idx]
%timeit mask_points_per_segment[:,idx]
%timeit mask_points_per_segment.T[idx]
%timeit mask_points_per_segment[:,idx]

for idx in range(mask_points_per_segment.shape[1]):
    # presently on segment idx
    mask_for_points = mask_points_per_segment[:, idx]
    # these are the points which satisfy this segment's planes
    v = sequence[idx]
    w = sequence[idx + 1]
    # these are the start/end of the segment
    best_points = closest_pts(
        v = v,
        w = w,
        p = points[mask_for_points],
    )
    break

idxs = np.arange(points.shape[0])
mkss = np.linalg.norm(points[mask_for_points] - best_points, axis=1) < 5
picked_points_idxs = idxs[mask_for_points][mkss]

points[picked_points_idxs].tolist()

closest_pts(
    v = np.array([0,0,0]),
    w = np.array([0,1,0]),
    p = np.array([
        [0,-2,0],
        [0,-1,0],
        [0,0.5,0],
        [0,0.5,0.1],
        [0,1,0],
        [0,2,0],
    ]),
)




# RDP the WBT sequence
# find planes for the segments
# find the masks for all the segments
# or for each nns point, find the relevant segments
# then for every segment, find the closest point
