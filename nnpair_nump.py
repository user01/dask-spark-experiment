import pandas as pd
import numpy as np
import json
from rdp import rdp


x_ = np.linspace(-5., 5., 40)

points = []
for x in x_:
    for y in x_:
        for z in x_:
            points.append((x,y,z))
points = np.stack(points)
# points.shape
# json.dumps(points.tolist())

def np_dot(x, y, axis=1):
    return np.sum(x * y, axis=axis)

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


def generate_all_masks(sequence):
    mask_ahead_ortho = _plane_masks(pts_plane=sequence[:-1], pts_forward=sequence[1:], pts_test=points)
    mask_behind_ortho = _plane_masks(pts_plane=sequence[1:], pts_forward=sequence[:-1], pts_test=points)
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
    # bisection masks at the extreme fail automatically - they have nothing to compare against

    mask_points_per_segment = (mask_ahead_ortho | mask_ahead_bisect) & (mask_behind_ortho | mask_behind_bisect)
    return mask_points_per_segment


def closest_pts(p, v, w):
    """For many points against a single segment, find the closest points"""
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


mds = np.array([0.0, 5.0])
# closests_pt(np.array([0,0,0]), vs, ws)
# closests_pt(np.array([0,1,0]), vs, ws)
# closests_pt(np.array([1,1,1]), vs, ws)
# closests_pt(np.array([0,1,1]), vs, ws)
sequence = np.array([
    [0,0,0],
    [0,1,0],
    [0,2,1],
    [0,3,4],
    [1,4,5],
])
sequence.tolist()
mask_points_per_segment = generate_all_masks(sequence)
threshold = 1.5
results = np.ones((mask_points_per_segment.shape[0], 8)) * -1
vs = sequence[1:]
ws = sequence[:-1]
results.shape
mask_points_per_segment.shape
for idx in range(mask_points_per_segment.shape[0]):
    mask_for_segments = mask_points_per_segment[idx]
    mask_for_segments.shape
    distance, point = closests_pt(points[idx], vs[mask_for_segments], ws[mask_for_segments])
    if distance <= threshold:
        results[idx] = np.array([
            distance,
            # mds[idx],
            0.0, # MD
            point[0],
            point[1],
            point[2],
            points[idx][0],
            points[idx][1],
            points[idx][2],
        ])

results[results[:, 0] > -1].shape
results[results[:, 0] > -1][:, -3:].shape
print(json.dumps(results[results[:, 0] > -1][:, -3:].round(2).tolist()))





# RDP the WBT sequence
# find planes for the segments
# find the masks for all the segments
# or for each nns point, find the relevant segments
# then for every segment, find the closest point
