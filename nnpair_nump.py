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

# given multiple segments, pick the closest point on any to the point p
def closests_pt(p, a_s, b_s):
    """Find the closest point on segments to the point"""
    assert a_s.shape == b_s.shape
    ab_s = b_s - a_s
    ap_s = p.reshape(1, 3) - a_s
    ab_sqr_len = np.sum(ab_s * ab_s, axis=1)

    t_s = np.sum(ab_s * ap_s, axis=1) / ab_sqr_len
    assert t_s.shape[0] == a_s.shape[0]

    smallest_distance = -1.0
    for idx in range(t_s.shape[0]):
        t = t_s[idx]
        if t < 0:
            picked = a_s[idx] # start
        elif t > 1:
            picked = b_s[idx] # end
        else:
            # pick along segment
            picked = a_s[idx] + t * (b_s[idx] - a_s[idx])
        distance = np.linalg.norm(picked - p)

        if smallest_distance < 0 or smallest_distance > distance:
            smallest_distance = distance
            best_point = picked

    return smallest_distance, best_point

# v = vs[mask_for_segments]
# w = ws[mask_for_segments]
#
# a = v[0]
# b = w[0]
# p = (v + 0.5 * (w - v))[0]
# ab = b - a
# ap = p - a
# ab_sqr_len = np.sum(ab * ab)
# t = np.sum(ab * ap) / ab_sqr_len
# (v + t * (w - v))[0]
#
# vw = (v[0] - w[0])
# vw_len = np.sum(vw * vw)
# np.dot(p - v[0], w[0] - v[0]) / d2
# np.sqrt(np.dot(p - v[0], w[0] - v[0])) / d2

mds = np.array([0.0, 5.0])

sequence = np.array([
    [0,0,0],
    [0,1,0],
    [0,2,1],
    [0,3,4],
    [1,4,5],
])
sequence = np.array([
    [0,2,1],
    [0,3,4],
])

# vector style
np.stack([sequence[idx:idx+2] for idx in range(sequence.shape[0] - 1)]).reshape(-1, 3).tolist()

mask_points_per_segment = generate_all_masks(sequence)

# print(json.dumps(points[mask_points_per_segment.T[0]].round(3).tolist()))

threshold = 1.5
results = np.ones((mask_points_per_segment.shape[0], 8)) * -1
vs = sequence[:-1]
ws = sequence[1:]
results.shape
mask_points_per_segment.shape
for idx in range(mask_points_per_segment.shape[0]):
    mask_for_segments = mask_points_per_segment[idx]
    mask_for_segments.shape
    if not np.any(mask_for_segments):
        continue
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


v = vs[mask_for_segments]
w = ws[mask_for_segments]


closests_pt(np.array([[0,0,0]]), vs[mask_for_segments], ws[mask_for_segments])[1]
closest_pts(np.array([[0,0,0]]), v, w)
closests_pt(w, vs[mask_for_segments], ws[mask_for_segments])[1]
closest_pts(w, v[0], w[0])
closests_pt(v, vs[mask_for_segments], ws[mask_for_segments])[1]
closest_pts(v, v[0], w[0])
closests_pt(v + 0.5 * (w - v), vs[mask_for_segments], ws[mask_for_segments])[1]
closest_pts(v + 0.5 * (w - v), v[0], w[0])

0

# RDP the WBT sequence
# find planes for the segments
# find the masks for all the segments
# or for each nns point, find the relevant segments
# then for every segment, find the closest point
