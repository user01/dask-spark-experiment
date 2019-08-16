import pandas as pd
import numpy as np
import json
from rdp import rdp
from numba import jit


@jit(nopython=True, fastmath=True)
def np_dot(x, y, axis=1):
    return np.sum(x * y, axis=axis)

@jit(nopython=True, fastmath=True)
def plane_masks(normals, pts_plane, pts_forward, pts_test):
    correct_side_signs = np_dot(pts_forward - pts_plane, normals, axis=1)
    diff = pts_test.reshape(-1, 1, 3) - pts_plane.reshape(1, -1, 3)
    masks = np_dot(diff, normals.reshape(1, -1, 3), axis=2) * correct_side_signs.reshape(1, -1) >= 0
    return masks

@jit(nopython=True, fastmath=True)
def _plane_masks(pts_plane, pts_forward, pts_test):
    normals = pts_forward - pts_plane
    correct_side_signs = np_dot(pts_forward - pts_plane, normals, axis=1)
    diff = pts_test.reshape(-1, 1, 3) - pts_plane.reshape(1, -1, 3)
    masks = np_dot(diff, normals.reshape(1, -1, 3), axis=2) * correct_side_signs.reshape(1, -1) >= 0
    return masks

@jit(nopython=True, fastmath=True)
def generate_all_masks(sequence, pts_test):
    assert pts_test.shape[1] == 3
    # need to compute ahead masks and behind masks
    # for each there's the orthogonal and the bisection
    # orthogonal exist for every segment, while bisections for all the first and last
    # so bisections automatically are false at the edges
    mask_ahead_ortho = _plane_masks(pts_plane=sequence[:-1], pts_forward=sequence[1:], pts_test=pts_test)
    mask_behind_ortho = _plane_masks(pts_plane=sequence[1:], pts_forward=sequence[:-1], pts_test=pts_test)
    # f"There are {mask_behind_ortho.shape[1]} segments in this sequence"

    pts_plane = sequence[:-2] + (0.5 * (sequence[2:] - sequence[:-2]))
    mask_ahead_bisect = np.concatenate(
        (
            np.array([False] * mask_ahead_ortho.shape[0]).reshape(-1, 1),
            _plane_masks(pts_plane=pts_plane, pts_forward=sequence[:-2], pts_test=pts_test),
        ),
        axis=1,
    )
    mask_behind_bisect = np.concatenate(
        (
            _plane_masks(pts_plane=pts_plane, pts_forward=sequence[2:], pts_test=pts_test),
            np.array([False] * mask_ahead_ortho.shape[0]).reshape(-1, 1),
        ),
        axis=1,
    )
    # bisection masks at the extreme fail automatically - they have nothing to compare against

    mask_points_per_segment = (mask_ahead_ortho | mask_ahead_bisect) & (mask_behind_ortho | mask_behind_bisect)
    assert mask_points_per_segment.shape == (pts_test.shape[0], sequence.shape[0] - 1)
    return mask_points_per_segment

@jit(nopython=True, fastmath=True)
def closests_pt(p, a_s, b_s):
    """Find the closest point on segments to the point"""
    # given multiple segments, pick the closest point on any to the point p
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

@jit(nopython=True, fastmath=True)
def pick_points(sequence, points, api_ids, mds, threshold:float):
    """
    Returns api_id, distance (m), md (m), wbt_pt(xyz), nns_pt(xyz)
    """
    assert points.shape[0] == mds.shape[0]
    assert points.shape[1] == 3
    assert len(mds.shape) == 1
    mask_points_per_segment = generate_all_masks(sequence, pts_test=points)
    results = np.ones((mask_points_per_segment.shape[0], 9)) * -1
    vs = sequence[:-1]
    ws = sequence[1:]
    for idx in range(mask_points_per_segment.shape[0]):
        mask_for_segments = mask_points_per_segment[idx]
        if not np.any(mask_for_segments):
            continue
        distance, point = closests_pt(points[idx], vs[mask_for_segments], ws[mask_for_segments])
        if distance <= threshold:
            results[idx] = np.array([
                api_ids[idx],   # 0
                distance,       # 1
                mds[idx],       # 2
                point[0],       # 3
                point[1],       # 4
                point[2],       # 5
                points[idx][0], # 6
                points[idx][1], # 7
                points[idx][2], # 8
            ])
    return results[results[:, 0] > -1]
#
#
# x_ = np.linspace(-2., 7., 40)
#
# points = []
# for x in x_:
#     for y in x_:
#         for z in x_:
#             points.append((x,y,z))
# points = np.stack(points)
#
# mds = np.random.RandomState(451).randn(points.shape[0])
# sequence = np.array([
#     [0,0,0],
#     [0,1,0],
#     [0,2,1],
#     [0,3,4],
#     [1,4,5],
# ])
#
# # vector style
# np.stack([sequence[idx:idx+2] for idx in range(sequence.shape[0] - 1)]).reshape(-1, 3).tolist()
#
# # points.shape
# %timeit pick_points(sequence=sequence.astype(np.float64), mds=mds, api_ids=mds, points=points.astype(np.float64), threshold=1.5)
# # res = pick_points.py_func(sequence=sequence.astype(np.float64), mds=mds, api_ids=mds, points=points.astype(np.float64), threshold=1.5)
# # %timeit pick_points.py_func(sequence=sequence.astype(np.float64), mds=mds, points=points.astype(np.float64), threshold=1.5)
# # # results = pick_points(sequence=sequence.astype(np.float64), points=points.astype(np.float64), threshold=1.5)
# # # results = pick_points(sequence=sequence.astype(np.float64), points=points.astype(np.float64), threshold=1.5)

# read local data
apis = pd.read_parquet('apis.pq')
coordinates = pd.read_parquet('coordinates.pq')
spi = pd.read_parquet('spi.pq')


# pandas building
spi_mapping = spi.reset_index().rename(columns={'index':'API_ID'})
api_mapping = spi_mapping[['API', 'API_ID']]

wbts_api_ids = apis.merge(api_mapping)['API_ID'].values.astype(np.float64)

# coordinates.merge(api_mapping).head(2)
coordinates_np = coordinates.merge(
    spi_mapping[['API_ID', 'API', 'PerfFrom', 'PerfTo']]
).pipe(
    lambda idf: idf[
        (idf['MD'] >= idf['PerfFrom']) & (idf['MD'] <= idf['PerfTo'])
    ]
)[
    ['API_ID', 'MD', 'X', 'Y', 'Z']
].values.astype(np.float64)
spi_values = spi_mapping[
    ['API_ID', 'X', 'Y', 'Z', 'X_East', 'Y_East', 'Z_East', 'X_North', 'Y_North', 'Z_North']
].values

# NOTE: Loop through the ids
wbts_api_id = wbts_api_ids[0]

wbt_mask = coordinates_np[:, 0] == wbts_api_id
coordinates_wbt = coordinates_np[wbt_mask, :]
coordinates_other = coordinates_np[~wbt_mask, :]
xyz_sequence = rdp(coordinates_wbt[:, 2:], 15)
# xyz_sequence.round(2).tolist()

apis_others = coordinates_other[:, 0]
md_others = coordinates_other[:, 1]
xyz_other = coordinates_other[:, 2:]

results = pick_points(
    sequence=np.ascontiguousarray(xyz_sequence),
    mds=np.ascontiguousarray(md_others),
    api_ids=np.ascontiguousarray(apis_others),
    points=np.ascontiguousarray(xyz_other),
    threshold=914.0,
)
# %timeit pick_points(sequence=np.ascontiguousarray(xyz_sequence), mds=np.ascontiguousarray(md_others), api_ids=np.ascontiguousarray(apis_others),points=np.ascontiguousarray(xyz_other),threshold=914,)

# vector style outputs
if False:
    results[results[:, 0] == 4, 3:9].reshape(-1, 3).round(0).tolist()

# Compute the common WBT values
spi_value = spi_values[spi_values[:, 0] == wbts_api_id][0]
wellhead = spi_value[1:4]
east = spi_value[4:7]
north = spi_value[7:10]
east_delta = east - wellhead
north_delta = north - wellhead
local_up = np.cross(east_delta, north_delta)

local_up_len = np.linalg.norm(local_up)
local_up_unit = local_up / local_up_len

wbt = results[:, 3:6]
nns = results[:, 6:9]
delta = nns - wbt
distance_3d = np.linalg.norm(delta, axis=1)
distance_3d_valid = distance_3d > 1e-9
distance_3d_local_safe = np.where(distance_3d_valid, distance_3d, 1)
projected_vertical = local_up_unit.reshape(-1, 3) * (np.sum(local_up_unit * delta, axis=1)).reshape(-1, 1)
distance_vertical = np.linalg.norm(projected_vertical, axis=1)
assert not (distance_vertical > distance_3d_local_safe).any()
theta_valid = distance_3d_valid & (distance_3d > distance_vertical)
theta = np.where(
    theta_valid,
    np.arcsin(
        np.clip(
            distance_vertical / distance_3d_local_safe,
            a_min=-1,
            a_max=1,
        )
    ),
    np.pi / 2,
)
distance_2d = (distance_3d ** 2 - distance_vertical ** 2) ** 0.5

# Returns api_id, distance (m), md (m), wbt_pt(xyz), nns_pt(xyz)

wbt_heel_xyz = xyz_sequence[0]
wbt_toe_xyz = xyz_sequence[-1]
lateral = wbt_toe_xyz - wbt_heel_xyz
lateral_unit = lateral / np.linalg.norm(lateral)
lateral_normal = np.cross(lateral_unit, local_up_unit)
correct_side_sign = np.dot(lateral_normal, lateral_normal)
# this will always be positive - dotting a vector with itself
# the correct side is the 'right' side. funny, right?

nns_ids = np.unique(results[:, 0])
stats = np.ones((nns_ids.shape[0], 32)) * -50

for idx, nns_id in enumerate(nns_ids):
    stats[idx, 0] = wbts_api_id
    stats[idx, 1] = nns_id
    mask_nns = results[:, 0] == nns_id
    results_nns = results[mask_nns]

    # md diffs
    distance = results_nns[-1, 2] - results_nns[0, 2]
    stats[idx, 2] = distance

    nns_heel_xyz = results[0, 6:9]
    nns_toe_xyz = results[-1, 6:9]
    # the vector 'shadow' on the right facing normal is the distance from the
    # lateral plane, the plane that touches the heel, toe, and a position up
    # up being determined by the wellhead
    # 1.0 == right
    # 2.0 == left
    sidenns_heel = 1.0 if np.dot(lateral_normal, nns_heel_xyz) > 0 else 2.0
    sidenns_toe = 1.0 if np.dot(lateral_normal, nns_toe_xyz) > 0 else 2.0
    stats[idx, 3] = sidenns_heel
    stats[idx, 4] = sidenns_toe

    distance_2d_nns = distance_2d[mask_nns]
    stats[idx, 5] = np.mean(distance_2d_nns)
    stats[idx, 6:11] = np.percentile(distance_2d_nns, [0, 25, 50, 75, 100])
    stats[idx, 11] = np.std(distance_2d_nns)

    distance_3d_nns = distance_3d[mask_nns]
    stats[idx, 12] = np.mean(distance_3d_nns)
    stats[idx, 13:18] = np.percentile(distance_3d_nns, [0, 25, 50, 75, 100])
    stats[idx, 18] = np.std(distance_3d_nns)

    distance_vertical_nns = distance_vertical[mask_nns]
    stats[idx, 19] = np.mean(distance_vertical_nns)
    stats[idx, 20:25] = np.percentile(distance_vertical_nns, [0, 25, 50, 75, 100])
    stats[idx, 25] = np.std(distance_vertical_nns)

    theta_nns = theta[mask_nns]
    stats[idx, 26] = np.mean(theta_nns)
    stats[idx, 26:31] = np.percentile(theta_nns, [0, 25, 50, 75, 100])
    stats[idx, 31] = np.std(theta_nns)
