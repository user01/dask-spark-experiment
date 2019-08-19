import pandas as pd
import numpy as np
import json
from numba import jit

# #############################################################################

# Heavily modified from: https://github.com/fhirschmann/rdp

# Copyright (c) 2014 Fabian Hirschmann <fabian@hirschmann.email>
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

@jit(nopython=True, fastmath=True, cache=True, parallel=False)
def pldist(point, start, end):
    """
    Calculates the distance from ``point`` to the line given
    by the points ``start`` and ``end``.

    :param point: a point
    :type point: numpy array
    :param start: a point of the line
    :type start: numpy array
    :param end: another point of the line
    :type end: numpy array
    """
    if np.all(np.equal(start, end)):
        return np.linalg.norm(point - start)

    segment = end - start
    point_start = point - start

    segment_sqr_length = np.sum(segment * segment)
    t = np.sum(segment * point_start) / segment_sqr_length
    if t < 0.0:
        near_pt = start
    elif t >= 1.0:
        near_pt = end
    else:
        near_pt = start + t * segment

    return np.linalg.norm(point - near_pt)

# pldist(np.array([0,0.]),np.array([0,0.]),np.array([1,0.]))
# pldist(np.array([1,0.]),np.array([0,0.]),np.array([1,0.]))
# pldist(np.array([0.5,0.]),np.array([0,0.]),np.array([1,0.]))
# pldist(np.array([2,0.]),np.array([0,0.]),np.array([1,0.]))
# pldist(np.array([2,1.]),np.array([0,0.]),np.array([1,0.]))


@jit(nopython=True, fastmath=True, cache=True, parallel=False)
def _rdp_iter(M, start_index, last_index, epsilon):
    stk = []
    stk.append([start_index, last_index])
    global_start_index = start_index
    indices = np.ones(last_index - start_index + 1) > 0

    while stk:
        start_index, last_index = stk.pop()

        dmax = 0.0
        index = start_index

        for i in range(index + 1, last_index):
            if indices[i - global_start_index]:
                d = pldist(M[i], M[start_index], M[last_index])
                if d > dmax:
                    index = i
                    dmax = d

        if dmax > epsilon:
            stk.append([start_index, index])
            stk.append([index, last_index])
        else:
            for i in range(start_index + 1, last_index):
                indices[i - global_start_index] = False

    return indices


@jit(nopython=True, fastmath=True, cache=True, parallel=False)
def rdp_iter(M, epsilon):
    """
    Simplifies a given array of points.

    Iterative version.

    :param M: an array
    :type M: numpy array
    :param epsilon: epsilon in the rdp algorithm
    :type epsilon: float
    :param dist: distance function
    :type dist: function with signature ``f(point, start, end)`` -- see :func:`rdp.pldist`
    :param return_mask: return the mask of points to keep instead
    :type return_mask: bool
    """
    mask = rdp_mask(M, epsilon)
    return M[mask]

@jit(nopython=True, fastmath=True, cache=True, parallel=False)
def rdp_mask(M, epsilon):
    """
    Simplifies a given array of points.

    Iterative version.

    :param M: an array
    :type M: numpy array
    :param epsilon: epsilon in the rdp algorithm
    :type epsilon: float
    :param dist: distance function
    :type dist: function with signature ``f(point, start, end)`` -- see :func:`rdp.pldist`
    :param return_mask: return the mask of points to keep instead
    :type return_mask: bool
    """
    mask = _rdp_iter(M, 0, len(M) - 1, epsilon)
    return mask


# #############################################################################


@jit(nopython=True, fastmath=True, cache=True, parallel=False)
def np_dot(x, y, axis=1):
    """
    Axis based dot product of vectors
    """
    return np.sum(x * y, axis=axis)

#
# @jit(nopython=True, fastmath=True, cache=True, parallel=False)
# def plane_masks(normals, pts_plane, pts_forward, pts_test):
#     """
#     """
#     correct_side_signs = np_dot(pts_forward - pts_plane, normals, axis=1)
#     diff = pts_test.reshape(-1, 1, 3) - pts_plane.reshape(1, -1, 3)
#     masks = np_dot(diff, normals.reshape(1, -1, 3), axis=2) * correct_side_signs.reshape(1, -1) >= 0
#     return masks


@jit(nopython=True, fastmath=True, cache=True, parallel=False)
def _plane_masks(pts_plane, pts_forward, pts_test):
    """
    Given points on the plane, a point forward from the
    plane (normal via the first), and points to test as forward of the plane,
    return the mask of points that test as in front
    """
    normals = pts_forward - pts_plane
    correct_side_signs = np_dot(pts_forward - pts_plane, normals, axis=1)
    diff = pts_test.reshape(-1, 1, 3) - pts_plane.reshape(1, -1, 3)
    masks = np_dot(diff, normals.reshape(1, -1, 3), axis=2) * correct_side_signs.reshape(1, -1) >= 0
    return masks


@jit(nopython=True, fastmath=True, cache=True, parallel=False)
def generate_all_masks(sequence, pts_test):
    """
    Generate all the masks for the sequence (a series of 3d points)
    To be 'adjacent' to a segment section, the point must be ahead of the
    'ahead' planes (ie in front of the starting plane) and behind the 'behind'
    planes (ie behind the ending plane)

    The planes are both orthogonal (flat capped for just this segment) and
    bisection (angled between the segment and it's neighboring segment). The
    orthogonal are always valid, but bisection is only valid when there is a
    neighboring segment

    If a point is ahead of either ahead planes and behind either behind
    planes, then the point is adjacent.
    """
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


@jit(nopython=True, fastmath=True, cache=True, parallel=False)
def closests_pt(p, a_s, b_s):
    """Find the closest point on series of segments to the point"""
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
            picked = a_s[idx]  # start
        elif t > 1:
            picked = b_s[idx]  # end
        else:
            # pick along segment
            picked = a_s[idx] + t * (b_s[idx] - a_s[idx])
        distance = np.linalg.norm(picked - p)

        if smallest_distance < 0 or smallest_distance > distance:
            smallest_distance = distance
            best_point = picked

    return smallest_distance, best_point


@jit(nopython=True, fastmath=True, cache=True, parallel=False)
def pick_points(sequence, points, wbt_id, api_ids, mds, threshold: float):
    """
    Given a WBT sequence, finds points within the planer masks and finds
    closest point on WBT

    Returns wbt_id, api_id, distance (m), md (m), wbt_pt(xyz), nns_pt(xyz)
    """
    assert points.shape[0] == mds.shape[0]
    assert points.shape[1] == 3
    assert len(mds.shape) == 1
    mask_points_per_segment = generate_all_masks(sequence, pts_test=points)
    results = np.ones((mask_points_per_segment.shape[0], 10)) * -1
    vs = sequence[:-1]
    ws = sequence[1:]
    for idx in range(mask_points_per_segment.shape[0]):
        mask_for_segments = mask_points_per_segment[idx]
        if not np.any(mask_for_segments):
            continue
        distance, point = closests_pt(points[idx], vs[mask_for_segments], ws[mask_for_segments])
        if distance <= threshold:
            results[idx] = np.array(
                [
                    wbt_id,          # 0
                    api_ids[idx],    # 1
                    distance,        # 2
                    mds[idx],        # 3
                    point[0],        # 4
                    point[1],        # 5
                    point[2],        # 6
                    points[idx][0],  # 7
                    points[idx][1],  # 8
                    points[idx][2],  # 9
                ]
            )
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


@jit(nopython=True, fastmath=True, cache=True, parallel=False)
def np_cross(a, b):
    """
    Simple numba compatible cross product of vectors
    """
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])

# a = np.array([-0.23193776, -0.70617841, -0.66896706])
# b = np.array([-0.14878152, -0.64968963,  0.74549812])
# np.allclose(np.cross(a, b), np_cross(a, b))


@jit(nopython=True, fastmath=True, cache=True, parallel=False)
def np_linalg_norm(data, axis=0):
    """
    Simple numba compatible cross product of vectors
    """
    return np.sqrt(np.sum(data * data, axis=axis))

# data = np.array([
#     [0,0,0],
#     [0,1,0],
#     [1,1,0],
#     [3,4,5],
# ])
# np.allclose(np_linalg_norm(data), np.sqrt(np.sum(data * data, axis=1)))
# np.allclose(np_linalg_norm(data), np.linalg.norm(data, axis=1))
# np.allclose(np_linalg_norm(np.array([3,4,0])), 5)

@jit(nopython=True, fastmath=True, cache=True, parallel=False)
def np_clip(arr, a_min, a_max):
    """
    Simple numba compatible clipping of array. Both min and max are required
    """
    lower = np.where(arr < a_min, a_min, arr)
    upper = np.where(lower > a_max, a_max, lower)
    return upper

# arr = np.arange(5)
# np.allclose(np.clip(arr, a_min=1, a_max=3), np.array([1, 1, 2, 3, 3]))
# np.allclose(np_clip(arr, a_min=1, a_max=3), np.array([1, 1, 2, 3, 3]))
# arr = np.array([
#     [1,0,1],
#     [4,-4,9.0],
#     [90,-9000.023,6],
# ])
# expected = np.array([
#     [1,0,1],
#     [1,0,1.0],
#     [1,0,1],
# ])
# np.allclose(np.clip(arr, a_min=0, a_max=1), expected)
# np.allclose(np_clip(arr, a_min=0, a_max=1), expected)


# ####################################################################


@jit(nopython=True, fastmath=True, cache=True, parallel=False)
def np_min(arr):
    """
    np.min across axis 0
    """
    # Interestingly, for small sets, this is 5x the default np.min(axis=0)
    return np.array([arr[:, axis].min() for axis in range(arr.shape[1])], dtype=np.float32)

@jit(nopython=True, fastmath=True, cache=True, parallel=False)
def np_max(arr):
    """
    np.max across axis 0
    """
    # Interestingly, for small sets, this is 5x the default np.max(axis=0)
    return np.array([arr[:, axis].max() for axis in range(arr.shape[1])], dtype=np.float32)

@jit(nopython=True, fastmath=True, cache=True, parallel=False)
def relevant_coordinates_mask(api_ids, points, sequence, threshold:float):
    """
    Row-wise mask for the apis that are within the simple bounding box
    for consideration
    """

    buffer_min = np_min(sequence) - threshold
    buffer_max = np_max(sequence) + threshold
    msk = np.zeros(api_ids.shape) <= 0.0
    for api_id in np.unique(api_ids):
        msk_current = api_ids == api_id
        points_api = points[msk_current]
        local_min = np_min(points_api)
        if np.any((buffer_min <= local_min) & (local_min <= buffer_max)):
            msk = msk | msk_current
            continue

        local_max = np_max(points_api)
        if np.any((buffer_min <= local_max) & (local_max <= buffer_max)):
            msk = msk | msk_current

    return msk

# ####################################################################


@jit(nopython=True, fastmath=True, cache=True, parallel=False)
def wbt_coordinate_prepare(wbts_api_id, coordinates_np, epsilon):
    """
    Prepare the coordinates for processing
    """
    wbt_mask = coordinates_np[:, 0] == wbts_api_id
    coordinates_wbt = coordinates_np[wbt_mask, :]
    coordinates_other = coordinates_np[~wbt_mask, :]
    xyz_sequence = rdp_iter(coordinates_wbt[:, 2:], epsilon=epsilon)

    apis_others = coordinates_other[:, 0]
    md_others = coordinates_other[:, 1]
    xyz_other = coordinates_other[:, 2:]

    return (
        np.ascontiguousarray(xyz_sequence),
        np.ascontiguousarray(md_others),
        np.ascontiguousarray(apis_others),
        np.ascontiguousarray(xyz_other),
    )


@jit(nopython=True, fastmath=True, cache=True, parallel=False)
def nnpairs(
    wbts_api_ids: np.array,
    coordinates: np.array,
    spi: np.array,
    threshold: float = 914.0,
    segment_length: float = 15.0,
):
    """
    Calculate the pairwise relationships between WBTs and NNs
    Returns the vector relations of nns position to wbt and the derived statistics
    Works only with float32
    """
    spi_values = spi.astype(np.float32)
    coordinates_np = interpolate_coords(coordinates.astype(np.float32), segment_length=segment_length)
    vectors_lst = []
    stats_lst = []
    for wbts_api_id in wbts_api_ids.astype(np.float32):
        sequence, mds, api_ids, points_all = wbt_coordinate_prepare(wbts_api_id, coordinates_np, epsilon=segment_length)

        # # benchmarks find this to be slightly slower under a test set
        # points_msk = relevant_coordinates_mask(api_ids, points_all, sequence, threshold=threshold)
        # points = points_all[points_msk]
        points = points_all

        vectors = pick_points(
            wbt_id=wbts_api_id,
            sequence=sequence,
            mds=mds,
            api_ids=api_ids,
            points=points,
            threshold=threshold,
        ).astype(
            np.float32
        )
        assert vectors.shape[1] == 10, "Improper vector size"

        vectors_lst.append(vectors)

        # vector style outputs - use this code to plot in view-points
        # if False:
        #     vectors[vectors[:, 0] == 4, 3:9].reshape(-1, 3).round(0).tolist()

        # Compute the common WBT values
        spi_value = spi_values[spi_values[:, 0] == wbts_api_id][0]
        vector_cos_angle_lat_wbt = spi_value[10] # angle in degrees
        wellhead = spi_value[1:4]
        east = spi_value[4:7]
        north = spi_value[7:10]
        east_delta = east - wellhead
        north_delta = north - wellhead
        local_up = np_cross(east_delta, north_delta)

        local_up_len = np.linalg.norm(local_up)
        local_up_unit = local_up / local_up_len

        wbt = vectors[:, 4:7]
        nns = vectors[:, 7:10]
        delta = nns - wbt
        distance_3d = np_linalg_norm(delta, axis=1)
        distance_3d_valid = distance_3d > 1e-9
        distance_3d_local_safe = np.where(distance_3d_valid, distance_3d, 1)
        projected_vertical = local_up_unit.reshape(-1, 3) * (np.sum(local_up_unit * delta, axis=1)).reshape(-1, 1)
        distance_vertical = np_linalg_norm(projected_vertical, axis=1)
        assert not (distance_vertical > distance_3d_local_safe).any()
        theta_valid = distance_3d_valid & (distance_3d > distance_vertical)
        theta = np.where(
            theta_valid, np.arcsin(np_clip(distance_vertical / distance_3d_local_safe, a_min=-1, a_max=1)), np.pi / 2
        )
        distance_2d = (distance_3d ** 2 - distance_vertical ** 2) ** 0.5

        # Returns api_id, distance (m), md (m), wbt_pt(xyz), nns_pt(xyz)

        wbt_heel_xyz = sequence[0]
        wbt_toe_xyz = sequence[-1]
        lateral = wbt_toe_xyz - wbt_heel_xyz
        lateral_unit = lateral / np_linalg_norm(lateral)
        lateral_normal = np_cross(lateral_unit, local_up_unit)
        # correct_side_sign = np.dot(lateral_normal, lateral_normal)
        # # this will always be positive - dotting a vector with itself
        # # the correct side is the 'right' side. funny, right?

        nns_ids = np.unique(vectors[:, 1])
        stats = np.ones((nns_ids.shape[0], 34)) * -50

        for idx, nns_id in enumerate(nns_ids):
            stats[idx, 0] = nns_id
            stats[idx, 1] = wbts_api_id
            mask_nns = vectors[:, 1] == nns_id
            vectors_nns = vectors[mask_nns]

            # md diffs
            distance = vectors_nns[-1, 3] - vectors_nns[0, 3]
            stats[idx, 2] = distance

            vector_cos_angle_lat_nns = spi_values[spi_values[:, 0] == nns_id][0, 10]
            stats[idx, 3] = angle_diff(vector_cos_angle_lat_wbt, vector_cos_angle_lat_nns)
            # TODO: Azimuth delta

            nns_heel_xyz = vectors[0, 7:10]
            nns_toe_xyz = vectors[-1, 7:10]
            # the vector 'shadow' on the right facing normal is the distance from the
            # lateral plane, the plane that touches the heel, toe, and a position up
            # up being determined by the wellhead
            # 1.0 == right
            # 2.0 == left
            sidenns_heel = 1.0 if np.dot(lateral_normal, nns_heel_xyz) > 0 else 2.0
            sidenns_toe = 1.0 if np.dot(lateral_normal, nns_toe_xyz) > 0 else 2.0
            stats[idx, 4] = sidenns_heel
            stats[idx, 5] = sidenns_toe

            PERCENTILES = [0, 25, 50, 75, 100]
            distance_2d_nns = distance_2d[mask_nns]
            stats[idx, 6] = np.mean(distance_2d_nns)
            stats[idx, 7] = np.std(distance_2d_nns)
            stats[idx, 8:13] = np.percentile(distance_2d_nns, PERCENTILES)

            distance_3d_nns = distance_3d[mask_nns]
            stats[idx, 13] = np.mean(distance_3d_nns)
            stats[idx, 14] = np.std(distance_3d_nns)
            stats[idx, 15:20] = np.percentile(distance_3d_nns, PERCENTILES)

            distance_vertical_nns = distance_vertical[mask_nns]
            stats[idx, 20] = np.mean(distance_vertical_nns)
            stats[idx, 21] = np.std(distance_vertical_nns)
            stats[idx, 22:27] = np.percentile(distance_vertical_nns, PERCENTILES)

            theta_nns = theta[mask_nns]
            stats[idx, 27] = np.mean(theta_nns)
            stats[idx, 28] = np.std(theta_nns)
            stats[idx, 29:34] = np.percentile(theta_nns, PERCENTILES)

        stats_lst.append(stats.astype(np.float32))


    # TODO: pull this out into numba function
    # stats_all = np.concatenate(stats_lst)
    size = 0
    idx = 0
    for arr in stats_lst:
        size += arr.shape[0]
    stats_all = np.empty(shape=(size, stats_lst[0].shape[1]), dtype=np.float32)
    for arr in stats_lst:
        size = arr.shape[0]
        stats_all[idx:idx+size, :] = arr
        idx += size

    size = 0
    idx = 0
    for arr in vectors_lst:
        size += arr.shape[0]
    vectors_all = np.empty(shape=(size, vectors_lst[0].shape[1]), dtype=np.float32)
    for arr in vectors_lst:
        size = arr.shape[0]
        vectors_all[idx:idx+size, :] = arr
        idx += size

    return vectors_all, stats_all

# ##################################################################

@jit(nopython=True, fastmath=True, cache=True, parallel=False)
def interpolate(coors, segment_length):
    """
    Interpolate new points along the coordinate set based on the desired
    segment length
    """
    # roughly 10x in numba
    md_min = coors[:,1][0]
    md_max = coors[:,1][-1]
    approx_count = int((md_max - md_min) / segment_length)
    mds_new = np.linspace(md_min, md_max, approx_count + 1)
    pts_new = np.empty((mds_new.shape[0], 5), dtype=np.float32)
    pts_new[:, 0] = coors[0,0] # assign the id to the new points
    pts_new[:, 1] = mds_new # assign the interpolated measure depths

    for idx, md in enumerate(mds_new):
        msk = coors[:,1] <= md
        data_current = coors[msk,:][-1]
        point_current = data_current[2:]
        if msk.all():
            pts_new[idx, 2:] = point_current
            break
        data_next = coors[~msk,:][0]
        point_next = data_next[2:]
        md_current = data_current[1]
        md_next = data_next[1]
        factor = (md - md_current) / (md_next - md_current)
        new_point = factor * (point_next - point_current) + point_current
        pts_new[idx, 2:] = new_point

    return pts_new


# %timeit interpolate.py_func(coors)
# %timeit interpolate(coors)

@jit(nopython=True, fastmath=True, cache=True, parallel=False)
def malloc_interpolate_coords(coordinates, segment_length):
    """
    Memory Allocation for the newly interpolated coordinates
    """
    new_size = 0
    for nns_id in np.unique(coordinates[:,0]):
        coors = coordinates[coordinates[:, 0] == nns_id]
        md_min = coors[:,1][0]
        md_max = coors[:,1][-1]
        new_size += 1 + int((md_max - md_min) / segment_length)

    coordinates_new = np.empty((new_size, 5), dtype=np.float32)
    return coordinates_new


@jit(nopython=True, fastmath=True, cache=True, parallel=False)
def interpolate_coords(coordinates, segment_length):
    """
    Perform interpolation for all coordinates
    """
    coords_new = malloc_interpolate_coords(coordinates, segment_length)
    ptr = 0
    for nns_id in np.unique(coordinates[:,0]):
        coors = coordinates[coordinates[:, 0] == nns_id]
        results = interpolate(coors, segment_length)
        coords_new[ptr:ptr+results.shape[0],:] = results
        ptr += results.shape[0]
    return coords_new


# interpolate_coords(coordinates_np, segment_length=segment_length)
# %timeit interpolate_coords(coordinates_np, segment_length=segment_length)
# %timeit interpolate_coords(coordinates_np, segment_length=segment_length)

# #############################################################


def side_np_to_str(srs:pd.Series):
    return np.where(srs.values < 1.5, "RIGHT", "LEFT")

# https://stackoverflow.com/a/7869457
# angle differences - in degrees
@jit(nopython=True, fastmath=True, cache=True, parallel=False)
def mod_flat(a, n):
    return (a % n + n) % n


@jit(nopython=True, fastmath=True, cache=True, parallel=False)
def angle_diff(a, b):
    """Compute the absolute degree angle difference between two directions"""
    res = a - b
    return np.abs(mod_flat((res + 180), 360) - 180)


# #############################################################


def blank_stats():
    """A blank instance of the data set returned by the pairs operation"""
    return pd.DataFrame({
        "NNS": pd.Series([], dtype="object"),
        "WBT": pd.Series([], dtype="object"),
        "distance_segment": pd.Series([], dtype="float64"),
        "azimuth_delta": pd.Series([], dtype="float64"),
        "sidenns_heel": pd.Series([], dtype="object"),
        "sidenns_toe": pd.Series([], dtype="object"),
        "distance_2d_mean": pd.Series([], dtype="float64"),
        "distance_2d_std": pd.Series([], dtype="float64"),
        "distance_2d_min": pd.Series([], dtype="float64"),
        "distance_2d_25percentile": pd.Series([], dtype="float64"),
        "distance_2d_50percentile": pd.Series([], dtype="float64"),
        "distance_2d_75percentile": pd.Series([], dtype="float64"),
        "distance_2d_max": pd.Series([], dtype="float64"),
        "distance_3d_mean": pd.Series([], dtype="float64"),
        "distance_3d_std": pd.Series([], dtype="float64"),
        "distance_3d_min": pd.Series([], dtype="float64"),
        "distance_3d_25percentile": pd.Series([], dtype="float64"),
        "distance_3d_50percentile": pd.Series([], dtype="float64"),
        "distance_3d_75percentile": pd.Series([], dtype="float64"),
        "distance_3d_max": pd.Series([], dtype="float64"),
        "distance_vertical_mean": pd.Series([], dtype="float64"),
        "distance_vertical_std": pd.Series([], dtype="float64"),
        "distance_vertical_min": pd.Series([], dtype="float64"),
        "distance_vertical_25percentile": pd.Series([], dtype="float64"),
        "distance_vertical_50percentile": pd.Series([], dtype="float64"),
        "distance_vertical_75percentile": pd.Series([], dtype="float64"),
        "distance_vertical_max": pd.Series([], dtype="float64"),
        "theta_mean": pd.Series([], dtype="float64"),
        "theta_std": pd.Series([], dtype="float64"),
        "theta_min": pd.Series([], dtype="float64"),
        "theta_25percentile": pd.Series([], dtype="float64"),
        "theta_50percentile": pd.Series([], dtype="float64"),
        "theta_75percentile": pd.Series([], dtype="float64"),
        "theta_max": pd.Series([], dtype="float64")
    })

def blank_vectors():
    """A blank instance of the raw the pairs operation"""
    return pd.DataFrame({
        "WBT": pd.Series([], dtype="object"),
        "NNS": pd.Series([], dtype="object"),
        "NNS_MD": pd.Series([], dtype="float64"),
        "Distance": pd.Series([], dtype="float64"),
        "WBT_X": pd.Series([], dtype="float64"),
        "WBT_Y": pd.Series([], dtype="float64"),
        "WBT_Z": pd.Series([], dtype="float64"),
        "NNS_X": pd.Series([], dtype="float64"),
        "NNS_Y": pd.Series([], dtype="float64"),
        "NNS_Z": pd.Series([], dtype="float64")
    })

# #############################################################


# read local data
apis = pd.read_parquet("apis.pq")
coordinates = pd.read_parquet("coordinates.pq")
spi = pd.read_parquet("spi.pq")

# pandas building
spi_mapping = spi.reset_index().rename(columns={"index": "API_ID"})
api_mapping = spi_mapping[["API", "API_ID"]]

wbts_api_ids = apis.merge(api_mapping)["API_ID"].values.astype(np.float64)

# coordinates.merge(api_mapping).head(2)
coordinates_np = (
    coordinates.merge(spi_mapping[["API_ID", "API", "PerfFrom", "PerfTo"]])
    .pipe(lambda idf: idf[(idf["MD"] >= idf["PerfFrom"]) & (idf["MD"] <= idf["PerfTo"])])[
        ["API_ID", "MD", "X", "Y", "Z"]
    ]
    .values.astype(np.float64)
)
spi_values = spi_mapping[
    ["API_ID", "X", "Y", "Z", "X_East", "Y_East", "Z_East", "X_North", "Y_North", "Z_North", 'Vector_Cos_Angle_Lat']
].values


threshold = 914.0
segment_length = 15.0

vectors_np, stats_np = nnpairs(
    wbts_api_ids,
    coordinates_np,
    spi_values,
    threshold=threshold,
    segment_length=segment_length,
)

vectors_blanker = blank_vectors()
stats_blanker = blank_stats()

vector_results = pd.DataFrame(
    vectors_np.astype(np.float64),
    columns=['WBT', 'NNS', 'NNS_MD', 'Distance', 'WBT_X', 'WBT_Y', 'WBT_Z', 'NNS_X', 'NNS_Y', 'NNS_Z'],
).assign(
    WBT=lambda idf: idf[['WBT']].astype(np.int64).merge(api_mapping, how='left', left_on='WBT', right_on='API_ID')['API'],
    NNS=lambda idf: idf[['NNS']].astype(np.int64).merge(api_mapping, how='left', left_on='NNS', right_on='API_ID')['API'],
)
assert vectors_blanker.dtypes.equals(vector_results.dtypes)


stats_results = pd.DataFrame(
    stats_np.astype(np.float64),
    columns=['NNS', 'WBT', 'distance_segment', 'azimuth_delta', 'sidenns_heel', 'sidenns_toe', 'distance_2d_mean', 'distance_2d_std', 'distance_2d_min', 'distance_2d_25percentile', 'distance_2d_50percentile', 'distance_2d_75percentile', 'distance_2d_max', 'distance_3d_mean', 'distance_3d_std', 'distance_3d_min', 'distance_3d_25percentile', 'distance_3d_50percentile', 'distance_3d_75percentile', 'distance_3d_max', 'distance_vertical_mean', 'distance_vertical_std', 'distance_vertical_min', 'distance_vertical_25percentile', 'distance_vertical_50percentile', 'distance_vertical_75percentile', 'distance_vertical_max', 'theta_mean', 'theta_std', 'theta_min', 'theta_25percentile', 'theta_50percentile', 'theta_75percentile', 'theta_max'],
).assign(
    WBT=lambda idf: idf[['WBT']].astype(np.int64).merge(api_mapping, how='left', left_on='WBT', right_on='API_ID')['API'],
    NNS=lambda idf: idf[['NNS']].astype(np.int64).merge(api_mapping, how='left', left_on='NNS', right_on='API_ID')['API'],
    sidenns_heel=lambda idf: idf['sidenns_heel'].pipe(side_np_to_str),
    sidenns_toe=lambda idf: idf['sidenns_toe'].pipe(side_np_to_str),
)

assert stats_blanker.dtypes.equals(stats_results.dtypes)

0

#
# f16s = np.random.RandomState(451).randn(1_000,5_000).astype(np.float16)
# f32s = np.random.RandomState(451).randn(1_000,5_000).astype(np.float32)
# f64s = np.random.RandomState(451).randn(1_000,5_000).astype(np.float64)
#
# %timeit np.std(np.mean(f16s, axis=1))
# %timeit np.std(np.mean(f32s, axis=1))
# %timeit np.std(np.mean(f64s, axis=1))
#
# f16s = np.random.RandomState(451).randn(5_000,5_000).astype(np.float16)
# f32s = np.random.RandomState(451).randn(5_000,5_000).astype(np.float32)
# f64s = np.random.RandomState(451).randn(5_000,5_000).astype(np.float64)
#
# %timeit np.std(np.mean(f16s, axis=1))
# %timeit np.std(np.mean(f32s, axis=1))
# %timeit np.std(np.mean(f64s, axis=1))
#
#
# [rdp(coordinates_np[coordinates_np[:, 0] == wbts_api_id, :], 15) for wbts_api_id in wbts_api_ids]
# [coordinates_np[coordinates_np[:, 0] == wbts_api_id, :][:, 2:].shape for wbts_api_id in wbts_api_ids]
