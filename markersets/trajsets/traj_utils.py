import numpy as np
import multiprocessing
import os
import warnings
from scipy.signal import savgol_filter
from numpy import ndarray


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def change_of_basis_in_time(query_v, new_basis):
    return np.einsum('ij,ikj->ik', query_v, new_basis)  # i:time, j:global dimension, k:axis


def traj_worker(args_in, set_class, save_dir_path, verbose=True,
                transpose=False, emg_slice=None, mocap_slice=None, nan_crop=False, name_tag=None, **kwargs):

    from markersets.trajsets.trajset import TrajSet
    if verbose:
        print(multiprocessing.current_process().name + ' working on ' + args_in[0])
    ms: TrajSet = set_class(*args_in, **kwargs)
    ms.se_proc_traj()

    if name_tag is None:
        save_name_end = '.json'
    else:
        save_name_end = f'_{name_tag}.json'

    ms.save_json(os.path.join(save_dir_path, os.path.splitext(os.path.basename(args_in[0]))[0] + save_name_end),
                 transpose=transpose, emg_slice=emg_slice, mocap_slice=mocap_slice, nan_crop=nan_crop)
    # ms.save_pickle(os.path.join(save_dir_path, os.path.splitext(os.path.basename(args_in[0]))[0] + '.pickle'))


def nan_region_find_dict(dict_in: dict):
    
    # arr_items = np.array([a[:, -3:] for a in list(dict_in.values())])
    # edges = map(lambda x: np.where(np.diff(x) != 0)[0] + 1, np.isnan(arr_items[:, :, -1]).astype(int))

    arr_items = np.array([np.sum(a, axis=1) for a in list(dict_in.values())])
    edges = map(lambda x: np.where(np.diff(x) != 0)[0] + 1, np.isnan(arr_items[:, :]).astype(int))

    leading = [0]
    trailing = [arr_items.shape[1]]  # Calculated as the positive distance from start
    for ang, edge in zip(arr_items, edges):
        if len(edge) - np.any(np.isnan(ang[0])) - np.any(np.isnan(ang[-1])) > 0:
            # All internal NaN regions should be interpolated!
            warnings.warn('NaN value detected inside time-series!')
            warnings.warn('Internal NaN region in thread: ' + multiprocessing.current_process().name)
        elif len(edge) - np.any(np.isnan(ang[0])) - np.any(np.isnan(ang[-1])) < 0:
            raise Exception('Incorrect NaN section detection...')  # Means edges was incorrectly calculated
        if np.any(np.isnan(ang[0])):
            leading.append(edge[0])
        if np.any(np.isnan(ang[-1])):
            trailing.append(edge[-1])
    return max(leading), min(trailing)


def nan_savgol_filter(arr: ndarray, non_nan_region, **kwargs):
    # assuming 2d (time, dimension) array
    if arr.ndim is not 2 or arr.shape[0] < arr.shape[1]:
        warnings.warn("non_savgol written for 2d array (time, dimension)")
    out = np.empty_like(arr)
    out[:] = np.nan
    out[non_nan_region[0]:non_nan_region[1], :] = savgol_filter(arr[non_nan_region[0]:non_nan_region[1]], **kwargs)
    return out


def compare_nan_array(func, a, thresh):
    out = ~np.isnan(a)
    out[out] = func(a[out], thresh)
    return out


def try_seaborn():
    try:
        import seaborn as sns
        sns.set()
    except ImportError:
        print('Could not import seaborn package, plotting with default matplotlib settings')
