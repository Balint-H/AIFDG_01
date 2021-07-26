"""
MarkerSet classes for extracting hip trajectory projected on the ground plane (assuming calibrated for the x-y plane).
Also loads sEMG recordings in the supplied file, and pairs it with the trajectory for prediction model training.

Aimed for driving parametrised gait controllers, primarily used in character animation, such as Motion Matching.

In this file the 'se' notation mnemonic refers to the method having a side effect on the object.

See also:

    Büttner, M. and Clavet, S., 2015. Motion Matching-The Road to Next Gen Animation. Proc. of Nucl. ai.

    Clavet, S., 2016. Motion matching and the road to next-gen animation. In Proc. of GDC.

    Bergamin, K., Clavet, S., Holden, D. and Forbes, J.R., 2019. DReCon: data-driven responsive control of physics-based
    characters. ACM Transactions On Graphics (TOG), 38(6), pp.1-11.

"""
from markersets.jcs import MarkerSet
from typing import Optional, Dict, Any, Union
from utility.C3D import C3DServer
import json
from scipy import interpolate
from math import floor, ceil
import matplotlib.pyplot as plt
import sys
from bvh import Bvh, BvhNode
from abc import ABC, abstractmethod
from numpy import ndarray
import numpy as np
import warnings
import os

from markersets.trajsets.traj_utils import nan_region_find_dict, compare_nan_array, try_seaborn, traj_worker, \
    nan_savgol_filter, change_of_basis_in_time


class TrajSet(MarkerSet, ABC):
    dict_features: Dict[str, ndarray]

    def __init__(self, mocap_file, emg_file=None,
                 window_length=41, polyorder=3,
                 horizon_vector=(20, 40, 60),
                 vel_cutoff=0.0008):

        super(TrajSet, self).__init__(mocap_file=mocap_file, emg_file=emg_file)

        self.list_marker = []
        self.list_emg = []

        self.window_length = window_length
        self.poly_order = polyorder
        self.horizon_vector = horizon_vector
        self.vel_cutoff = vel_cutoff
        self.dict_features = dict()
        self.non_nan_region = (None, None)

        self.has_processed = False

    def se_proc_traj(self, load_emg=True, extract_motion_match_features=True):
        """
        Uses features described by DReCon paper. Generates vector of trajectory positions (t) and orientations (d) at
        20, 40 and 60 frames in the future (@ 60 Hz, this is adapted for the recording frequency).

        {t_20, t_40, t_60, d_20, d_40, d_60}
            t E R^2,
            d E R^2

        """
        if self.has_processed:
            warnings.warn("Repeated processing of " + self.mocap_file + "!")
        else:
            self.has_processed = True

        if not self.dict_mocap:
            self.se_load_mocap(load_emg=load_emg)
            self.se_marker_preproc()
        if not self.dict_emg and load_emg and self.emg_file is not None:
            self.dict_emg, self.emg_freq = self.get_emg_data()

        n_c3d_samples = self.dict_mocap['VHIP'].shape[0]

        t = self.dict_mocap['VHIP'][:, :2]
        t_lerper = interpolate.interp1d(range(n_c3d_samples),
                                        t,
                                        axis=0,
                                        assume_sorted=True)

        v = nan_savgol_filter(self.dict_mocap['VHIP'], self.non_nan_region,
                              window_length=self.window_length, polyorder=self.poly_order,
                              deriv=1,
                              axis=0)

        v_lerper = interpolate.interp1d(range(n_c3d_samples),
                                        v,
                                        axis=0,
                                        assume_sorted=True)

        def v_lerper_complex(q):
            cur_v = v_lerper(q)[:, :2]
            return cur_v[:, 0] + cur_v[:, 1] * 1j

        def d_lerper_complex(q):
            cur_d = v_lerper(q)[:, :2]
            normed_d = cur_d / np.linalg.norm(cur_d, axis=1)[:, None]
            return normed_d[:, 0] + normed_d[:, 1] * 1j

        horizon = np.array(self.horizon_vector) * self.mocap_freq / 60

        t_h = np.empty((n_c3d_samples, len(self.horizon_vector), 2))  # (ref time, horizon time, axis) rel position
        t_h[:] = np.nan

        v_h = np.empty((n_c3d_samples, len(self.horizon_vector), 2))  # (ref time, horizon time, axis) rel velocity
        v_h[:] = np.nan

        d_h = np.empty((n_c3d_samples, len(self.horizon_vector), 2))  # (ref time, horizon time, axis) rel orientation
        d_h[:] = np.nan

        s_h = np.empty((n_c3d_samples, len(self.horizon_vector), 2))  # (ref time, horizon time, axis) rel scaled ori.
        s_h[:] = np.nan

        for i in range(n_c3d_samples - ceil(max(horizon))):

            ref_dir = (self.dict_mocap['HipOrientation'][i, 1, 0] +
                       self.dict_mocap['HipOrientation'][i, 1, 1] * 1j)

            if np.isnan(ref_dir):
                continue

            ref_dir = ref_dir / abs(ref_dir)

            t_unrotated = t_lerper(i + horizon) - t[i]
            t_rotated_c = (t_unrotated[:, 0] + t_unrotated[:, 1] * 1j) / ref_dir
            t_h[i, :, :] = np.array([t_rotated_c.real, t_rotated_c.imag]).T

            cur_v_h = v_lerper(i + horizon)[:, :2]
            cur_v_h = (cur_v_h[:, 0] + cur_v_h[:, 1] * 1j) / ref_dir
            v_h[i, :, :] = np.array([cur_v_h.real, cur_v_h.imag]).T

            comp_d = d_lerper_complex(i + horizon) / ref_dir

            d_h[i, :, :] = np.array([comp_d.real, comp_d.imag]).T

        s_h = d_h[:, :, :] * np.linalg.norm(v_h, axis=2)[:, :, None]
        d_h = d_h * \
              (compare_nan_array(np.greater, np.linalg.norm(v_h, axis=2), self.vel_cutoff) *
               compare_nan_array(np.greater, np.linalg.norm(v, axis=1), self.vel_cutoff)[:, None])[:, :, None]

        self.dict_features = {
            't_h': t_h,
            'd_h': d_h,
        }

        # Global velocities just means dont subtract reference frame's velocity, but I think still use orientation!

        if extract_motion_match_features:
            v_g_l = change_of_basis_in_time(v, self.dict_mocap['HipOrientation'])

            p_l_rfoot = change_of_basis_in_time(self.dict_mocap['RVAN'] - self.dict_mocap['VHIP'],
                                                self.dict_mocap['HipOrientation'])
            p_l_lfoot = change_of_basis_in_time(self.dict_mocap['LVAN'] - self.dict_mocap['VHIP'],
                                                self.dict_mocap['HipOrientation'])

            v_g_rfoot = nan_savgol_filter(self.dict_mocap['RVAN'], self.non_nan_region,
                                          window_length=self.window_length, polyorder=self.poly_order,
                                          deriv=1,
                                          axis=0)
            v_g_rfoot = change_of_basis_in_time(v_g_rfoot, self.dict_mocap['HipOrientation'])

            v_g_lfoot = nan_savgol_filter(self.dict_mocap['LVAN'], self.non_nan_region,
                                          window_length=self.window_length, polyorder=self.poly_order,
                                          deriv=1,
                                          axis=0)
            v_g_lfoot = change_of_basis_in_time(v_g_lfoot, self.dict_mocap['HipOrientation'])

            motion_match_features = {
                'v_g_l': v_g_l,
                'p_l_lfoot': p_l_lfoot,
                'p_l_rfoot': p_l_rfoot,
                'v_g_lfoot': v_g_lfoot,
                'v_g_rfoot': v_g_rfoot
            }

            self.dict_features.update(motion_match_features)

        return

    @abstractmethod
    def se_marker_preproc(self):
        raise NotImplementedError()

    @abstractmethod
    def get_emg_data(self, file_reader: Optional[C3DServer] = None):
        raise NotImplementedError()

    def save_json(self, save_path, indent=None,
                  transpose=False, mocap_slice: Optional[slice] = None, emg_slice: Optional[slice] = None,
                  nan_crop=False, keys=None, **kwargs):

        if keys is None:
            keys = self.dict_features.keys()

        non_nan_features = nan_region_find_dict({k: i.reshape((i.shape[0], -1)) for k, i in self.dict_features.items()
                                                 if k in keys})
        if nan_crop:
            slice_start = (max((non_nan_features[0], mocap_slice.start))
                           if mocap_slice is not None and mocap_slice.start is not None else non_nan_features[0])
            slice_stop = (min((non_nan_features[1], mocap_slice.stop))
                           if mocap_slice is not None and mocap_slice.stop is not None else non_nan_features[1])
            slice_step = mocap_slice.step if mocap_slice is not None else 1
            mocap_slice = slice(slice_start, slice_stop, slice_step)

        dict_out = self.get_dict_out(mocap_slice=mocap_slice, emg_slice=emg_slice, transpose=transpose)

        with open(save_path, 'w') as fp:
            json.dump(dict_out, fp, indent=indent, **kwargs)
        return

    def get_dict_out(self, mocap_slice=None, emg_slice=None, transpose=False):
        if mocap_slice is None:
            dict_out = {k: i.reshape((i.shape[0], -1)).tolist()
                        if not transpose else i.reshape((i.shape[0], -1)).T.tolist()
                        for k, i in self.dict_features.items()}
        else:
            dict_out = {k: i[mocap_slice].reshape((i[mocap_slice].shape[0], -1)).tolist()
                        if not transpose else i[mocap_slice].reshape((i[mocap_slice].shape[0], -1)).T.tolist()
                        for k, i in self.dict_features.items()}

        if emg_slice is None:
            dict_out['EMG'] = {k: i.tolist() for k, i in self.dict_emg.items()}
        else:
            dict_out['EMG'] = {k: i[emg_slice].tolist() for k, i in self.dict_emg.items()}
        dict_out['Framerate'] = self.mocap_freq
        dict_out['Sampling Frequency'] = self.emg_freq
        return dict_out

    def show_ref_sys(self, mode='2d', save=False, frame_slice: Optional[slice] = None, **kwargs):
        try_seaborn()

        if frame_slice is None:
            frame_slice = slice(0, self.dict_mocap['HipOrientation'].shape[0])
        hip_orientation = self.dict_mocap['HipOrientation'][frame_slice]

        # region Insert 0 and nan values so all vectors of same local direction are plotted with one line
        # TODO: this could have been simpler with using the axis parameter in np.insert
        L = hip_orientation.shape[0]
        ix = np.insert(hip_orientation[:, 0, 0], slice(0, L + 10), 0)
        iy = np.insert(hip_orientation[:, 0, 1], slice(0, L + 10), 0)
        jx = np.insert(hip_orientation[:, 1, 0], slice(0, L + 10), 0)
        jy = np.insert(hip_orientation[:, 1, 1], slice(0, L + 10), 0)

        ix = np.insert(ix, slice(2, ix.shape[0] + 10, 2), np.nan)
        iy = np.insert(iy, slice(2, iy.shape[0] + 10, 2), np.nan)
        jx = np.insert(jx, slice(2, jx.shape[0] + 10, 2), np.nan)
        jy = np.insert(jy, slice(2, jy.shape[0] + 10, 2), np.nan)
        # endregion

        # region ---------------3D plot---------------
        if mode is '3d':
            from mpl_toolkits.mplot3d import Axes3D
            ax = plt.axes(projection='3d')
            ax.plot(ix, iy, np.arange(ix.shape[0]), **kwargs)
            ax.plot(jx, jy, np.arange(jx.shape[0]), **kwargs)

            ax.set_facecolor('none')
            ax.set_xlabel("Global X (AU)", fontsize=16)
            ax.set_ylabel("Global Y (AU)", fontsize=16)
            ax.set_zlabel('Frame (@' + str(self.mocap_freq) + ' Hz)', fontsize=16)

        # endregion

        # region --------------Animation--------------
        elif mode is 'anim':
            import matplotlib.animation as animation
            o = hip_orientation[:, :2, :2]
            fig, ax = plt.subplots(figsize=(10, 10))
            lines = []
            line_i, = ax.plot(np.append(0., o[0, 0, 0]), np.append(0., o[0, 0, 1]), label='Hip Right', **kwargs)
            line_j, = ax.plot(np.append(0., o[0, 1, 0]), np.append(0., o[0, 1, 1]), label='Hip Front', **kwargs)
            ax.legend()
            lines = [line_i, line_j]

            def init():

                ax.set_aspect('equal')
                ax.set_xlim([-1.1, 1.1])
                ax.set_ylim([-1.1, 1.1])

                ax.set_xlabel("Global X (m)", fontsize=16)
                ax.set_ylabel("Global Y (m)", fontsize=16)
                return lines

            def animate(i_frame):
                line_i.set_data(np.append(0., o[i_frame, 0, 0]), np.append(0., o[i_frame, 0, 1]))
                line_j.set_data(np.append(0., o[i_frame, 1, 0]), np.append(0., o[i_frame, 1, 1]))
                return lines

            ani = animation.FuncAnimation(fig, animate, interval=round(1 / self.mocap_freq * 1000),
                                          blit=True, init_func=init, frames=L)
            if save:
                ani.save('CoordSys.mp4', writer="ffmpeg")
        # endregion

        # region ------------Regular Plot-------------
        else:
            plt.plot(ix, iy)
            plt.plot(jx, jy)
        # endregion

        plt.show()

    def show_trajectories(self, mode='2d', save=False, frame_slice: Optional[slice] = None, **kwargs):
        try_seaborn()
        if frame_slice is None:
            frame_slice = slice(0, self.dict_features['t_h'].shape[0])

        t_h = self.dict_features['t_h'][frame_slice]
        n_horizon = t_h.shape[1]
        n_time = t_h.shape[0]
        t_h_stack = t_h.reshape((n_time * n_horizon, t_h.shape[2]))
        t_h_stack = np.insert(t_h_stack, slice(0, t_h_stack.shape[0] + 100, n_horizon), 0, axis=0)
        t_h_stack = np.insert(t_h_stack, slice(n_horizon + 1, t_h_stack.shape[0] + 100, n_horizon + 1), np.nan, axis=0)

        # region ---------------3D plot---------------
        if mode is '3d':
            from mpl_toolkits.mplot3d import Axes3D
            ax = plt.axes(projection='3d')

            ax.set_facecolor('none')
            ax.plot(t_h_stack[:, 0], t_h_stack[:, 1], np.arange(t_h_stack.shape[0]), **kwargs)
            ax.set_xlabel("Hip Forward (m)", fontsize=16)
            ax.set_ylabel("Hip Left (m)", fontsize=16)
            ax.set_zlabel('Frame (@' + str(self.mocap_freq) + ' Hz)', fontsize=16)
        # endregion

        # region --------------Animation--------------
        elif mode is 'anim':
            import matplotlib.animation as animation

            fig, ax = plt.subplots(figsize=(12, 10))
            line, = ax.plot(np.append(0., t_h[0, :, 0]), np.append(0, t_h[0, :, 1]), **kwargs)
            ax.set_aspect('equal')

            ax.set_xlim(- 0.1,
                        np.max(t_h_stack[~np.isnan(t_h_stack[:, 0]), 0]) + 0.1)
            ax.set_ylim(np.min(t_h_stack[~np.isnan(t_h_stack[:, 1]), 1]) - 0.1,
                        np.max(t_h_stack[~np.isnan(t_h_stack[:, 1]), 1]) + 0.1)
            ax.set_xlabel("Hip Forward (m)", fontsize=16)
            ax.set_ylabel("Hip Left (m)", fontsize=16)

            def animate(i_frame):
                line.set_xdata(np.append(0., t_h[i_frame, :, 0]))
                line.set_ydata(np.append(0., t_h[i_frame, :, 1]))
                return line,

            ani = animation.FuncAnimation(
                fig, animate, interval=round(1 / self.mocap_freq * 1000), blit=True, frames=n_time)

            if save:
                ani.save('RelativeTraj.mp4', writer="ffmpeg")
        # endregion

        # region ------------Regular Plot-------------
        else:
            plt.plot(t_h_stack[:, 0], t_h_stack[:, 1], **kwargs)
            plt.axis('equal')
            plt.xlabel("Hip Forward (m)", fontsize=16)
            plt.ylabel("Hip Left (m)", fontsize=16)
        # endregion
        plt.show()

    def show_global(self, save=False, frame_slice: Optional[slice] = None,
                    ref_kwargs=None, traj_kwargs=None):
        import matplotlib.animation as animation
        try_seaborn()

        if frame_slice is None:
            frame_slice = slice(0, self.dict_features['t_h'].shape[0])
        if ref_kwargs is None:
            ref_kwargs = {}
        if traj_kwargs is None:
            traj_kwargs = {}

        t_h = self.dict_features['t_h'][frame_slice]
        t_h = np.append(np.zeros((t_h.shape[0], 1, t_h.shape[2])), t_h, axis=1)
        ref = self.dict_mocap['HipOrientation'][frame_slice, :2, :2]

        comp_rotated_traj = (t_h[:, :, 0] + t_h[:, :, 1] * 1j) * (ref[:, 1, 0] + ref[:, 1, 1] * 1j)[:, None]
        t_h_global = np.dstack((comp_rotated_traj.real, comp_rotated_traj.imag))
        v_g_l = self.dict_features['v_g_l'][frame_slice]
        comp_rotated_v = (v_g_l[:, 0] + v_g_l[:, 1] * 1j) * (ref[:, 1, 0] + ref[:, 1, 1] * 1j)
        v_g_global = np.vstack((comp_rotated_v.imag, -comp_rotated_v.real)).T*10

        d_h = self.dict_features['d_h'][frame_slice]
        comp_rotated_direc = (d_h[:, :, 0] + d_h[:, :, 1] * 1j) * (ref[:, 1, 0] + ref[:, 1, 1] * 1j)[:, None]
        d_h_global = np.dstack((comp_rotated_direc.real, comp_rotated_direc.imag))

        root_pos = self.dict_mocap['VHIP'][frame_slice, :2]
        fig, ax = plt.subplots(figsize=(5, 10))

        line_i, = ax.plot(np.append(0., ref[0, 0, 0]) * 0.2 + root_pos[0, 0],
                          np.append(0., ref[0, 0, 1]) * 0.2 + root_pos[0, 1],
                          label='Hip Right', **ref_kwargs)

        line_j, = ax.plot(np.append(0., ref[0, 1, 0]) * 0.2 + root_pos[0, 0],
                          np.append(0., ref[0, 1, 1]) * 0.2 + root_pos[0, 1],
                          label='Hip Front', **ref_kwargs)

        line_t, = ax.plot(t_h_global[0, :, 0] + root_pos[0, 0],
                          t_h_global[0, :, 1] + root_pos[0, 1],
                          label='Trajectory', **traj_kwargs)

        line_v, = ax.plot(np.append(0., v_g_global[0, 0]) + root_pos[0, 0],
                         np.append(0., v_g_global[0, 1]) + root_pos[0, 1],
                         label='Velocity', **traj_kwargs)



        ax.legend()
        lines = [line_i, line_j, line_t, line_v]
        ax.set_aspect('equal')
        ax.set_xlabel('Global X (m)')
        ax.set_ylabel('Global Y (m)')
        plt.tight_layout()

        ax.set_xlim([np.min([root_pos[:, 0]]) - 1, np.max([root_pos[:, 0]]) + 1])
        ax.set_ylim([np.min([root_pos[:, 1]]) - 1, np.max([root_pos[:, 1]]) + 1])

        def animate(i_frame):
            line_i.set_data(np.append(0., ref[i_frame, 0, 0]) * 0.3 + root_pos[i_frame, 0],
                            np.append(0., ref[i_frame, 0, 1]) * 0.3 + root_pos[i_frame, 1])

            line_j.set_data(np.append(0., ref[i_frame, 1, 0]) * 0.3 + root_pos[i_frame, 0],
                            np.append(0., ref[i_frame, 1, 1]) * 0.3 + root_pos[i_frame, 1])

            line_t.set_data(t_h_global[i_frame, :, 0] + root_pos[i_frame, 0],
                            t_h_global[i_frame, :, 1] + root_pos[i_frame, 1], )
            line_v.set_data(np.append(0., v_g_global[i_frame, 0]) + root_pos[i_frame, 0],
                            np.append(0., v_g_global[i_frame, 1]) + root_pos[i_frame, 1],)

            ax.patches = []
            arrow_x = root_pos[i_frame, 0] + t_h_global[i_frame, 1:, 0]
            arrow_y = root_pos[i_frame, 1] + t_h_global[i_frame, 1:, 1]
            arrow_dx = d_h_global[i_frame, :, 0] / 10
            arrow_dy = d_h_global[i_frame, :, 1] / 10

            for (cur_x, cur_y, cur_dx, cur_dy) in zip(arrow_x, arrow_y, arrow_dx, arrow_dy):
                arr = plt.arrow(cur_x, cur_y, cur_dx, cur_dy,
                                width=0.005, ec=(0.2980392156862745, 0.4470588235294118, 0.6901960784313725), alpha=0.7,
                                head_width=0.03, overhang=0.5)
                ax.add_patch(arr)

            return lines+ax.patches

        ani = animation.FuncAnimation(
            fig, animate, interval=round(1 / self.mocap_freq * 1000), blit=True, frames=t_h.shape[0])

        if save:
            ani.save(os.path.splitext(os.path.basename(self.mocap_file))[0] + 'GlobalTraj.mp4',
                     writer="ffmpeg", dpi=300, bitrate=-1)

        plt.show()
