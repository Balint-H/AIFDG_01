"""
The following are specific to the NDMSTrajSet:
- Only extracts trajectory from the NDMS01 marker set. (As such show_global is also changed)
- Also reads the concurrent sEMG signals.
- Doesn't flatten the trajectory X and Y dimensions. X and Y trajectories are stored in separate, NxH dimensional
  arrays.
"""

import sys

from markersets.trajsets.trajset import TrajSet
from utility.C3D import C3DServer
import numpy as np
from markersets.trajsets.traj_utils import *
from typing import Optional
import markersets.jcs
import matplotlib.pyplot as plt


class NDMSTrajSet(TrajSet):

    def __init__(self, mocap_file, emg_file=None,
                 **kwargs):

        super(NDMSTrajSet, self).__init__(mocap_file=mocap_file, emg_file=emg_file, **kwargs)
        self.list_marker = ['SACR',
                            'RASI', 'LASI']

        self.list_emg = ["Sensor {}.EMG{}".format(i, i) for i in range(1, 9)]

    def se_marker_preproc(self):
        """
        Adds a virtual marker at the center of the hip markers, giving the SACR marker double the weight.
        """

        for key in self.dict_mocap.keys():
            self.dict_mocap[key][np.sum(self.dict_mocap[key], 1) == 0, :] = np.nan

        self.non_nan_region = nan_region_find_dict(self.dict_mocap)
        self.dict_mocap['VHIP'] = (2 * self.dict_mocap['SACR'] + self.dict_mocap['RASI'] + self.dict_mocap['LASI']) \
                                  / 4

        i_vect = (self.dict_mocap['RASI'] - self.dict_mocap['LASI'])
        i_vect[:, 2] = 0
        i_vect = i_vect / np.linalg.norm(i_vect, axis=1)[:, None]
        # k_vect = np.cross(self.dict_marker['RASI'] - self.dict_marker['SACR'],
        #                   self.dict_marker['LASI'] - self.dict_marker['SACR'], axis=1)
        # k_vect = k_vect / np.linalg.norm(k_vect, axis=1)[:, None]
        k_vect = np.array([[0, 0, 1]] * i_vect.shape[0])
        j_vect = np.cross(k_vect, i_vect)

        # Always use vertical axis as up, that will be the same always

        # (time, axis [i,j,k][Right, Front, Up], dimension [global x,y,z])
        self.dict_mocap['HipOrientation'] = np.rollaxis(np.dstack((i_vect, j_vect, k_vect)), 2, 1)
        return

    def get_emg_data(self, file_reader: Optional[C3DServer] = None):
        """
        Returns the EMG data contained in the c3d file of the object, coupled with the frequency in a tuple
        :type file_reader:      Optional[C3DServer]
        :param file_reader:     Optional C3DServer that contains EMG "analog" data. If None, attempts to instantiate one
                            from the self.mocap_file path. Use to not repeatedly open/close the c3d file.
        :return: (EMG dictionary, EMG frame rate)
        """
        if self.dict_emg and self.emg_freq:
            return self.dict_emg, self.emg_freq
        if file_reader is None:
            with C3DServer() as file_reader:
                file_reader.open_c3d(self.mocap_file)
                return file_reader.get_analog_dict(self.list_emg), file_reader.get_analog_frame_rate()
        return file_reader.get_analog_dict(self.list_emg), file_reader.get_analog_frame_rate()

    def se_proc_traj(self, load_emg=True, extract_motion_match_features=False):
        super(NDMSTrajSet, self).se_proc_traj(load_emg=load_emg, extract_motion_match_features=False)

    def get_dict_out(self, mocap_slice=None, emg_slice=None, transpose=False):
        if mocap_slice is None:
            dict_out = {'t_h_x': self.dict_features['t_h'][:, :, 0].tolist(),
                        't_h_y': self.dict_features['t_h'][:, :, 1].tolist()} if not transpose else \
                       {'t_h_x': self.dict_features['t_h'][:, :, 0].T.tolist(),
                        't_h_y': self.dict_features['t_h'][:, :, 1].T.tolist()}

        else:
            dict_out = {'t_h_x': self.dict_features['t_h'][mocap_slice, :, 0].tolist(),
                        't_h_y': self.dict_features['t_h'][mocap_slice, :, 1].tolist()} if not transpose else \
                       {'t_h_x': self.dict_features['t_h'][mocap_slice, :, 0].T.tolist(),
                        't_h_y': self.dict_features['t_h'][mocap_slice, :, 1].T.tolist()}

        if emg_slice is None:
            dict_out['EMG'] = {k: i.tolist() for k, i in self.dict_emg.items()}
        else:
            dict_out['EMG'] = {k: i[emg_slice].tolist() for k, i in self.dict_emg.items()}
        dict_out['Framerate'] = self.mocap_freq
        dict_out['Sampling Frequency'] = self.emg_freq
        return dict_out

    def show_global(self, save=False, frame_slice: Optional[slice] = None,
                    ref_kwargs=None, traj_kwargs=None):
        import matplotlib.animation as animation
        try_seaborn()

        if frame_slice is None:
            frame_slice = slice(self.non_nan_region[0], self.non_nan_region[1])
        if ref_kwargs is None:
            ref_kwargs = {}
        if traj_kwargs is None:
            traj_kwargs = {}

        t_h = self.dict_features['t_h'][frame_slice]
        t_h = np.append(np.zeros((t_h.shape[0], 1, t_h.shape[2])), t_h, axis=1)
        ref = self.dict_mocap['HipOrientation'][frame_slice, :2, :2]

        comp_rotated_traj = (t_h[:, :, 0] + t_h[:, :, 1] * 1j) * (ref[:, 1, 0] + ref[:, 1, 1] * 1j)[:, None]
        t_h_global = np.dstack((comp_rotated_traj.real, comp_rotated_traj.imag))

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



        ax.legend()
        lines = [line_i, line_j, line_t]
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


def main():

    n_pool = 12
    if sys.gettrace() is not None:
        n_pool = 1

    markersets.jcs.dir_proc(NDMSTrajSet, sys.argv[1],
                            save_dir_path=sys.argv[2],
                            work_fun=traj_worker,
                            extension='.c3d',
                            transpose=False,
                            mocap_slice=slice(0, None),
                            nan_crop=False,
                            horizon_vector=tuple(range(1,61,1)),
                            n_pool=n_pool,)
    pass


if __name__ == '__main__':
    main()