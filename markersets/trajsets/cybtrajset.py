import sys

from markersets.trajsets.trajset import TrajSet
from utility.C3D import C3DServer
import numpy as np
from markersets.trajsets.traj_utils import *
from typing import Optional
import markersets.jcs


class CybTrajSet(TrajSet):

    def __init__(self, mocap_file, emg_file=None,
                 **kwargs):

        super(CybTrajSet, self).__init__(mocap_file=mocap_file, emg_file=emg_file, **kwargs)
        self.list_marker = ['SACR',
                            'RASI', 'LASI',
                            'RANK', 'LANK',
                            'RANM', 'LANM']

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
        self.dict_mocap['LVAN'] = (self.dict_mocap['LANK'] + self.dict_mocap['LANM']) / 2
        self.dict_mocap['RVAN'] = (self.dict_mocap['RANK'] + self.dict_mocap['RANM']) / 2

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


def main():
    markersets.jcs.dir_proc(CybTrajSet, sys.argv[1],
                            save_dir_path=sys.argv[2],
                            work_fun=traj_worker,
                            extension='.c3d',
                            transpose=False,
                            mocap_slice=slice(0, None),
                            nan_crop=False)
    pass


if __name__ == '__main__':
    main()