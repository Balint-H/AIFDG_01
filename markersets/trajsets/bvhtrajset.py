from markersets.trajsets.trajset import *
from bvh import *
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys

from markersets.jcs import dir_proc


class BvhTrajSet(TrajSet):
    """
    Assuming single root in bvh

    Reference:
    Maddock, M.M.S. and Maddock, S., 2001. Motion capture file formats explained. Department of Computer Science,
    University of Sheffield.
    """

    def __init__(self, mocap_file, emg_file=None,
                 **kwargs):

        super(BvhTrajSet, self).__init__(mocap_file=mocap_file, emg_file=emg_file, **kwargs)
        self.list_marker = ['Hips',
                            'RightFoot', 'LeftFoot']

        self.list_emg = []

    def se_proc_traj(self, load_emg=False):
        if load_emg:
            raise NotImplementedError("No EMG loading data for bvh yet")
        super(BvhTrajSet, self).se_proc_traj(load_emg=load_emg)

    def get_emg_data(self, file_reader=None):
        raise NotImplementedError()

    def se_load_mocap(self, load_emg=False):
        with open(self.mocap_file) as f:
            mocap = Bvh(f.read())

        joint_names = np.array(mocap.get_joints_names())
        joint_names = joint_names[[np.nonzero([cur_j in j for j in joint_names])[0][0]
                                   for cur_j in self.list_marker]]
        hip_channels = np.array(mocap.joint_channels(joint_names[0]))  # Y up, and weird rot orders !
        hip_translation_channels = hip_channels[['POSITION' in cur_ch.upper() for cur_ch in hip_channels]]
        hip_rotation_channels = hip_channels[['ROTATION' in cur_ch.upper() for cur_ch in hip_channels]]

        frames = np.array(mocap.frames, dtype=float)

        def get_frames(j_in, chs_in):
            joint_index = mocap.get_joint_channels_index(j_in)
            channel_indices = np.array([mocap.get_joint_channel_index(j_in, ch) for ch in chs_in])
            return frames[:, joint_index+channel_indices]

        self.dict_mocap['VHIP'] = \
            np.array(get_frames(joint_names[0], hip_translation_channels))[:, [2, 0, 1]] * 0.01

        intrinsic_rot_labels = ''.join(['X' if 'X' in cur_label else 'Y' if 'Y' in cur_label else 'Z'
                                        for cur_label in hip_rotation_channels])

        if not (intrinsic_rot_labels.count('Z') == 1
                and intrinsic_rot_labels.count('X') == 1
                and len(intrinsic_rot_labels) == 3
                and len(hip_translation_channels) == 3):
            raise Exception('Unexpected formatting of bvh file, missing channel labels/ incorrect order')

        # Hip orientation, while moving the Up axis (Y) to the 3rd index (XYZ -> XZY), and rolling the time/R matrix
        # so dimensions are (time, axis [i,j,k][Right, Front, Up], dimension [global x,z,y])
        rot_mat = R.from_euler(intrinsic_rot_labels,
                               get_frames(joint_names[0], hip_rotation_channels),
                               degrees=True).as_matrix()

        hip_rotation = np.zeros_like(rot_mat)
        hip_rotation[:, 1, :2] = rot_mat[:, [2, 0], 1]
        hip_rotation[:, 1, :2] = \
            hip_rotation[:, 1, :2]/np.linalg.norm(hip_rotation[:, 1, :2], axis=1)[:, None]

        hip_rotation[:, 2, :] = np.array([0, 0, 1])[None, :]

        hip_rotation[:, 0, 0] = hip_rotation[:, 1, 1]
        hip_rotation[:, 0, 1] = -hip_rotation[:, 1, 0]

        self.dict_mocap['HipOrientation'] = hip_rotation

        # 4x4 unscaled rigid-body transformation matrix in the form M = TR, T: translation matrix
        # Assuming same rotation order for all joints!
        def M_local(joint: BvhNode):
            M = np.empty((mocap.nframes, 4, 4))
            M[:, :3, :3] = \
                R.from_euler(intrinsic_rot_labels,
                             get_frames(joint.name, hip_rotation_channels), degrees=True).as_matrix()
            # Currently in (time, global dim, axis) format, will be converted at the end
            if 'ROOT' in joint.value:
                M[:, :3, 3] = get_frames(joint_names[0], hip_translation_channels)
            else:
                M[:, :3, 3] = joint['OFFSET']
            M[:, 3, :] = [0, 0, 0, 1]
            return M

        def M_global(joint: BvhNode):
            if 'ROOT' in joint.value:
                return M_local(joint)
            else:
                return np.einsum('tij, tjk -> tik', M_global(joint.parent), M_local(joint))

        self.dict_mocap['LVAN'] = M_global(
            mocap.get_joint(joint_names[self.list_marker.index('LeftFoot')]))[:, [2, 0, 1], 3] * 0.01
        self.dict_mocap['RVAN'] = M_global(
            mocap.get_joint(joint_names[self.list_marker.index('RightFoot')]))[:, [2, 0, 1], 3] * 0.01
        self.mocap_freq = np.round(1./mocap.frame_time)
        return

    def se_marker_preproc(self):
        # self.non_nan_region = nan_region_find_dict(self.dict_mocap)
        pass


if __name__ == '__main__':

    # dir_proc(BvhTrajSet, sys.argv[1],
    #          save_dir_path=sys.argv[2],
    #          work_fun=traj_worker,
    #          extension='.bvh',
    #          transpose=True,
    #          mocap_slice=slice(0, None),
    #          nan_crop=True)
    ts = BvhTrajSet(sys.argv[1],
                    horizon_vector=tuple(range(20, 61, 20)))
    ts.se_proc_traj()
    ts.show_global(save=False, frame_slice=slice(1204, 1205), traj_kwargs={'alpha': 0.3})

    # C:\Users\hbkm9\Documents\Projects\CYB\La_Forge_Dataset\bvh\
    print('done!')