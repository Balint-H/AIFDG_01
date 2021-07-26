"""
DataGenerator classes for tensorflow.keras, feeding network training with windows of EMG data and target features

Designed with joint angles as target features in mind, docstrings will often refer to angles where general features
could also be applicable.

Reads from .json of dictionaries or .pickle files of MarkerSet objects  with the labeled joint angle timeseries and the
EMG timeseries. EMG can either be a dict itself (with source muscle group names as keys) or a list of lists.
Leading and trailing sections of NaNs allowed in joint angle data, but there shouldn't be any inside the timeseries.
Negative delay is not supported.

TODO: Untangle spaghetti code; could be improved by making methods into functions instead, abstracting the target
    variable names (from 'angle' to 'target' in TCNGenerator, and 'classifications' in its child classes)

TODO: Memory requirement could be reduced if stride is already accounted for when storing angle data. This isn't
    straightforward for EMG, since need full resolution for the windows. However, could be used in case stride
    larger than window size (uncommon). Or if stride is an integer multiple of timestep.
"""
from typing import List, Any

import numpy as np
import json
from tensorflow.keras.utils import Sequence, to_categorical
import copy
import os
import pickle
import warnings

from markersets.jcs import KinematicSet


def _raise(ex):
    raise ex


def _pass(x, **kwargs):
    return x


def read_pickle(file_in):
    ks: KinematicSet = pickle.load(file_in)
    dict_out = ks.dict_joint
    dict_out['EMG'] = ks.dict_emg
    dict_out['Framerate'] = ks.mocap_freq
    dict_out['Sampling Frequency'] = ks.emg_freq
    return dict_out


class TCNDataGenerator(Sequence):
    emg_data: List[np.ndarray] # [file][muscle, time]
    feat_data: List[np.ndarray] # [file][joint, time, angle]

    # Generates data for Keras
    def __init__(self, data_dir, file_names, window_size=32, stride=1, batch_size=32, freq_factor=20, delay=1,
                 dims=(0,), shuffle=True, feature_names=('LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle'),
                 preproc=_pass, feat_preproc=_pass, ppkwargs=None, gap_windows=None, channel_mask=None, time_step=1,
                 load_method=json.load):
        """
        Constructor for the data generator, which also immediately loads in the files given. Recommend specifying all
        these kwargs in a separate dict then using **

        :param data_dir: Path for where the files are located in
        :param file_names: List of the names of JSON files containing dictionaries of joint kinematics and EMG for a
                           given trial
        :param window_size: Segment the EMG into a window of this size. This many samples are used to regress to a
                            single joint angle
        :param stride: The difference in samples between two windows. Overlap = window_size - stride
        :param batch_size: Supply how many windows to the network in each batch
        :param freq_factor: Frequency factor between mocap and EMG. f_emg/f_mocap. Used in upsampling the angle data
                            since stride can be less than window_size, needing interpolation
        :param delay: How far ahead are we predicting? In EMG samples
        :param dims: Which angles are we regressing to? flexion = 0, abduction = 1, rotation = 2, supplied in a tuple
        :param shuffle: Do we shuffle the data between epochs?
        :param feature_names: List of identifying strings also used in JSON files containing data. Be consistent.
        :param preproc: Function called when loading the EMG, optionally preprocessing it
        :param feat_preproc: Function called when loading the angle data, optionally preprocessing it
        :param ppkwargs: kwargs supplied to preproc when preprocessing EMG
        :param gap_windows: The generated input can have a gap in it (two smaller, noncontinous windows) (start, end) of
                            gap
        :param channel_mask: Exclude idx or key of EMG channel to ignore
        :param time_step: Effectively reduce sampling rate of EMG by skipping this many -1 samples between them
        :param load_method: How are dictionaries loaded? If not JSON is used, you can supply your own custom function
        """
        self.k_idx = list()

        self.batch_size = batch_size
        self.delay = delay
        self.data_dir = data_dir
        self.dims = dims  # Spacial dims of angles/ Dims of features
        self.shuffle = shuffle

        self.file_names = sorted(file_names)

        self.window_size = window_size
        self.stride = stride
        self.feature_names = feature_names
        self.freq_factor = freq_factor
        self.gap_windows = gap_windows
        self.channel_mask = channel_mask
        self.time_step = time_step

        self.preproc = preproc
        self.feat_preproc = feat_preproc
        if ppkwargs is None:
            self.ppkwargs = {}
        else:
            self.ppkwargs = ppkwargs

        self.n_windows = 0
        self.window_index_heads = list()
        self.emg_data = list()
        self.feat_data = list()
        self.load_method = load_method

        self.load_files()
        if self.emg_data:
            self.n_channels = len(self.emg_data[0])
        if self.feat_data:
            self.n_feats = len(self.feat_data[0])
        self.window_idx = np.arange(self.n_windows)

        self.on_epoch_end()

    def load_features(self, dict_data):
        """
        Method updating *this* object's angle_data fields from the supplied dictionary (single recording), appending the
        incoming joint kinematics to the list of all recordings.

        :param dict_data: Input dictionary read from JSON file produced by c3dcyb.py, jcs.py or equivalent (i.e. have
                          keys from self.joint_names)
        """
        features = list()
        emg_length = len(list(dict_data['EMG'].values())[0]) \
            if type(dict_data['EMG']) is dict else len(dict_data['EMG'][0])
        for feat in self.feature_names:
            if self.freq_factor > 0:
                xp = np.arange(len(dict_data[feat])) * self.freq_factor
                x = np.arange(0, len(dict_data[feat]) * self.freq_factor + self.delay)
            else:
                xp = np.linspace(0, emg_length, len(dict_data[feat])+1)[:-1]  # Assuming 0-0 overlap
                x = np.linspace(0, emg_length - 1 + self.delay, emg_length + self.delay)

            def interp_f(data):
                return np.interp(x, xp, data)

            upsampled = \
                np.apply_along_axis(interp_f, 0, self.feat_preproc(np.array(dict_data[feat])[:, list(self.dims)]))
            features.append(upsampled)
        self.feat_data.append(np.array(features))

    def load_emg(self, dict_data):
        """
        Method updating *this* object's emg_data fields from the supplied dictionary, appending the incoming EMG data.

        :param dict_data: Input dictionary read from JSON file produced by c3dcyb.py, jcs.py or equivalent (i.e. have
                          "EMG" key)
        """
        if self.channel_mask is None:
            if type(dict_data["EMG"]) is dict:
                self.emg_data.append(self.preproc(np.array(list(dict_data["EMG"].values())), **self.ppkwargs))
            else:
                self.emg_data.append(self.preproc(np.array(dict_data["EMG"]), **self.ppkwargs))
        else:
            self.emg_data.append(
                self.preproc(np.array([dict_data["EMG"][i] for i in self.channel_mask]), **self.ppkwargs))

    def nan_crop_file(self, file_id):
        """
        Checks if data needs cropping to avoid outputting NaN values in angle data. Leaves them in if delay skips over
        all entire region (alternatively, delay could be altered). Accounts for leading and trailing NaN regions;
        internal NaN regions should have been interpolated out during angle extraction, and issues warning if they are
        present. Automatically updates to cropped data for the object, for all joint angles, cropped to the narrowest
        overlapping non-NaN-region found in joints.

        NaN regions represent missing joint angle trajectory segments caused by obscured markers in the recording.

        This step needs to be performed before updating the window indices, as this alters the number of possible
        windows.

        TODO: Handle internal NaN regions.

        :param file_id: File index to crop (index in self.file_names). -1 for most recent file.
        :return: self.angle_data[file_id), self.emg_data[file_id]: After cropping
        """
        #

        # edges holds the index of samples where isnan of angles is different from the previous one
        # detects start and end of NaN regions. Doesn't include edge if at start or end, we check for those cases
        # manually
        edges = map(lambda x: np.where(np.diff(x) != 0)[0] + 1,
                    np.isnan(np.sum(self.feat_data[file_id], axis=2)).astype(int))

        # Default values (in case there are no NaN regions).
        # These denote the index where leading regions end and trailing regions start
        leading = [0]
        trailing = [self.feat_data[file_id].shape[1]]  # Calculated as the positive distance from start
        for feat, edge in zip(self.feat_data[file_id], edges):
            if len(edge) - np.any(np.isnan(feat[0])) - np.any(np.isnan(feat[-1])) > 0:
                # All internal NaN regions should be interpolated!
                warnings.warn('NaN value detected inside time-series!')
            elif len(edge) - np.any(np.isnan(feat[0])) - np.any(np.isnan(feat[-1])) < 0:
                raise Exception('Incorrect NaN section detection...')  # Means edges was incorrectly calculated
            if np.any(np.isnan(feat[0])):
                leading.append(edge[0])
            if np.any(np.isnan(feat[-1])):
                trailing.append(edge[-1])

        # Now crop down samples if NaN region is not skipped by delay, assuming delay>0 (hence always crop trailing)
        if self.delay < max(leading):
            self.feat_data[file_id] = self.feat_data[file_id][:, max(leading) - self.delay:, :]
            self.emg_data[file_id] = self.emg_data[file_id][:, max(leading) - self.delay:]
        self.feat_data[file_id] = self.feat_data[file_id][:, :min(trailing), :]
        self.emg_data[file_id] = self.emg_data[file_id][:, :min(trailing)]
        return self.feat_data[file_id], self.emg_data[file_id]

    def update_window_index(self):
        """
        Method appending which window indices the most recent file's start and end represent in the grand total

        TODO: make internal, perhaps make it harder to be called unless file has actually been loaded, but indices
              not yet updated
        """
        self.window_index_heads.append((self.n_windows,
                                        self.n_windows +
                                        int((len(self.emg_data[-1][0]) - self.window_size + 1) / self.stride)))
        self.n_windows = self.window_index_heads[-1][1]

    def load_files(self):
        """
        Method that takes whatever list of files is currently in file_names, and loads them in, appending their angle
        and EMG data as separate rows to angle_data and emg_data (i.e. the first index of these lists corresponds to
        which file they came from)
        """
        self.emg_data = list()
        self.window_index_heads = list()
        self.feat_data = list()
        for file in self.file_names:
            with open(os.path.join(self.data_dir, file), 'rb') as cur_file:
                dict_data = self.load_method(cur_file)
                self.load_features(dict_data)
                self.load_emg(dict_data)
                self.nan_crop_file(-1)
                self.update_window_index()

        self.on_epoch_end()
        return

    def force_unwrap(self):  # only for first angle for now!
        """
        There was a time when I thought angles weren't correctly unwrapping during earlier angle extraction and added
        this in an attempt to unwrap again. It turned out that the error was even earlier, two marker labels flipped
        during mocap labelling. Remains here as I can see myself needing this in the future, or at least grim reminder
        of the hours wasted trying to debug code working OK on faulty data. Also makes sure that unwrapping has resolved
        to the same multiple of 2 pi.
        """
        if not self.freq_factor > 0:
            raise Exception('force_unwrap needs constant freq_factor!')
        angle_means = list()
        for i in range(self.n_feats):
            angle_means.append(np.mean(self.feat_data[0][i][::self.freq_factor, 0]))

        for f in range(1, len(self.feat_data)):
            for a in range(self.n_feats):
                self.feat_data[f][a][:, 0] = np.unwrap(self.feat_data[f][a][:, 0] * 4) / 4
                if np.mean(np.mean(self.feat_data[f][a][::self.freq_factor, 0])) < angle_means[a] - 3:
                    self.feat_data[f][a][:, 0] = self.feat_data[f][a][:, 0] + np.pi
                elif np.mean(np.mean(self.feat_data[f][a][::self.freq_factor, 0])) > angle_means[a] + 3:
                    self.feat_data[f][a][:, 0] = self.feat_data[f][a][:, 0] - np.pi

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.ceil(len(self.window_idx) / self.batch_size))

    def __getitem__(self, batch_index):
        """
        Generate one batch of data
        """

        # Generate indexes of the batch
        cur_indexes = self.window_idx[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        # Generate data
        X, Y = self.data_generation(cur_indexes)
        return X, Y

    def data_generation(self, cur_indexes):
        head_tails = self.window_index_heads  # Starting and ending index of windows for each file

        # Find which files the window indices are found in, and zip them together
        ids = [(file_id, cur_idx - head_tails[file_id][0])
               for cur_idx in cur_indexes for file_id, head_tail in enumerate(head_tails)
               if head_tail[0] <= cur_idx < head_tail[1]]

        X = np.array(
            [self.emg_data[file_id][:, win_id * self.stride:
                                       win_id * self.stride + self.window_size:self.time_step].transpose()
             for file_id, win_id in ids])
        Y = np.array(
            [self.feat_data[file_id][:, win_id * self.stride + self.delay, :].flatten()
             for file_id, win_id in ids])

        # If you want a gap in the input (i.e. output two smaller, noncontinuous windows as 2 inputs, do it here)
        if self.gap_windows is not None:
            return [X[:, :self.gap_windows[0], :], X[:, -self.gap_windows[1]:, :]], Y
        return X, Y

    def on_epoch_end(self):
        """
        Shuffles indexes after each epoch
        """

        if not self.k_idx:
            self.window_idx = np.arange(self.n_windows)
        if self.shuffle:
            np.random.shuffle(self.window_idx)

    def save(self, path, unload=True):
        """
        The datagenerator can be saved to document how the training data was structured, and which files were used.
        Serializes the generator, without the actual data preferably.
        Then you can just call load_data once you unpickle.
        """
        import pickle
        if unload:
            self.emg_data = list()
            self.feat_data = list()
            self.n_windows = 0
            self.window_index_heads = list()
            self.window_idx = np.array([])
        with open(path, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def validation_split(self, k=2, file_split=True, file_shuffle=True):
        """
        Used for k-fold crossvalidation. Splits window indices into equal (ish) chunks. Preferably do it so no fold
        has data from the same trial.

        :param k: How many folds?
        :param file_split: Do the split based on files?
        :param file_shuffle: Shuffle the files first?
        """
        if not file_split:
            self.k_idx = np.array_split(self.window_idx, k)
        else:
            file_idx = np.array(self.window_index_heads)
            if file_shuffle:
                np.random.shuffle(file_idx)
            self.k_idx = np.array_split(file_idx, k)
            for i in range(len(self.k_idx)):
                buff = np.array([], dtype=int)
                for j in range(len(self.k_idx[i])):
                    buff = np.append(buff, np.arange(*self.k_idx[i][j]))
                self.k_idx[i] = buff

    def get_k(self, cur_k, k=2, **kwargs):
        """
        Used in k-fold crossvalidation.
        From the master list of window indices for all folds (in k_idx) get a train - test split (i.e. which fold it is)
        Makes this generator only use the training indices of this fold in the upcoming training.

        :param cur_k: Fold idx
        :param k: In case k_idx has not been generated, do it now, with this many folds
        :param kwargs: kwargs to pass into the split
        :return: a shallow copy of this generator (i.e. the data is not duplicated), only difference made is the
        window_idx (i.e. which windows are used during data generation)
        """
        if not self.k_idx:
            self.validation_split(k, **kwargs)

        ks = list(self.k_idx)
        val_idx = ks.pop(cur_k)
        train_idx = np.concatenate(ks)
        valid_gen = copy.copy(self)
        valid_gen.window_idx = val_idx
        self.window_idx = train_idx
        self.on_epoch_end()
        valid_gen.on_epoch_end()
        return valid_gen

    def show(self):
        """
        Quickly plot what will this generator feed into the model if used now.
        """
        import matplotlib.pyplot as plt
        from utility.plot_utility import plot_columns
        _, y0 = self.data_generation(np.sort(self.window_idx))
        plot_columns(y0, titles=self.feature_names, sb=True, sharex=True)
        plt.suptitle('Joint Angles Over Time')
        plot_columns(np.concatenate(self.emg_data, axis=1).T[::self.stride, :], sb=True, sharex=True,
                     titles=self.channel_mask
                     if self.channel_mask is not None else list(range(self.emg_data[0].shape[0])))
        plt.suptitle('Muscle sEMG Signals')
        plt.show()


class TCNClassGenerator(TCNDataGenerator):
    def __init__(self, class_enum=('Walk', 'Sit', 'Stair'), **kwargs):
        self.class_enum = class_enum
        super(TCNClassGenerator, self).__init__(**kwargs)

    def load_files(self):
        self.emg_data = list()
        self.angle_data = list()
        self.window_index_heads = list()
        for file in self.file_names:
            with open(os.path.join(self.data_dir, file)) as json_file:
                dict_data = json.load(json_file)
                self.load_emg(dict_data)
                self.update_window_index()
        return

    def data_generation(self, cur_indexes):
        head_tails = self.window_index_heads
        ids = [(file_id, cur_idx - head_tails[file_id][0])
               for cur_idx in cur_indexes for file_id, head_tail in enumerate(head_tails)
               if head_tail[0] <= cur_idx < head_tail[1]]

        X = np.array(
            [self.emg_data[file_id][:,
             win_id * self.stride:win_id * self.stride + self.window_size:self.time_step].transpose()
             for file_id, win_id in ids])
        Y = to_categorical([[i for i, e in enumerate(self.class_enum) if e in self.file_names[file_id]][0]
                            for file_id, _ in ids], num_classes=len(self.class_enum))
        if self.gap_windows is not None:
            return [X[:, :self.gap_windows[0], :], X[:, -self.gap_windows[1]:, :]], Y
        return X, Y


class StepTCNGenerator(TCNDataGenerator):
    def __init__(self, **kwargs):
        self.step_classes = list()
        super(StepTCNGenerator, self).__init__(**kwargs)

    def load_files(self):
        self.emg_data = list()
        self.angle_data = list()
        self.window_index_heads = list()
        for file in self.file_names:
            with open(os.path.join(self.data_dir, file)) as json_file:
                dict_data = json.load(json_file)
                self.load_emg(dict_data)
                self.update_window_index()

                step_class = np.repeat(dict_data["step_class"], self.freq_factor)
                step_class = np.append(step_class, [step_class[-1]] * max(self.delay - self.window_size + 1, 0))
                self.step_classes.append(step_class)
        return

    def data_generation(self, cur_indexes):
        head_tails = self.window_index_heads
        ids = [(file_id, cur_idx - head_tails[file_id][0])
               for cur_idx in cur_indexes for file_id, head_tail in enumerate(head_tails)
               if head_tail[0] <= cur_idx < head_tail[1]]

        X = np.array(
            [self.emg_data[file_id][:,
             win_id * self.stride:win_id * self.stride + self.window_size:self.time_step].transpose()
             for file_id, win_id in ids])

        Y = to_categorical(
            [self.step_classes[file_id][win_id * self.stride + self.delay] for file_id, win_id in ids], num_classes=3)
        if self.gap_windows is not None:
            return [X[:, :self.gap_windows[0], :], X[:, -self.gap_windows[1]:, :]], Y
        return X, Y


class ParamTCNGenerator(TCNDataGenerator):
    def __init__(self, params=("step_heights", "stride_lengths", "step_speed"), **kwargs):
        self.step_params = list()
        self.params = params
        super(ParamTCNGenerator, self).__init__(**kwargs)

    def load_files(self):
        self.emg_data = list()
        self.angle_data = list()
        self.window_index_heads = list()
        for file in self.file_names:
            with open(os.path.join(self.data_dir, file)) as json_file:
                dict_data = json.load(json_file)
                self.load_emg(dict_data)
                self.update_window_index()
                cur_params = list()
                for p in self.params:
                    cur_value = np.array(dict_data[p])
                    cur_value[:] = 1
                    cur_value[np.array(dict_data[p]) == 0] = 0
                    cur_value[np.array(dict_data[p]) > np.abs(np.max(dict_data[p])) * 0.8] = 2
                    cur_params.append(np.atleast_2d(np.repeat(cur_value, self.freq_factor)))

                step_param = np.vstack(cur_params)
                step_param = np.hstack((step_param,
                                        np.tile(np.atleast_2d(step_param[:, -1]).T,
                                                (1, max(self.delay - self.window_size + 1, 0)))))
                self.step_params.append(self.feat_preproc(step_param))
        return

    def data_generation(self, cur_indexes):
        head_tails = self.window_index_heads
        ids = [(file_id, cur_idx - head_tails[file_id][0])
               for cur_idx in cur_indexes for file_id, head_tail in enumerate(head_tails)
               if head_tail[0] <= cur_idx < head_tail[1]]
        X = np.array(
            [self.emg_data[file_id][:,
             win_id * self.stride:win_id * self.stride + self.window_size:self.time_step].transpose()
             for file_id, win_id in ids])
        Y = to_categorical(
            [self.step_params[file_id][:, win_id * self.stride + self.delay] for file_id, win_id in ids], num_classes=3)
        if self.gap_windows is not None:
            return [X[:, :self.gap_windows[0], :], X[:, -self.gap_windows[1]:, :]], Y
        return X, Y


if __name__ == '__main__':
    from data_gen.preproc import norm_emg
    gen_params = {
        'data_dir': '.',
        ###################
        'window_size': 1000,
        'delay': 1000,
        'gap_windows': None,
        ###################
        'stride': 20,
        'freq_factor': -1,
        'file_names': sorted([r'C:\Users\hbkm9\Documents\Projects\CYB\PyCYB\004_Walk07.json',
                              r'C:\Users\hbkm9\Documents\Projects\CYB\PyCYB\004_Walk08.json']),
        'channel_mask': None,
        'time_step': 1,
        ###############################################################################
        'preproc': norm_emg,
        ###############################################################################
        'batch_size': 64,
        ###############################################################################
    }

    tcn_generator = TCNDataGenerator(**gen_params)
    tcn_generator.show()

