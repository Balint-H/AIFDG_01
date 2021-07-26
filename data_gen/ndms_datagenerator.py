import numpy as np
from data_gen.datagenerator import TCNDataGenerator


class NDMSDataGenerator(TCNDataGenerator):

    # def load_files(self):
    #     super(NDMSDataGenerator, self).load_files()
    #     for file_n in range(len(self.feat_data)):
    #         self.feat_data[file_n][1, :, :] = self.feat_data[file_n][1, :, :]*2
    #     # self.feat_data = np.array(self.feat_data)
    #     # self.emg_data = np.array(self.emg_data)

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
            [self.feat_data[file_id][:, win_id * self.stride + self.delay, :].T
             for file_id, win_id in ids])

        return X, Y
    #
    # def data_generation_fast(self, cur_indexes):
    #     head_tails = self.window_index_heads  # Starting and ending index of windows for each file
    #
    #     # Find which files the window indices are found in, and zip them together
    #     file_ids = list()
    #     feat_idxs = list()
    #     emg_slices = list()
    #     for cur_idx in cur_indexes:
    #         for file_id, head_tail in enumerate(head_tails):
    #             if head_tail[0] <= cur_idx < head_tail[1]:
    #                 file_ids.append(file_id)
    #                 scaled_idx = (cur_idx - head_tails[file_id][0])*self.stride
    #                 feat_idxs.append(scaled_idx + self.delay)
    #                 emg_slices.append(slice(scaled_idx, scaled_idx+self.window_size, self.time_step))
    #
    #     X = np.empty((len(cur_indexes), self.window_size, self.n_channels))
    #     for i in range(len(cur_indexes)):
    #         X[i, :, :] = self.emg_data[file_ids[i], :, emg_slices[i]].T
    #
    #     Y = self.feat_data[file_ids, :, feat_idxs, :].T
    #
    #     return X, Y