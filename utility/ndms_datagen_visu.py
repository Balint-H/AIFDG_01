from data_gen.datagenerator import TCNDataGenerator
import numpy as np
from markersets.trajsets.ndmstrajset import NDMSTrajSet
from markersets.trajsets.traj_utils import try_seaborn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import animation
from math import floor
from matplotlib.widgets import Slider

global window_slider

model_parent_dir = r'C:\Users\hbkm9\Documents\School\PhD Year 1\AIRDG\traj\AIFDG_01\Models'


def animate_traj(data_gen: TCNDataGenerator, recording):
    """
    Quickly plot what will this generator feed into the model if used now.
    """

    from matplotlib.ticker import FormatStrFormatter
    try_seaborn()

    n_channels = data_gen.n_channels
    nrows = floor(n_channels / 2)

    fig = plt.figure(figsize=(16, 9))
    spec = gridspec.GridSpec(nrows=nrows, ncols=2, width_ratios=[1, 2])

    channel_names = ['Internal Obliques',
                     'External Obliques',
                     'Lower ES',
                     'Upper ES']

    animation_indices = np.arange(data_gen.window_index_heads[recording][0], data_gen.window_index_heads[recording][1])

    emg, traj = data_gen.data_generation([0])

    emg_axes = list()
    emg_lines = list()
    for i in range(nrows):
        ax = fig.add_subplot(spec[i, 0])

        if i == nrows - 1:
            ax.set_xlabel('Sample in Window')
        else:
            labels = [item.get_text() for item in ax.get_xticklabels()]
            empty_string_labels = [''] * len(labels)
            ax.set_xticklabels(empty_string_labels)

        ax.set_ylabel(f'Channel {i * 2} + {i * 2 + 1}' if not channel_names else channel_names[i])
        ax.set_ylim([np.min(data_gen.emg_data[recording][i * 2, :]), np.max(data_gen.emg_data[recording][i * 2, :])])

        emg_axes.append(ax)
        emg_lines.append(ax.plot(0 + np.arange(data_gen.window_size), emg[0][:, i * 2])[0])
        emg_lines.append(ax.plot(0 + np.arange(data_gen.window_size), emg[0][:, i * 2 + 1])[0])

        if i == 0:
            plt.legend(['Left', 'Right'])

    fig.align_ylabels(emg_axes)
    traj_ax = fig.add_subplot(spec[:, 1])
    traj_line = traj_ax.plot(np.append(0., traj[0][:, 0]), np.append(0, traj[0][:, 1]))[0]
    lines = [*emg_lines, traj_line]
    traj_ax.set_aspect('equal')

    traj_ax.set_xlim([-1.5, 1.5])
    traj_ax.set_ylim([-1.5, 1.5])
    traj_ax.set_xlabel("Hip Forward (m)", fontsize=16)
    traj_ax.set_ylabel("Hip Left (m)", fontsize=16)

    plt.tight_layout()

    def animate(i_frame):

        idx = animation_indices[i_frame]

        emg, traj = data_gen.data_generation([idx])

        for j in range(nrows):
            emg_lines[j * 2].set_ydata(emg[0][:, j * 2])
            emg_lines[j * 2 + 1].set_ydata(emg[0][:, j * 2 + 1])

        traj_line.set_xdata(np.append(0., traj[0][:, 0]))
        traj_line.set_ydata(np.append(0., traj[0][:, 1]))
        return lines

    plt.tight_layout()
    plt.tight_layout()

    ani = animation.FuncAnimation(fig, animate,
                                  interval=round(1 / 100 * 2000), blit=True, frames=2000)

    # print('Saving Animation')
    # ani.save('TrainAnim.mp4', writer="ffmpeg", dpi=100, bitrate=None)
    plt.show()


def plot_traj_slider(data_gen: TCNDataGenerator, recording):
    global window_slider
    try_seaborn()

    n_channels = data_gen.n_channels
    nrows = floor(n_channels / 2)

    fig = plt.figure()
    spec = gridspec.GridSpec(nrows=nrows + 1, ncols=2, height_ratios=nrows * [2] + [1], width_ratios=[1, 2])

    slider_ax = plt.subplot(spec[-1, :], facecolor='lightgoldenrodyellow')

    emg, traj = data_gen.data_generation([0])

    channel_names = ['Internal Obliques',
                     'External Obliques',
                     'Lower ES',
                     'Upper ES']

    emg_axes = list()
    emg_lines = list()
    for i in range(nrows):
        ax = fig.add_subplot(spec[i, 0])

        if i == nrows - 1:
            ax.set_xlabel('Sample in Window')
        else:
            labels = [item.get_text() for item in ax.get_xticklabels()]
            empty_string_labels = [''] * len(labels)
            ax.set_xticklabels(empty_string_labels)

        ax.set_ylabel(f'Channel {i * 2} + {i * 2 + 1}' if not channel_names else channel_names[i])
        ax.set_ylim([np.min(data_gen.emg_data[recording][i * 2, :]), np.max(data_gen.emg_data[recording][i * 2, :])])

        emg_axes.append(ax)
        emg_lines.append(ax.plot(0 + np.arange(data_gen.window_size), emg[0][:, i * 2])[0])
        emg_lines.append(ax.plot(0 + np.arange(data_gen.window_size), emg[0][:, i * 2 + 1])[0])

        if i == 0:
            plt.legend(['Left', 'Right'])

    fig.align_ylabels(emg_axes)
    traj_ax = fig.add_subplot(spec[:-1, 1])
    traj_line = traj_ax.plot(np.append(0., traj[0][:, 0]), np.append(0, traj[0][:, 1]))[0]
    lines = [*emg_lines, traj_line]
    traj_ax.set_aspect('equal')

    traj_ax.set_xlim([-1.5, 1.5])
    traj_ax.set_ylim([-1.5, 1.5])
    traj_ax.set_xlabel("Hip Forward (m)", fontsize=16)
    traj_ax.set_ylabel("Hip Left (m)", fontsize=16)

    plt.tight_layout()

    def update(i_frame):

        idx = int(i_frame)

        emg, traj = data_gen.data_generation([idx])

        for j in range(nrows):
            emg_lines[j * 2].set_ydata(emg[0][:, j * 2])
            emg_lines[j * 2 + 1].set_ydata(emg[0][:, j * 2 + 1])

        traj_line.set_xdata(np.append(0., traj[0][:, 0]))
        traj_line.set_ydata(np.append(0., traj[0][:, 1]))
        return lines

    window_slider = Slider(
        ax=slider_ax,
        label='Window Index',
        valmin=data_gen.window_index_heads[recording][0],
        valmax=data_gen.window_index_heads[recording][1],
        valinit=data_gen.window_index_heads[recording][0],
        valstep=1
    )

    window_slider.on_changed(update)

    plt.tight_layout()
    plt.show()


def plot_time_series(model_num, file_id=0, nrows=3, best_not_end=True, files=None, stride=None, time_idxs=None):
    import pickle
    from tensorflow.keras.models import load_model
    from utility.plot_utility import plot_columns
    try_seaborn()

    model_dir = f'C:\\Users\\hbkm9\\Documents\\Projects\\CYB\\PyCYB\\Models\\model_{model_num}'

    with open(f'{model_dir}\\valid_gen_{model_num}.pickle', "rb") as input_file:
        data_gen: TCNDataGenerator = pickle.load(input_file)

    if stride is not None:
        data_gen.stride = stride

    if files is None:
        data_gen.load_files()
    else:
        data_gen.file_names = files
        data_gen.load_files()

    cur_model = load_model(f'{model_dir}\\best_model_{model_num}.h5') if best_not_end \
        else load_model(f'{model_dir}\\model_{model_num}.h5')

    idxs = np.arange(data_gen.window_index_heads[file_id][0], data_gen.window_index_heads[file_id][1])

    emg, traj = data_gen.data_generation(idxs)

    pred = cur_model.predict(emg)

    if time_idxs is None:
        row_idxs = np.round(np.linspace(0, traj.shape[1] - 1, nrows)).astype(int)
    else:
        row_idxs = time_idxs

    traj = traj[:, row_idxs, :]
    pred = pred[:, row_idxs, :]

    # traj = np.moveaxis(traj, 2, 1)
    # pred = np.moveaxis(pred, 2, 1)

    traj = traj.reshape(traj.shape[0], -1)
    pred = pred.reshape(pred.shape[0], -1)

    to_plot = np.stack([traj, pred])
    plot_columns(to_plot, 2,
                 xlabels=[''] * ((nrows - 1) * 2) + ['Frame index', 'Frame index'],
                 ylabels=['1/3 second', None, '2/3 second', None, '1 second', None],
                 remove_xticklabels=[True] * ((nrows - 1) * 2) + [False, False],
                 legends=[None] + [['True', 'Predicted']] + [None] * ((nrows - 1) * 2),
                 titles=['Forward', 'Left / Right'] + [None] * ((nrows - 1) * 2),
                 sharex=True)

    plt.show()


def animate_model_traj(model_num, file_id=0, best_not_end=True, files=None, stride=None, save=False):
    import pickle
    from tensorflow.keras.models import load_model
    from utility.plot_utility import plot_columns
    try_seaborn()

    model_dir = f'{model_parent_dir}\\model_{model_num}'

    with open(f'{model_dir}\\valid_gen_{model_num}.pickle', "rb") as input_file:
        data_gen: TCNDataGenerator = pickle.load(input_file)

    if stride is not None:
        data_gen.stride = stride

    if files is None:
        data_gen.load_files()
    else:
        data_gen.file_names = files
        data_gen.load_files()

    cur_model = load_model(f'{model_dir}\\best_model_{model_num}.h5') if best_not_end \
        else load_model(f'{model_dir}\\model_{model_num}.h5')

    idxs = np.arange(data_gen.window_index_heads[file_id][0], data_gen.window_index_heads[file_id][1])

    _, traj = data_gen.data_generation(idxs)

    pred = cur_model.predict(data_gen.data_generation(idxs[0:1])[0])


    fig = plt.figure(figsize=(16, 9))
    traj_ax = fig.add_subplot()
    traj_line = traj_ax.plot(np.append(0., traj[0][:, 0]), np.append(0, traj[0][:, 1]))[0]
    pred_line = traj_ax.plot(np.append(0., pred[0][:, 0]), np.append(0, pred[0][:, 1]))[0]
    plt.legend(['Truth', 'Prediction'])

    lines = [traj_line, pred_line]
    traj_ax.set_aspect('equal')

    traj_ax.set_xlim([-1.5, 1.5])
    traj_ax.set_ylim([-1.5, 1.5])
    traj_ax.set_xlabel("Hip Forward (m)", fontsize=16)
    traj_ax.set_ylabel("Hip Left (m)", fontsize=16)

    plt.tight_layout()

    def animate(i_frame):

        idx = idxs[i_frame]

        cur_pred = cur_model.predict(data_gen.data_generation([idx])[0])

        traj_line.set_xdata(np.append(0., traj[idx][:, 0]))
        traj_line.set_ydata(np.append(0., traj[idx][:, 1]))

        pred_line.set_xdata(np.append(0., cur_pred[0][:, 0]))
        pred_line.set_ydata(np.append(0., cur_pred[0][:, 1]))

        return lines

    plt.tight_layout()
    plt.tight_layout()

    ani = animation.FuncAnimation(fig, animate,
                                  interval=round(1 / 10 * stride), blit=True, frames=2000)

    if save:
        print('Saving Animation')
        ani.save(f'predict_{model_num}.mp4', writer="ffmpeg", dpi=100, bitrate=None)
    plt.show()

    pass


def slider_model(model_num: int, file_id: int = 0, plot_emg=False, plot_end=False, files=None, stride=None):
    global window_slider
    import pickle
    from tensorflow.keras.models import load_model
    try_seaborn()

    model_dir = f'{model_parent_dir}\\model_{model_num}'

    with open(f'{model_dir}\\valid_gen_{model_num}.pickle', "rb") as input_file:
        data_gen: TCNDataGenerator = pickle.load(input_file)

    if stride is not None:
        data_gen.stride = stride

    if files is None:
        data_gen.load_files()
    else:
        data_gen.file_names = files
        data_gen.load_files()

    best_model = load_model(f'{model_dir}\\best_model_{model_num}.h5')
    end_model = load_model(f'{model_dir}\\model_{model_num}.h5')

    n_channels = data_gen.n_channels
    nrows = floor(n_channels / 2)

    fig = plt.figure()
    spec = gridspec.GridSpec(nrows=nrows + 1, ncols=2, height_ratios=nrows * [2] + [1], width_ratios=[1, 2])

    slider_ax = plt.subplot(spec[-1, :], facecolor='lightgoldenrodyellow')

    idxs = np.arange(data_gen.window_index_heads[file_id][0], data_gen.window_index_heads[file_id][1])

    emg, traj = data_gen.data_generation([idxs[0]])

    pred = best_model.predict(data_gen.data_generation(idxs[0:1])[0][0:1])
    end = end_model.predict(data_gen.data_generation(idxs[0:1])[0][0:1])

    channel_names = ['Internal Obliques',
                     'External Obliques',
                     'Lower ES',
                     'Upper ES']

    emg_axes = list()
    emg_lines = list()
    for i in range(nrows):
        if plot_emg:
            ax = fig.add_subplot(spec[i, 0])

            if i == nrows - 1:
                ax.set_xlabel('Sample in Window')
            else:
                labels = [item.get_text() for item in ax.get_xticklabels()]
                empty_string_labels = [''] * len(labels)
                ax.set_xticklabels(empty_string_labels)

            ax.set_ylabel(f'Channel {i * 2} + {i * 2 + 1}' if not channel_names else channel_names[i])
            ax.set_ylim([np.min(data_gen.emg_data[file_id][i * 2, :]), np.max(data_gen.emg_data[file_id][i * 2, :])])

            emg_axes.append(ax)
            emg_lines.append(ax.plot(0 + np.arange(data_gen.window_size), emg[0][:, i * 2])[0])
            emg_lines.append(ax.plot(0 + np.arange(data_gen.window_size), emg[0][:, i * 2 + 1])[0])

        if i == 0:
            plt.legend(['Left', 'Right'])

    fig.align_ylabels(emg_axes)
    if plot_emg:
        traj_ax = fig.add_subplot(spec[:-1, 1])
    else:
        traj_ax = fig.add_subplot(spec[:-1, :])
    traj_line = traj_ax.plot(np.append(0., traj[0][:, 0]), np.append(0, traj[0][:, 1]))[0]
    pred_line = traj_ax.plot(np.append(0., pred[0][:, 0]), np.append(0, pred[0][:, 1]))[0]
    end_line_c = []
    if plot_end:
        end_line = traj_ax.plot(np.append(0., end[0][:, 0]), np.append(0, end[0][:, 1]))[0]
        end_line_c.append(end_line)
    lines = [*emg_lines, traj_line, pred_line, *end_line_c]
    traj_ax.set_aspect('equal')

    traj_ax.set_xlim([-1.5, 1.5])
    traj_ax.set_ylim([-1.5, 1.5])
    traj_ax.set_xlabel("Hip Forward (m)", fontsize=16)
    traj_ax.set_ylabel("Hip Left (m)", fontsize=16)

    plt.tight_layout()

    def update(i_frame):

        idx = int(i_frame)

        emg, traj = data_gen.data_generation([idx])
        pred = best_model.predict(emg)

        if plot_emg:
            for j in range(nrows):
                emg_lines[j * 2].set_ydata(emg[0][:, j * 2])
                emg_lines[j * 2 + 1].set_ydata(emg[0][:, j * 2 + 1])

        traj_line.set_xdata(np.append(0., traj[0][:, 0]))
        traj_line.set_ydata(np.append(0., traj[0][:, 1]))

        pred_line.set_xdata(np.append(0., pred[0][:, 0]))
        pred_line.set_ydata(np.append(0., pred[0][:, 1]))

        if plot_end:
            end = end_model.predict(emg)
            end_line.set_xdata(np.append(0., end[0][:, 0]))
            end_line.set_ydata(np.append(0., end[0][:, 1]))
        return lines

    window_slider = Slider(
        ax=slider_ax,
        label='Window Index',
        valmin=data_gen.window_index_heads[file_id][0],
        valmax=data_gen.window_index_heads[file_id][1],
        valinit=data_gen.window_index_heads[file_id][0],
        valstep=1
    )

    window_slider.on_changed(update)

    plt.tight_layout()
    plt.show()


def main():
    import os

    dir_num = 713
    val_dir = r'C:\Users\hbkm9\Documents\School\PhD Year 1\AIRDG\traj\AIFDG_01\Dataset\Validation'

    def valid_target_file(f):
        return 'Freeform' in f

    files = sorted([f for f in os.listdir(val_dir) if f.endswith('.json') and valid_target_file(f)])

    # slider_model(dir_num, file_id=0, stride=10)
    plot_time_series(dir_num, file_id=0, files=files)

    # animate_model_traj(dir_num, files=files, stride=60, save=False)

    pass


if __name__ == '__main__':
    main()
