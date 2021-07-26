import re

import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import numpy as np
import os


def plot_history(history, data_dir):
    sns.set()
    sns.set_context('paper')
    plt.figure()
    f, axs = plt.subplots(nrows=1, ncols=int(len(history)/2))
    if not isinstance(axs, np.ndarray):
        axs = [axs]
    legends = [[] for _ in range(int(len(history)/2))]
    for i, (k, v) in enumerate(history.items()):
        axs[i%int(len(history)/2)].plot(v)
        legends[i%int(len(history)/2)].append(k)

    for i in range(int(len(history)/2)):
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel(legends[i][0])
        axs[i].legend(legends[i])

    fig = plt.gcf()
    fig.set_size_inches((15.6, 6.775), forward=False)
    plt.savefig(data_dir + '//' + "history_"+ "{:.4f}".format(np.min(history["val_loss"])) + ".svg")
    return


def ndms_plot_pred(emg_data, Y0, model, data_dir, background_signal=None):
    sns.set()
    sns.set_context('paper')

    Y = model.predict(emg_data)
    dir_number = int(re.search(r'(\d+)$', str(os.path.splitext(data_dir)[0])).group(0))
    best_model = load_model(f'{data_dir}\\best_model_{dir_number}.h5')
    Yb = best_model.predict(emg_data)

    time_samples = np.round(np.linspace(0, emg_data.shape[0]-1, 7)).astype(int)

    for i, t in enumerate(time_samples):
        fig = plt.figure()
        traj_ax = fig.add_subplot()

        traj_ax.plot(np.append(0., Y[t][:, 0]), np.append(0, Y[t][:, 1]))
        traj_ax.plot(np.append(0., Yb[t][:, 0]), np.append(0, Yb[t][:, 1]))
        traj_ax.plot(np.append(0., Y0[t][:, 0]), np.append(0, Y0[t][:, 1]))

        plt.legend(['Predicted', 'Best prediction', 'Ground Truth'])

        traj_ax.set_aspect('equal')

        traj_ax.set_xlim([-1.5, 1.5])
        traj_ax.set_ylim([-1.5, 1.5])
        traj_ax.set_xlabel("Hip Forward (m)", fontsize=16)
        traj_ax.set_ylabel("Hip Left (m)", fontsize=16)

        plt.tight_layout()

        plt.savefig(f'{data_dir}//val_plot_{i}.svg')

    return

