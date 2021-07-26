import matplotlib.pyplot as plt
import numpy as np


def plot_columns(array_in: np.ndarray, ncols=None, xlabels=None, ylabels=None, titles=None, legends=None,
                 remove_xticklabels=None,
                 sb=False, **kwargs):

    if sb:
        import seaborn as sns
        sns.set()

    if ncols is None:
        ncols = int(np.floor(np.sqrt(array_in.shape[-1])))
    nrows = int(np.ceil(array_in.shape[-1] / ncols))

    fig: plt.Figure
    fig, axs = plt.subplots(nrows, ncols, **kwargs)

    ax: plt.Axes
    i = 0
    for i, (ax, col) in enumerate(zip(axs.flatten(), array_in.T)):  # (time, channel) or (subchannel, time, channel)
        ax.plot(col)
        if xlabels is not None:
            ax.set_xlabel(xlabels[i])
        if ylabels is not None:
            ax.set_ylabel(ylabels[i])
        if titles is not None:
            ax.set_title(titles[i])
        if legends is not None:
            if legends[i] is not None:
                ax.legend(legends[i])
        if remove_xticklabels is not None:
            if remove_xticklabels[i]:
                labels = [item.get_text() for item in ax.get_xticklabels()]
                empty_string_labels = [''] * len(labels)
                #ax.set_xticklabels(empty_string_labels)
    for j in range(i+1, axs.size):
        fig.delaxes(axs.flatten()[j])
    fig.align_ylabels()
    plt.tight_layout()


if __name__ == '__main__':
    a = np.arange(300).reshape((3, 100)).T
    plot_columns(a, ncols=2, titles=['Foo', 'Bar', 'Baz'], xlabels=['x']*3, ylabels=['y']*3, sb=True)
    plt.show()
