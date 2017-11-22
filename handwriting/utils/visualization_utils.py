from __future__ import print_function

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec


def get_stroke_arrays(stroke):
    """Utility to split a (n, 3) stroke array

    Args:
        stroke (np.array): input stroke

    Returns:
        x (np.array): stroke line x coordinates
        x (np.array): stroke line y coordinates
        cuts (np.array): indicates where the pen is lifted
    """

    x = np.cumsum(stroke[:, 1])
    y = np.cumsum(stroke[:, 2])

    cuts = np.where(stroke[:, 0] == 1)[0]

    return x, y, cuts


def plot_stroke(stroke, save_name=None):
    """Utility to plot a single stroke, optionally saving it

    Args:
        stroke (np.array): input stroke
        save_name (str): save a figure with name = save_name

    """

    # Plot a single example.
    f, ax = plt.subplots()

    x, y, cuts = get_stroke_arrays(stroke)

    start = 0
    size_x = x.max() - x.min() + 1.
    size_y = y.max() - y.min() + 1.

    f.set_size_inches(5. * size_x / size_y, 5.)

    for i, cut_value in enumerate(cuts):
        if i % 2 == 0:
            color = "C0"
        else:
            color = "C1"
        ax.plot(x[start:cut_value], y[start:cut_value],
                linestyle="-", linewidth=3, color=color)
        start = cut_value + 1
    ax.axis('equal')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if save_name is None:
        plt.show()
    else:
        try:
            plt.savefig(
                save_name,
                bbox_inches='tight',
                pad_inches=0.5)
        except Exception:
            print("Error building image!: " + save_name)

    plt.close()


def plot_stroke_on_axis(ax, stroke):
    """Utility to plot a stroke on a given axis

    Args:
        ax (matplotlib axis): a matplotlib axis
        stroke (np.array): input stroke
    """

    x, y, cuts = get_stroke_arrays(stroke)

    start = 0

    for cut_value in cuts:
        ax.plot(x[start:cut_value], y[start:cut_value],
                'k-', linewidth=3)
        start = cut_value + 1
    ax.axis('equal')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)


def plot_stroke_from_batch(X_var, Y_var):
    """Utility to plot a stroke from a torch batch
    Useful for debugging training

    Args:
        X_var (torch Variable): batch of strokes
        Y_var (torch Variable): batch of strokes incremented by 1 in stroke steps
    """

    if X_var.size(1) < 2:
        return

    # Plot 2 strokes and the target
    plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(2,2)
    for i in range(2):
        ax = plt.subplot(gs[0, i])
        stroke = X_var[:, i, :].data.cpu().numpy()
        plot_stroke_on_axis(ax, stroke)

        ax = plt.subplot(gs[1, i])
        stroke = Y_var[:, i, :].data.cpu().numpy()
        plot_stroke_on_axis(ax, stroke)

    plt.show()


def plot_stroke_with_text(settings, data):
    """Utility to plot a stroke with corresponding text

    We verify that we recover the text from the onehot encoding

    Args:
        settings (ExperimentSettings): custom class holding parameters of interest
        data (list): list containing an aray of strokes, str texts and onehot encoding of the text.
    """

    # Roll out data
    strokes, texts, onehots = data
    # Pick a random sequence
    idx = np.random.randint(0, len(strokes))

    stroke = strokes[idx]
    text = texts[idx]
    onehot = onehots[idx]

    # Reconstruct text from onehot
    reconstructed_text = ""
    for i in range(onehot.shape[0]):
        char_idx = np.argmax(onehot[i, 0, :])
        char = settings.d_idx_to_char[char_idx]
        reconstructed_text += char

    print("Original:", text)
    print("Reconstructed:", reconstructed_text)
    assert text == reconstructed_text

    # Plot a single example.
    f, ax = plt.subplots()

    x, y, cuts = get_stroke_arrays(stroke)

    start = 0
    size_x = x.max() - x.min() + 1.
    size_y = y.max() - y.min() + 1.

    f.set_size_inches(5. * size_x / size_y, 5.)

    for i, cut_value in enumerate(cuts):
        if i % 2 == 0:
            color = "C0"
        else:
            color = "C1"

        ax.plot(x[start:cut_value], y[start:cut_value],
                linestyle="-", linewidth=3, color=color)
        start = cut_value + 1
    ax.axis('equal')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    plt.show()
    plt.close()


def plot_conditional(settings, plot_data, fig_name):
    """Grid plot showing phi, attention window, density and sampled stroke

    Args:
        settings (ExperimentSettings): custom class holding parameters of interest
        plot_data (PlotData): custom class holding plotting data
        fig_name (str): path where to save figure
    """

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 4)
    gs.update(hspace=0.1)

    ##################
    # Attention plots
    ##################
    ax_phi = plt.subplot(gs[0, :2])
    ax_window = plt.subplot(gs[0, 2:])

    ax_phi.set_title('Phis', fontsize=20)
    ax_phi.set_xlabel("One-hot steps", fontsize=15)
    ax_phi.set_ylabel("Stroke steps", fontsize=15)
    ax_phi.imshow(plot_data.phi, interpolation='nearest', aspect='auto', cmap="viridis")

    ax_window.set_title('Attention window', fontsize=20)
    ax_window.set_xlabel("Vocabulary", fontsize=15)
    ax_window.set_xticks(np.arange(settings.n_alphabet))
    ax_window.set_xticklabels(settings.alphabet)
    ax_window.imshow(plot_data.window, interpolation='nearest', aspect='auto', cmap="viridis")

    ##################
    # Density plot
    ##################
    ax_density = plt.subplot(gs[1, :])

    # cumulative sum because the mu1 / mu2 are the parameters of the delta_stroke
    arr_density = plot_data.density
    arr_density[:,:2] = np.cumsum(arr_density[:,:2], axis=0)

    # Define a grid on which we'll plot the gaussian 2d density
    minx, maxx = np.min(arr_density[:,0]) - 1, np.max(arr_density[:,0]) + 1
    miny, maxy = np.min(arr_density[:,1]) - 1, np.max(arr_density[:,1]) + 1
    delta = abs(maxx - minx) / 400.
    x = np.arange(minx, maxx, delta)
    y = np.arange(miny, maxy, delta)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(arr_density.shape[0]):
        # We don't use sigmaxy (== rho), this causes plotting failures
        gauss = mlab.bivariate_normal(X, Y, mux=arr_density[i,0], muy=arr_density[i,1],
                                      sigmax=arr_density[i,2], sigmay=arr_density[i,3], sigmaxy=0)
        Z += gauss / (np.max(gauss) + 1E-6)
    ax_density.imshow(Z, interpolation='nearest', aspect="auto", cmap="viridis")
    # Reverse axes to plot it in the correct orientation
    ax_density.invert_yaxis()
    ax_density.set_xlabel("2D Density for text: %s" % plot_data.text, fontsize=16)
    ax_density.set_xticks([])
    ax_density.set_yticks([])

    ##################
    # Stroke plot
    ##################
    ax_stroke = plt.subplot(gs[2, :])

    x, y, cuts = get_stroke_arrays(plot_data.stroke)
    start = 0
    for i, cut_value in enumerate(cuts):
        if i % 2 == 0:
            color = "C0"
        else:
            color = "C1"
        ax_stroke.plot(x[start:cut_value], y[start:cut_value],
                       linestyle="-", linewidth=3, color=color)
        start = cut_value + 1
    ax_stroke.set_xlabel("Sampled stroke for text: %s" % plot_data.text, fontsize=16)
    ax_stroke.set_xticks([])
    ax_stroke.set_yticks([])

    gs.tight_layout(fig)
    plt.savefig(fig_name)
    plt.clf()
    plt.close()
