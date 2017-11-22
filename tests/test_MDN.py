import numpy as np
from tqdm import tqdm
from sklearn import datasets
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm


from handwriting.training import models
from handwriting.utils import experiment_settings
from handwriting.utils import training_utils as tu
from handwriting.utils import inference_utils as iu

import torch
from torch.autograd import Variable


def test_MDN():

    # In this test, we check our MDN code against a simple example
    settings = experiment_settings.ExperimentSettings({})

    # Set parameters manually to be able to run test_files individually
    # Otherwise, conflicts with argparse in conf.py
    settings.train_conditional = False
    settings.train_unconditional = True
    settings.bptt = 100
    settings.n_gaussian = 8
    settings.num_layers = 2
    settings.recurrent_dropout = 0.2
    settings.dense_dropout = 0.2
    settings.use_cuda = False
    settings.optimizer = "adam"
    settings.learning_rate = 1E-3
    settings.batch_size = 32
    settings.gradient_clipping = 10
    settings.sampling_len = 100
    settings.nb_epoch = 10
    settings.hidden_dim = 16
    settings.debug = False
    settings.bias = 1.0

    #######
    # Data
    #######
    NPTS = 1000
    # X, X_idx = datasets.make_blobs(n_samples=NPTS, centers=10, cluster_std=0.1, center_box=[-2.0, 2.0])
    X, X_idx = datasets.make_moons(n_samples=NPTS, noise=0.1)

    # Split train / val
    X_train = X[:NPTS // 2].astype(np.float32)
    X_val = X[NPTS // 2:].astype(np.float32)
    # x_min, x_max = X_train.min(), X_train.max()
    # X_val = np.random.uniform(x_min, x_max, (NPTS, 2)).astype(np.float32)

    input_dim = X_train.shape[-1]
    hidden_dim = 64
    n_gaussian = 20

    ######################
    # Model specification
    ######################
    model = models.DenseMDN(input_dim, hidden_dim, n_gaussian)
    optimizer = tu.get_optimizer(settings, model)

    # Keep track of losses for plotting
    loss_str = ""

    ##########
    # Training
    ##########
    for epoch in tqdm(range(settings.nb_epoch), desc="Training"):

        desc = "Epoch: %s -- %s" % (epoch, loss_str)

        list_loss_nll = []

        num_elem = X_train.shape[0]
        num_batches = num_elem / settings.batch_size
        list_batches = np.array_split(np.arange(num_elem), num_batches)

        for batch_idxs in tqdm(list_batches, desc=desc):

            start, end = batch_idxs[0], batch_idxs[-1] + 1
            X_batch = X_train[start: end]
            X_var = Variable(torch.from_numpy(X_batch))

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            mdnparams = model(X_var)

            X1, X2 = X_var.split(1, dim=1)

            # Compute nll loss for next stroke 2D prediction
            nll = tu.gaussian_2Dnll(X1, X2, mdnparams)

            # Divide by batch size
            nll = nll / float(X_var.size(0))  # divide by batch size

            # Backward pass
            nll.backward()
            # Weight update
            optimizer.step()

            # Monitoring
            list_loss_nll.append(nll.data.cpu().numpy()[0])

        # Prepare loss_str to update progres bar
        loss_str = "NLL : %.3g" % np.mean(list_loss_nll)

        # Sample from NDM and plot
        X_samples = iu.sample_test_MDN(settings, model, X_val)

        plt.figure()
        gs = gridspec.GridSpec(1, 2)
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])

        _, x_bins, y_bins, _ = ax0.hist2d(X_train[:, 0], X_train[:, 1], bins=100, norm=LogNorm())
        ax1.hist2d(X_samples[:, 0], X_samples[:, 1], bins=[x_bins, y_bins], norm=LogNorm())

        ax1.set_xlim(ax0.get_xlim())
        ax1.set_ylim(ax0.get_ylim())

        ax0.set_aspect("equal")
        ax1.set_aspect("equal")

        ax0.set_title("Truth", fontsize=18)
        ax1.set_title("Samples", fontsize=18)

        plt.savefig("figures/ndm_samples.png")
        plt.clf()
        plt.close()


if __name__ == '__main__':

    test_MDN()
