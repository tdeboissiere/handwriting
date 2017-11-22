import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from ..utils import training_utils as tu
from ..utils import logging_utils as lu
from ..utils import inference_utils as iu
from ..utils import visualization_utils as vu


def train_unconditional(settings):
    """Train RNN for unconditional handwriting generation

    Train for specified number of epochs and batches
    Display 2D NLL loss and BCE loss to monitor training
    Save model every 5 epochs
    Save a generated sample every epoch

    Args:
        settings (ExperimentSettings): custom class to hold hyperparams
    """

    #######
    # Data
    #######
    data = tu.load_data(settings)

    ######################
    # Model specification
    ######################
    input_dim = data.strokes[0].shape[-1]
    output_dim = 6 * settings.n_gaussian + 1
    rnn = tu.get_model(settings, input_dim, output_dim)
    optimizer = tu.get_optimizer(settings, rnn)

    # Use GPU if required
    if settings.use_cuda:
        rnn.cuda()

    # Keep track of losses for display
    loss_str = ""
    d_monitor = defaultdict(list)

    ##########
    # Training
    ##########
    lu.print_green("Starting training")
    for epoch in tqdm(range(settings.nb_epoch), desc="Training"):

        # Track the training losses over an epoch
        d_epoch_monitor = defaultdict(list)

        # Loop over batches
        desc = "Epoch: %s -- %s" % (epoch, loss_str)
        for batch in tqdm(range(settings.n_batch_per_epoch), desc=desc):

            # Sample a batch (X, Y)
            X_var, Y_var = tu.get_random_unconditional_training_batch(settings, data)

            # Train step = forward + backward + weight update
            d_loss = tu.train_step(settings, rnn, X_var, Y_var, optimizer)

            # Optional visualization for debugging
            if settings.debug:
                vu.plot_stroke_from_batch(X_var, Y_var)

            d_epoch_monitor["bce"].append(d_loss["bce"])
            d_epoch_monitor["nll"].append(d_loss["nll"])
            d_epoch_monitor["total"].append(d_loss["total"])

        # Sample a sequence to follow progress and save the plot
        plot_data = iu.sample_unconditional_sequence(settings, rnn)
        vu.plot_stroke(plot_data.stroke, "figures/unconditional_samples/epoch_%s.png" % epoch)

        # Update d_monitor with the mean over an epoch
        for key in d_epoch_monitor.keys():
            d_monitor[key].append(np.mean(d_epoch_monitor[key]))
        # Prepare loss_str to update progress bar
        loss_str = "Total : %.3g -- NLL : %.3g -- BCE: %.3g" % (d_monitor["total"][-1],
                                                                d_monitor["nll"][-1],
                                                                d_monitor["bce"][-1])

        # Save the model at regular intervals
        if epoch % 5 == 0:

            # Move model to cpu before training to allow inference on cpu
            rnn.cpu()
            torch.save(rnn, settings.unconditional_model_path)
            if settings.use_cuda:
                rnn.cuda()

    lu.print_green("Finished training")


def train_conditional(settings):
    """Train RNN for conditional handwriting generation

    Train for specified number of epochs and batches
    Display 2D NLL loss and BCE loss to monitor training
    Save model every epoch
    Save a generated sample every epoch + other plots like attention

    Args:
        settings (ExperimentSettings): custom class to hold hyperparams
    """

    #######
    # Data
    #######
    list_data_train = tu.load_data(settings)

    if settings.debug:
        vu.plot_stroke_with_text(settings, list_data_train)

    ######################
    # Model specification
    ######################
    input_size = list_data_train[0][0].shape[1]
    onehot_dim = list_data_train[-1][0].shape[-1]
    output_size = 6 * settings.n_gaussian + 1
    rnn = tu.get_model(settings, input_size, output_size, onehot_dim=onehot_dim)
    optimizer = tu.get_optimizer(settings, rnn)

    # Use GPU if required
    if settings.use_cuda:
        rnn.cuda()

    # Keep track of losses for display
    loss_str = ""
    d_monitor = defaultdict(list)

    ##########
    # Training
    ##########
    lu.print_green("Starting training")
    for epoch in tqdm(range(settings.nb_epoch), desc="Training"):

        # Track the training losses over an epoch
        d_epoch_monitor = defaultdict(list)

        # Loop over batches
        desc = "Epoch: %s -- %s" % (epoch, loss_str)
        for batch in tqdm(range(settings.n_batch_per_epoch), desc=desc):

            # Sample a batch (X, Y)
            X_var, Y_var, onehot_var = tu.get_random_conditional_training_batch(settings, list_data_train)

            # Train step.
            d_loss = tu.train_step(settings, rnn, X_var, Y_var, optimizer, onehot=onehot_var)

            d_epoch_monitor["bce"].append(d_loss["bce"])
            d_epoch_monitor["nll"].append(d_loss["nll"])
            d_epoch_monitor["total"].append(d_loss["total"])

        # Update d_monitor with the mean over an epoch
        for key in d_epoch_monitor.keys():
            d_monitor[key].append(np.mean(d_epoch_monitor[key]))
        # Prepare loss_str to update progress bar
        loss_str = "Total : %.3g -- NLL : %.3g -- BCE: %.3g" % (d_monitor["total"][-1],
                                                                d_monitor["nll"][-1],
                                                                d_monitor["bce"][-1])

        plot_data = iu.sample_fixed_sequence(settings, rnn)
        vu.plot_conditional(settings, plot_data, "figures/conditional_samples/epoch_%s.png" % epoch)

        # Move model to cpu before training to allow inference on cpu
        rnn.cpu()
        torch.save(rnn, settings.conditional_model_path)
        if settings.use_cuda:
            rnn.cuda()

    lu.print_green("Finished training")
