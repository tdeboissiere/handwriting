import os
import torch

from ..utils import logging_utils as lu
from ..utils import training_utils as tu
from ..utils import inference_utils as iu
from ..utils import visualization_utils as vu


def validate_unconditional(settings):
    """Validate RNN for unconditional handwriting generation

    Plot a series of samples (10) then return
    Save results to figures/validation

    Args:
        settings (ExperimentSettings): custom class to hold hyperparams
    """

    # Re-load data to get the mapping onehot <--> string
    tu.load_data(settings, validate=True)

    if not os.path.isfile(settings.unconditional_model_path):
        lu.print_red("Unconditional model does not exist. Please train one first")

    # Load model
    rnn = torch.load(settings.unconditional_model_path)

    # Use GPU if required
    if settings.use_cuda:
        rnn.use_cuda = True
        rnn.cuda()
    else:
        rnn.use_cuda = False

    # Count figure number
    counter = 0
    for counter in range(10):

        # Sample a sequence to follow progress and save the plot
        plot_data = iu.sample_unconditional_sequence(settings, rnn)
        vu.plot_stroke(plot_data.stroke, "figures/validation/unconditional_sample_%s.png" % counter)

    lu.print_green("Results saved to figures/validation")


def validate_conditional(settings):
    """Validate RNN for conditional handwriting generation

    Ask user for text input to generate indefinitely
    Save results to figures/validation

    Args:
        settings (ExperimentSettings): custom class to hold hyperparams
    """

    # Re-load data to get the mapping onehot <--> string
    tu.load_data(settings, validate=True)

    if not os.path.isfile(settings.conditional_model_path):
        lu.print_red("Conditional model does not exist. Please train one first")

    # Load model
    rnn = torch.load(settings.conditional_model_path)

    # Use GPU if required
    if settings.use_cuda:
        rnn.use_cuda = True
        rnn.cuda()
    else:
        rnn.use_cuda = False

    # Count figure number
    counter = 0
    continue_flag = None
    while True:

        input_text = input("Enter text: ")

        # Check all characters are allowed
        for char in input_text:
            try:
                settings.d_char_to_idx[char]
            except KeyError:
                lu.print_red("%s not in alphabet" % char)
                continue_flag = True

        # Ask for a new input text in case of failure
        if continue_flag:
            continue

        plot_data = iu.sample_fixed_sequence(settings, rnn, truth_text=input_text)
        vu.plot_conditional(settings, plot_data, "figures/validation/conditional_sample_%s.png" % counter)
        lu.print_green("Results saved to figures/validation")

        counter += 1


def generate_unconditionally(settings):
    """Function for lyrebird notebook

    Args:
        settings (ExperimentSettings): custom class to hold hyperparams

    Return:
        (np.array) generated stroke
    """

    # Re-load data to get the mapping onehot <--> string
    # Change path as this will be launched from the notebook repo
    tu.load_data(settings, data_path="../data/raw", validate=True)

    if not os.path.isfile("../pretrained/unconditional.pt"):
        lu.print_red("Unconditional model does not exist. Please train one first")

    # Load model
    rnn = torch.load("../pretrained/unconditional.pt")

    # Use GPU if required
    if settings.use_cuda:
        rnn.use_cuda = True
        rnn.cuda()
    else:
        rnn.use_cuda = False

    # Sample a sequence to follow progress and save the plot
    plot_data = iu.sample_unconditional_sequence(settings, rnn)

    return plot_data.stroke


def generate_conditionally(settings, input_text="welcome to lyrebird"):
    """Function for lyrebird notebook

    Args:
        settings (ExperimentSettings): custom class to hold hyperparams

    Return:
        (np.array) generated stroke
    """

    # Re-load data to get the mapping onehot <--> string
    # Change path as this will be launched from the notebook repo
    tu.load_data(settings, data_path="../data/raw", validate=True)

    if not os.path.isfile("../pretrained/conditional.pt"):
        lu.print_red("Conditional model does not exist. Please train one first")

    # Load model
    rnn = torch.load("../pretrained/conditional.pt")

    # Use GPU if required
    if settings.use_cuda:
        rnn.use_cuda = True
        rnn.cuda()
    else:
        rnn.use_cuda = False

    # Sample a sequence to follow progress and save the plot
    plot_data = iu.sample_fixed_sequence(settings, rnn, truth_text=input_text)

    return plot_data.stroke
