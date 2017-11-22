import pytest
import torch
from handwriting.training import train
from handwriting.utils import experiment_settings


@pytest.mark.parametrize("layer_type, use_cuda", [
    ("lstm", True),
    ("lstm", False),
    ("gru", True),
    ("gru", False),
])
def test_unconditional(layer_type, use_cuda):

    settings = experiment_settings.ExperimentSettings({})

    settings.layer_type = layer_type
    settings.use_cuda = use_cuda

    # Don't test GPU is not available
    if use_cuda and not torch.cuda.is_available():
        return

    # Set parameters manually to be able to run test_files individually
    # Otherwise, conflicts with argparse in conf.py
    settings.train_conditional = False
    settings.train_unconditional = True
    settings.bptt = 100
    settings.n_gaussian = 8
    settings.num_layers = 2
    settings.recurrent_dropout = 0.2
    settings.optimizer = "adam"
    settings.learning_rate = 1E-3
    settings.batch_size = 32
    settings.gradient_clipping = 10
    settings.sampling_len = 100
    settings.nb_epoch = 6
    settings.hidden_dim = 16
    settings.n_batch_per_epoch = 2
    settings.debug = False
    settings.bias = 1.0
    settings.conditional_model_path = "models/test.pt"
    settings.unconditional_model_path = "models/test.pt"

    train.train_unconditional(settings)


@pytest.mark.parametrize("layer_type, use_cuda", [
    ("lstm", True),
    ("lstm", False),
    ("gru", True),
    ("gru", False),
])
def test_conditional(layer_type, use_cuda):

    settings = experiment_settings.ExperimentSettings({})

    settings.layer_type = layer_type
    settings.use_cuda = use_cuda

    # Don't test GPU is not available
    if use_cuda and not torch.cuda.is_available():
        return

    # Set parameters manually to be able to run test_files individually
    # Otherwise, conflicts with argparse in conf.py
    settings.train_conditional = True
    settings.train_unconditional = False
    settings.bptt = 100
    settings.n_gaussian = 8
    settings.n_window = 2
    settings.num_layers = 2
    settings.recurrent_dropout = 0.2
    settings.optimizer = "adam"
    settings.learning_rate = 1E-3
    settings.batch_size = 32
    settings.gradient_clipping = 10
    settings.sampling_len = 100
    settings.nb_epoch = 2
    settings.hidden_dim = 16
    settings.n_batch_per_epoch = 2
    settings.debug = False
    settings.bias = 1.0
    settings.conditional_model_path = "models/test.pt"
    settings.unconditional_model_path = "models/test.pt"

    train.train_conditional(settings)
