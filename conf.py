import argparse
from handwriting.utils import experiment_settings


def get_args():

    parser = argparse.ArgumentParser(description='Handwriting generation')

    #######################
    # General parameters
    #######################
    parser.add_argument('--train_unconditional', action="store_true", help="Train RNN for unconditional generation")
    parser.add_argument('--train_conditional', action="store_true", help="Train RNN for conditional generation")
    parser.add_argument('--validate_unconditional', action="store_true", help="validate unconditional RNN")
    parser.add_argument('--validate_conditional', action="store_true", help="validate conditional RNN")
    parser.add_argument('--debug', action="store_true", help="Activate debugging options (mostly plotting)")

    #######################
    # Model saving
    #######################    
    parser.add_argument('--unconditional_model_path', type=str,
                        default="models/unconditional.pt", help="Path where to save/load a model")
    parser.add_argument('--conditional_model_path', type=str,
                        default="models/conditional.pt", help="Path where to save/load a model")

    ######################
    # Model parameters
    ######################
    parser.add_argument('--layer_type', default="lstm", type=str, choices=["gru", "lstm"], help="Model type")
    parser.add_argument('--hidden_dim', default=256, type=int, help="Hidden layer dimension")
    parser.add_argument('--num_layers', default=2, type=int, help="Number of recurrent layers")
    parser.add_argument('--recurrent_dropout', default=0, type=float, help="Dropout on recurrent layers")
    parser.add_argument('--n_gaussian', type=int, default=20, help='# of gaussian mixture components')
    parser.add_argument('--n_window', type=int, default=10, help='# of gaussian window for conditional model')

    ######################
    # Optimizer parameters
    ######################
    parser.add_argument('--optimizer', default="adam", type=str, choices=["adam", "rmsprop"], help="Learning rate")
    parser.add_argument('--learning_rate', default=1E-3, type=float, help="Learning rate")
    parser.add_argument('--gradient_clipping', default=5, type=float, help="Max norm allowed for a gradient")

    ######################
    # Training parameters
    ######################
    parser.add_argument('--use_cuda', action="store_true", help="Use GPU (pytorch backend only)")
    parser.add_argument('--nb_epoch', default=100, type=int, help="Number of batches per epoch")
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--n_batch_per_epoch', type=int, default=100, help='Number of batches per epoch')
    parser.add_argument('--bptt', type=int, default=150, help='sequence length')

    ######################
    # Inference parameters
    ######################
    parser.add_argument('--sampling_len', type=int, default=700, help='Max size of sequence to sample from')
    parser.add_argument('--bias', type=float, default=1.0, help='Bias when sampling')

    args = parser.parse_args()

    # Initialize a settings instance
    settings = experiment_settings.ExperimentSettings(args)

    return settings
