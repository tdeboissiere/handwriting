import conf
from handwriting.training import train
from handwriting.validation import validate


if __name__ == "__main__":

    # Get conf parameters
    settings = conf.get_args()

    ################
    # TRAINING
    ################
    if settings.train_unconditional:
        train.train_unconditional(settings)

    if settings.train_conditional:
        train.train_conditional(settings)

    ################
    # VALIDATION
    ################
    if settings.validate_unconditional:
        validate.validate_unconditional(settings)

    if settings.validate_conditional:
        validate.validate_conditional(settings)
