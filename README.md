# handwriting repository

![figure](/docs/walkthrough/sample_figure.png)

## Repository overview

    ├── data
        ├──raw                  --> contains raw data
    ├── figures                 --> contains figures
    ├── pretrained              --> contains pretrained rnn models
    ├── models                  --> stores rnn models during training
    ├── handwriting             --> contains main module
        ├──utils                --> contains utilities used throughout the module
        ├──training             --> contains training scripts and rnn model specification
        validation              --> contains validation scripts
    ├── tests                   --> contains unit tests to pipeline

## Building the docs

    cd docs && make clean && make html && firefox _build/html/index.html

Replace firefox with your web browser if needed.
The docs contain detailed information about the code and the experiments.
For a quicker overview, see below:

Quick peek :

![figure](/docs/installation/peek.png)

## Set up environment

Install python3 and required packages

    bash setup_conda.sh

This creates a `customconda` folder in your home directory with a virtualenv tailored to this repo.
To activate the virtual env:

    bash virtualenv.sh


## Pipeline overview

- Data is stored in the data folder
- The handwriting module contains the core of the code
- The utils submodule contains snippets for training, plotting and sampling
- The training submodule contains the code to define and train the models
- The validation submodule contains the code to sample and evaluate models after training
- Tests are stored in the tests folder.


## Training part

The models in pretrained were resp. generated with :

    python main.py --optimizer rmsprop --learning_rate 1E-4 --use_cuda --bptt 150 --train_unconditional --batch_size 64 --hidden_dim 200 --n_batch_per_epoch 500 --nb_epoch 500 --num_layers 2
    python main.py --optimizer rmsprop --learning_rate 1E-4 --use_cuda --bptt 150 --train_conditional --batch_size 64 --hidden_dim 200 --n_batch_per_epoch 500 --nb_epoch 500 --num_layers 2

Inspect `conf.py` for the description of available training options


## Validation part

Sample from trained models with:

    python main.py --validate_unconditional --use_cuda --unconditional_model_path XXX
    python main.py --validate_conditional --use_cuda --conditional_model_path XXX

The path to the model defaults to `models/unconditional.pt` and `models/conditional.pt`


## Running the notebook

    jupyter notebook notebooks

Then select either the `visualization.ipynb` (which visualizes the data) or `results.ipynb` (which shows the results of trained models)


## Running tests with py.test

    PYTHONPATH=$PWD:$PYTHONPATH pytest
