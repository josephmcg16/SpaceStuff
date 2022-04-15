"""TwoBodyProblem experiment file using convolutional layers for autoencoder.
"""

from tensorflow import keras
from keras.regularizers import l1_l2
from keras.activations import relu

from architecture.utils import run_experiment, getdatasize
from architecture.ConvResBlock import ConvResBlock
from architecture.RelMSE import RelMSE

if __name__ == "__main__":
    EXPT_NAME = 'TwoBodyProblem_Expt'
    DATA_FILE_PREFIX = 'data/TwoBodyProblem'
    LEN_TIME, N_INPUTS = getdatasize(DATA_FILE_PREFIX)[1:]

    # Set parameters
    N_LATENT = 20  # size of autencoder latent dimension
    NUM_OF_TRAIN = 20  # number of training data files
    L_DIAG = False  # Whether the dynamics matrix is forced to be diagonal
    NUM_SHIFTS = 720
    NUM_SHIFTS_MIDDLE = 720
    LOSS_WEIGHTS = [1, 1, 1, 1, 1]  # Weights of 5 loss functions

    # Set up encoder and decoder configuration dict(s)
    activation = relu
    initializer = keras.initializers.VarianceScaling()
    regularizer = l1_l2(0, 1e-8)

    convlay_config = {'kernel_size': 4,
                      'strides': 1,
                      'padding': 'SAME',
                      'activation': activation,
                      'kernel_initializer': initializer,
                      'kernel_regularizer': regularizer}

    poollay_config = {'pool_size': 2,
                      'strides': 2,
                      'padding': 'VALID'}

    dense_config = {'activation': activation,
                    'kernel_initializer': initializer,
                    'kernel_regularizer': regularizer}

    output_config = {'activation': None,
                     'kernel_initializer': initializer,
                     'kernel_regularizer': regularizer}

    outer_config = {'n_inputs': N_INPUTS,
                    'num_filters': [8, 16, 32, 64],
                    'convlay_config': convlay_config,
                    'poollay_config': poollay_config,
                    'dense_config': dense_config,
                    'output_config': output_config}

    inner_config = {'kernel_regularizer': regularizer}

    # Set up network configuration dict
    network_config = {'n_inputs': N_INPUTS,
                      'n_latent': N_LATENT,
                      'len_time': LEN_TIME,
                      'num_shifts': NUM_SHIFTS,
                      'num_shifts_middle': NUM_SHIFTS_MIDDLE,
                      'outer_encoder': ConvResBlock(**outer_config),
                      'outer_decoder': ConvResBlock(**outer_config),
                      'inner_config': inner_config,
                      'L_diag': L_DIAG}

    # Aggregate all the training options in one dictionary
    training_options = {'aec_only_epochs': 3,
                        'init_full_epochs': 15,
                        'best_model_epochs': 300,
                        'num_init_models': 20,
                        'loss_fn': RelMSE(),
                        'optimizer': keras.optimizers.Adam,
                        'optimizer_opts': {},
                        'batch_size': 32,
                        'data_train_len': NUM_OF_TRAIN,
                        'loss_weights': LOSS_WEIGHTS}

    # RUN THE EXPERIMENT
    SEED = 123
    # Set the custom objects used in the model (for loading purposes)
    custom_objs = {"RelMSE": RelMSE}

    run_experiment(random_seed=SEED,
                   expt_name=EXPT_NAME,
                   data_file_prefix=DATA_FILE_PREFIX,
                   training_options=training_options,
                   network_config=network_config,
                   custom_objects=custom_objs)
