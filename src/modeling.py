"""
modeling.py — The AI Architecture

Pure TensorFlow/Keras logic. Isolates deep learning from civil engineering.
Contains the legacy build_model() plus new modular autoencoder components.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense


# ---------------------------------------------------------------------------
# Legacy model builder (kept for testing / comparison)
# ---------------------------------------------------------------------------

def build_model(input_dim, layer_sizes, hidden_activation,
                output_dim, output_activation):
    """
    Build a generic feedforward Keras model.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input.
    layer_sizes : list[int]
        Sizes of each hidden layer.
    hidden_activation : str
        Activation function for hidden layers.
    output_dim : int
        Dimensionality of the output.
    output_activation : str
        Activation function for the output layer.

    Returns
    -------
    tensorflow.keras.models.Model
    """
    input_layer = Input(shape=(input_dim,))
    x = input_layer
    for size in layer_sizes:
        x = Dense(size, activation=hidden_activation)(x)
    output = Dense(output_dim, activation=output_activation, name='output')(x)
    return Model(inputs=input_layer, outputs=output)


# ---------------------------------------------------------------------------
# New modular architecture: Encoder → Predictor / Decoder
# ---------------------------------------------------------------------------

def build_encoder():
    """
    Build the encoder network that maps sparse sensor data
    to a shared latent (bottleneck) space.

    TODO: Implement architecture — will be provided later.
    """
    pass


def build_predictor():
    """
    Build the predictor network that maps the latent representation
    to damage severity predictions across structural zones.

    TODO: Implement architecture — will be provided later.
    """
    pass


def build_decoder():
    """
    Build the decoder network that reconstructs the original
    sensor data from the latent representation (autoencoder path).

    TODO: Implement architecture — will be provided later.
    """
    pass
