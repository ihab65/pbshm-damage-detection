"""
modeling.py -- The AI Architecture

Pure TensorFlow/Keras logic. Isolates deep learning from civil engineering.
Contains the legacy build_model() plus the dual-branch denoising autoencoder:

  Branch 1 (self-supervised):  Encoder -> Decoder   (reconstruction)
  Branch 2 (supervised):       Encoder -> Predictor  (damage severity)

Key design rationale
--------------------
- The *encoder* compresses the 60-D flexibility indicator vector
  (20 optimal sensor locations x 3 DOFs) into a 16-D latent space.
- During Phase 1 training a *masking noise layer* randomly zeros out
  input channels (simulating missing / absent sensors on B-type
  structures), and the decoder learns to reconstruct the CLEAN signal.
  This makes the latent space robust to sensor-count variability
  and acts as a pseudo transfer-learning mechanism from
  well-instrumented Structure A -> partially-instrumented Structure B.
- In Phase 2 the encoder weights are frozen and a *predictor* head
  maps the latent code to the 3-zone damage severities.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Layer, BatchNormalization
)
from tensorflow.keras.regularizers import l2


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


# ===================================================================
# Dual-branch denoising autoencoder components
# ===================================================================

# Architecture constants ------------------------------------------------
INPUT_DIM   = 60   # 20 sensors x 3 DOFs
LATENT_DIM  = 16   # bottleneck size
OUTPUT_ZONES = 3   # Zone1_Sev, Zone2_Sev, Zone3_Sev


# ---------------------------------------------------------------------------
# Custom masking noise layer
# ---------------------------------------------------------------------------

class SensorMaskingNoise(Layer):
    """
    At training time, randomly sets entire input features to zero.

    This simulates the scenario where certain sensors are absent
    on a B-type structure. Unlike standard Dropout (which scales by
    1/(1-rate)), this layer applies *true zero-masking* with no
    rescaling -- matching the real-world situation where a missing
    sensor simply reports nothing.

    Parameters
    ----------
    drop_rate : float
        Fraction of input features zeroed out during training.
    """

    def __init__(self, drop_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.drop_rate = drop_rate

    def call(self, inputs, training=None):
        if training:
            # Binary mask: 1 = keep, 0 = drop  (no rescaling)
            mask = tf.random.uniform(tf.shape(inputs)) >= self.drop_rate
            return inputs * tf.cast(mask, inputs.dtype)
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({"drop_rate": self.drop_rate})
        return config


# ---------------------------------------------------------------------------
# Encoder: 60 -> 46 -> 32 -> 16 (latent z)
# ---------------------------------------------------------------------------

def build_encoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM, name="encoder"):
    """
    Build the encoder that compresses the flexibility-indicator vector
    into a compact latent representation.

    Architecture
    ------------
    Input(60) -> Dense(46) -> BN -> ReLU
              -> Dense(32) -> BN -> ReLU
              -> Dense(16) -> BN -> ReLU   = z

    BatchNormalization after each layer stabilises training and
    prevents the network from collapsing to a trivial linear mapping.
    The final bottleneck uses ReLU so the latent space is non-negative
    and sparse, which improves disentanglement for the downstream
    predictor.

    Returns
    -------
    tf.keras.Model  with  input_shape=(60,)  output_shape=(16,)
    """
    inp = Input(shape=(input_dim,), name="encoder_input")

    x = Dense(46, kernel_regularizer=l2(1e-4), name="enc_dense_1")(inp)
    x = BatchNormalization(name="enc_bn_1")(x)
    x = tf.keras.layers.Activation("relu", name="enc_act_1")(x)

    x = Dense(32, kernel_regularizer=l2(1e-4), name="enc_dense_2")(x)
    x = BatchNormalization(name="enc_bn_2")(x)
    x = tf.keras.layers.Activation("relu", name="enc_act_2")(x)

    x = Dense(latent_dim, kernel_regularizer=l2(1e-4), name="enc_dense_z")(x)
    x = BatchNormalization(name="enc_bn_z")(x)
    z = tf.keras.layers.Activation("relu", name="latent_z")(x)

    return Model(inp, z, name=name)


# ---------------------------------------------------------------------------
# Decoder: 16 -> 32 -> 46 -> 60  (symmetric reconstruction)
# ---------------------------------------------------------------------------

def build_decoder(latent_dim=LATENT_DIM, output_dim=INPUT_DIM, name="decoder"):
    """
    Build the decoder that reconstructs the CLEAN 60-D signal from z.

    Architecture
    ------------
    Input(16) -> Dense(32) -> BN -> ReLU
              -> Dense(46) -> BN -> ReLU
              -> Dense(60, linear)

    The final layer uses linear activation because the input features
    are StandardScaler'd (zero-mean, unit-variance, unbounded).

    Returns
    -------
    tf.keras.Model  with  input_shape=(16,)  output_shape=(60,)
    """
    inp = Input(shape=(latent_dim,), name="decoder_input")

    x = Dense(32, kernel_regularizer=l2(1e-4), name="dec_dense_1")(inp)
    x = BatchNormalization(name="dec_bn_1")(x)
    x = tf.keras.layers.Activation("relu", name="dec_act_1")(x)

    x = Dense(46, kernel_regularizer=l2(1e-4), name="dec_dense_2")(x)
    x = BatchNormalization(name="dec_bn_2")(x)
    x = tf.keras.layers.Activation("relu", name="dec_act_2")(x)

    out = Dense(output_dim, activation="linear", name="reconstruction")(x)
    return Model(inp, out, name=name)


# ---------------------------------------------------------------------------
# Predictor: z(16) -> 16 -> 8 -> 8 -> 3
# ---------------------------------------------------------------------------

def build_predictor(latent_dim=LATENT_DIM, output_zones=OUTPUT_ZONES,
                    name="predictor"):
    """
    Build the predictor head that maps the frozen latent code to
    per-zone damage severity estimates.

    Architecture
    ------------
    Input(16) -> Dense(16, relu) -> Dense(8, relu) -> Dense(8, relu)
              -> Dense(3, sigmoid)

    The output uses *sigmoid* because damage severities live in [0, 1]
    (the dataset ranges from 0.0 to 0.85, well within [0,1]).

    Returns
    -------
    tf.keras.Model  with  input_shape=(16,)  output_shape=(3,)
    """
    inp = Input(shape=(latent_dim,), name="predictor_input")
    x = Dense(16, activation="relu", name="pred_dense_1")(inp)
    x = Dense(8,  activation="relu", name="pred_dense_2")(x)
    x = Dense(8,  activation="relu", name="pred_dense_3")(x)
    out = Dense(output_zones, activation="sigmoid", name="severity_output")(x)
    return Model(inp, out, name=name)


# ---------------------------------------------------------------------------
# Composite model builders (for convenient single-call training)
# ---------------------------------------------------------------------------

def build_denoising_autoencoder(drop_rate=0.2,
                                input_dim=INPUT_DIM,
                                latent_dim=LATENT_DIM):
    """
    Build the full Phase-1 denoising autoencoder.

    Data flow
    ---------
    clean_input -> SensorMaskingNoise(drop_rate) -> Encoder -> Decoder -> recon

    The loss is MSE(recon, clean_input) -- i.e. the model must recover
    the ORIGINAL (unmasked) signal, not the corrupted one.

    Returns
    -------
    dae : tf.keras.Model
        End-to-end model  (input=clean, output=reconstruction).
    encoder : tf.keras.Model
        Standalone encoder (needed later for Phase 2).
    decoder : tf.keras.Model
        Standalone decoder.
    """
    encoder = build_encoder(input_dim, latent_dim)
    decoder = build_decoder(latent_dim, input_dim)

    clean_input = Input(shape=(input_dim,), name="clean_input")
    noisy = SensorMaskingNoise(drop_rate, name="masking_noise")(clean_input)
    z = encoder(noisy)
    recon = decoder(z)

    dae = Model(clean_input, recon, name="denoising_autoencoder")
    return dae, encoder, decoder


def build_severity_predictor(encoder, latent_dim=LATENT_DIM,
                             output_zones=OUTPUT_ZONES,
                             freeze_encoder=True):
    """
    Build the Phase-2 severity predictor.

    Data flow
    ---------
    clean_input -> Encoder (optionally frozen) -> Predictor -> severity

    Parameters
    ----------
    encoder : tf.keras.Model
        Pre-trained encoder from Phase 1.
    freeze_encoder : bool
        If True, freeze encoder weights (only train predictor head).
        If False, fine-tune encoder jointly with predictor.

    Returns
    -------
    full_model : tf.keras.Model
        End-to-end model  (input=clean, output=3-zone severity).
    predictor : tf.keras.Model
        Standalone predictor head.
    """
    encoder.trainable = not freeze_encoder

    predictor = build_predictor(latent_dim, output_zones)

    clean_input = Input(shape=(encoder.input_shape[1],), name="clean_input")
    z = encoder(clean_input, training=not freeze_encoder)
    severity = predictor(z)

    full_model = Model(clean_input, severity, name="severity_model")
    return full_model, predictor


def build_dual_branch_model(encoder, decoder, predictor=None,
                            drop_rate=0.2,
                            input_dim=INPUT_DIM,
                            latent_dim=LATENT_DIM,
                            output_zones=OUTPUT_ZONES):
    """
    Build a joint dual-branch model that trains BOTH branches
    simultaneously with a combined loss.

    Data flow
    ---------
    clean_input -> SensorMaskingNoise -> Encoder -> z
                                                 |-> Decoder   -> recon  (MSE vs clean)
                                                 |-> Predictor -> sev    (MSE vs labels)

    Both losses are combined:  total = recon_weight * MSE_recon + pred_weight * MSE_sev

    This forces the encoder to learn a latent space that is good
    for BOTH reconstruction AND severity prediction.

    Parameters
    ----------
    encoder : tf.keras.Model
        Encoder (will be trained / fine-tuned).
    decoder : tf.keras.Model
        Decoder (will be trained / fine-tuned).
    predictor : tf.keras.Model or None
        If None, a new predictor is built.

    Returns
    -------
    model : tf.keras.Model
        Dual-output model: outputs = [reconstruction, severity]
    """
    if predictor is None:
        predictor = build_predictor(latent_dim, output_zones)

    # Ensure all components are trainable
    encoder.trainable = True
    decoder.trainable = True
    predictor.trainable = True

    clean_input = Input(shape=(input_dim,), name="clean_input")
    noisy = SensorMaskingNoise(drop_rate, name="masking_noise")(clean_input)

    z = encoder(noisy)
    recon = decoder(z)
    severity = predictor(z)

    model = Model(
        inputs=clean_input,
        outputs=[recon, severity],
        name="dual_branch_model"
    )
    return model, predictor

