"""Keras implementation of the supplied PyTorch SimpleWaveNet, with MSE output."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_wavenet(seq_len=40000):
    adc = keras.Input(shape=(seq_len,), dtype='int32', name='adc_shifted')
    x = layers.Embedding(256, 16, name='embedding')(adc)
    x = layers.Conv1D(32, 1, name='input_conv')(x)
    skips = []
    for i in range(10):
        dilation = 2 ** i
        z = layers.Conv1D(64, 12, padding='causal', dilation_rate=dilation,
                          name=f'block{i}_causal_conv')(x)
        filt = layers.Lambda(lambda t: t[:, :, :32], output_shape=(seq_len, 32),
                             name=f'block{i}_filter_slice')(z)
        gate = layers.Lambda(lambda t: t[:, :, 32:], output_shape=(seq_len, 32),
                             name=f'block{i}_gate_slice')(z)
        filt = layers.Activation('tanh', name=f'block{i}_tanh')(
            layers.Conv1D(32, 1, name=f'block{i}_filter_conv')(filt))
        gate = layers.Activation('sigmoid', name=f'block{i}_sigmoid')(
            layers.Conv1D(32, 1, name=f'block{i}_gate_conv')(gate))
        gated = layers.Multiply(name=f'block{i}_gated')([filt, gate])
        skips.append(layers.Conv1D(32, 1, name=f'block{i}_skip_conv')(gated))
        resid = layers.Conv1D(32, 1, name=f'block{i}_residual_conv')(gated)
        x = layers.Add(name=f'block{i}_residual_add')([x, resid])
    x = layers.Add(name='skip_sum')(skips)
    x = layers.ReLU(name='output_relu1')(x)
    x = layers.Conv1D(32, 1, name='output_conv1')(x)
    x = layers.ReLU(name='output_relu2')(x)
    out = layers.Conv1D(1, 1, name='output_conv2_mse')(x)
    return keras.Model(adc, out, name='WaveNet_MSE_FP32')
