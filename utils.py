import numpy as np
import tensorflow as tf


def psnr(x, y, maximum=1.0):
    return 20 * np.log10(maximum / np.sqrt(np.mean((x - y) ** 2)))


def tone_mapping(x, mu=5000.0):
    return tf.log(1.0 + mu * x) / tf.log(1.0 + mu)
