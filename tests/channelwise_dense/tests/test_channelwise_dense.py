from layers.channelwise_dense.channelwise_dense import ChannelwiseDense
from layers import ChannelwiseDense
import tensorflow as tf
import pytest
import numpy as np


@pytest.mark.parametrize("input_shape", [[1, 21, 37, 3], [50, 21, 37, 3], [1, 21, 37, 1], [50, 21, 37, 1], [1, 21, 21, 1], [50, 21, 21, 1], [1, 21, 21, 3], [50, 21, 21, 3]])
def test_call_returns_valid_output_shape(input_shape):
    img = tf.random.normal(input_shape)
    layer = ChannelwiseDense(input_shape[1:], 'relu')
    layer.build(input_shape)

    out = layer(img)
    out_shape = tf.shape(out)

    assert np.array_equal(out_shape, input_shape)
