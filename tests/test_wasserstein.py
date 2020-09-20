import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tfp_max import wasserstein as ws

tfd = tfp.distributions


def test_sample():
    tf.random.set_seed(0)
    offset = 2.0
    n0 = tfd.Normal(0.0, 1.0)
    n1 = tfd.Normal(offset, 1.0)
    sample_ws = ws.wasserstein_sampled(n0, n1, 1024)
    np.testing.assert_allclose(sample_ws.numpy(), offset, atol=0.1)


def test_cdf():
    tf.random.set_seed(0)
    offset = 2.0
    n0 = tfd.Normal(0.0, 1.0)
    n1 = tfd.Normal(offset, 1.0)
    cdf_ws = ws.wasserstein_cdf(n0, n1, tf.linspace(-5.0, 6.0, 1024))
    np.testing.assert_allclose(cdf_ws.numpy(), offset, atol=1e-3)


def test_compare_ws_cdf():
    tf.random.set_seed(0)
    n0 = tfd.Gamma(0.5, 2.0)
    n1 = tfd.Gamma(2.0, 1.6)
    sample_ws = ws.wasserstein_sampled(n0, n1, 1024 * 128)
    cdf_ws = ws.wasserstein_cdf(n0, n1, tf.linspace(0.0, 10.0, 1001))
    np.testing.assert_allclose(sample_ws.numpy(), cdf_ws.numpy(), rtol=1e-2)


def test_compare_samples_to_point():
    tf.random.set_seed(0)
    batch_size = 4
    num_samples = 32
    point = tf.random.uniform(shape=(batch_size,))
    shape = (batch_size, num_samples)
    samples = tf.random.normal(shape=shape)
    np.testing.assert_allclose(
        ws.wassersein_samples_to_point(samples, point, axis=1).numpy(),
        ws.wasserstein_from_samples(
            samples, tf.tile(tf.expand_dims(point, axis=-1), (1, num_samples)), axis=1,
        ).numpy(),
        rtol=1e-6,
    )
