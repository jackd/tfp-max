import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tfp_max.reduce_max import RepeatedMax, ReduceMax

tfd = tfp.distributions


def _check_cdf(dist_a: tfd.Distribution, dist_b: tfd.Distribution, x: tf.Tensor):
    cdf_a = dist_a.cdf(x)
    cdf_b = dist_b.cdf(x)
    np.testing.assert_allclose(cdf_a.numpy(), cdf_b.numpy(), rtol=1e-6, atol=1e-6)


def _check_probs(dist_a: tfd.Distribution, dist_b: tfd.Distribution, x: tf.Tensor):
    prob_a = dist_a.prob(x)
    prob_b = dist_b.prob(x)
    np.testing.assert_allclose(prob_a.numpy(), prob_b.numpy(), rtol=1e-6, atol=1e-6)


def test_repeated_max_cdf_1d():
    n = 5
    reduced = ReduceMax(tfd.Normal(tf.zeros((n,)), tf.ones((n,))))
    repeated = RepeatedMax(tfd.Normal(0.0, 1.0), n)

    x = tf.linspace(-2.0, 4.0, 21)
    _check_cdf(reduced, repeated, x)
    _check_probs(reduced, repeated, x)


def test_repeated_max_cdf_1d_rect():
    n = 5
    m = 3
    loc = tf.range(m, dtype=tf.float32)
    scale = 1.0 / tf.range(m, dtype=tf.float32)

    reduced = ReduceMax(
        tfd.Normal(
            scale=tf.tile(tf.expand_dims(scale, -1), [1, n]),
            loc=tf.tile(tf.expand_dims(loc, -1), [1, n]),
        )
    )
    repeated = RepeatedMax(tfd.Normal(scale=scale, loc=loc), [n] * m)

    x = tf.expand_dims(tf.linspace(-2.0, m * 5, 21), axis=-1)
    reduced.cdf(x)
    _check_cdf(reduced, repeated, x)
    _check_probs(reduced, repeated, x)


def test_repeated_max_cdf_ragged():
    repeats = [2, 5, 7]
    m = 3
    loc = tf.range(m, dtype=tf.float32)
    scale = 1.0 / tf.range(m, dtype=tf.float32)

    x = tf.linspace(-2.0, m * 5, 21)
    reduced_cdfs = []
    for i in range(m):
        normal = tfd.Normal(scale=scale[i], loc=loc[i])
        repeated = RepeatedMax(normal, repeats[i])
        reduced_cdfs.append(repeated.cdf(x))
    reduced_cdf = tf.stack(reduced_cdfs, axis=-1)

    repeated = RepeatedMax(tfd.Normal(scale=scale, loc=loc), repeats)
    repeated_cdf = repeated.cdf(tf.expand_dims(x, axis=-1))
    np.testing.assert_allclose(
        repeated_cdf.numpy(), reduced_cdf.numpy(), rtol=1e-6, atol=1e-6
    )
