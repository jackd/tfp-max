import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def wasserstein_sampled(
    dist_a: tfd.Distribution, dist_b: tfd.Distribution, num_samples: int
):
    return wasserstein_from_samples(
        dist_a.sample(num_samples), dist_b.sample(num_samples), axis=0
    )


def wasserstein_cdf(
    dist_a: tfd.Distribution, dist_b: tfd.Distribution, abscissas: tf.Tensor,
):
    right = abscissas[1:]
    left = abscissas[:-1]
    ndims = dist_a.batch_shape.ndims
    shape = (-1, *((1,) * ndims))
    mid = tf.reshape((left + right) / 2, shape)
    width = tf.reshape(right - left, shape)
    cdf_a = dist_a.cdf(mid)
    cdf_b = dist_b.cdf(mid)
    return tf.math.reduce_sum(tf.abs(cdf_a - cdf_b) * width, axis=0)


def wasserstein_from_samples(
    samples_a: tf.Tensor, samples_b: tf.Tensor, axis=0, are_sorted=False
):
    if not are_sorted:
        samples_a = tf.sort(samples_a, axis=axis)
        samples_b = tf.sort(samples_b, axis=axis)

    return tf.math.reduce_mean(tf.abs(samples_a - samples_b), axis=axis)


def wassersein_samples_to_point(samples: tf.Tensor, point: tf.Tensor, axis=0):
    assert samples.shape.ndims == point.shape.ndims + 1
    return tf.math.reduce_mean(
        tf.abs(samples - tf.expand_dims(point, axis=axis)), axis=axis
    )
