from typing import Union
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class ScalarMax(tfd.Distribution):
    """Max of a distribution and a scalar."""

    def __init__(self, distribution: tfd.Distribution, lower_bound: tf.Tensor):
        self._distribution = distribution
        self._lower_bound = lower_bound
        super().__init__(
            dtype=distribution.dtype,
            reparameterization_type=distribution.reparameterization_type,
            validate_args=False,
            parameters=dict(locals()),
            allow_nan_stats=distribution.allow_nan_stats,
            name="Max{}".format(distribution.name),
        )

    def _sample_n(self, n, seed=None, **kwargs):
        samples = self._distribution.sample(n, seed=seed, **kwargs)
        return tf.maximum(samples, self._lower_bound)

    def _cdf(self, value, **kwargs):
        cdf = self._distribution.cdf(value, **kwargs)
        return tf.where(value < self._lower_bound, tf.zeros_like(cdf), cdf)

    def _prob(self, value, **kwargs):
        probs = self._distribution.prob(value, **kwargs)
        return tf.where(
            value > self._lower_bound,
            probs,
            tf.where(
                value < self._lower_bound,
                tf.zeros_like(probs),
                tf.ones_like(probs) * np.inf,
            ),
        )


class RepeatedMax(tfd.Distribution):
    """
    The distribution resulting from taking the max of repeated samples.

    If x_i ~ X and z = max_i x_i ~ Z, then this represents Z.

    Sampling is simply the max of sampling `repeats` samples of X.

    CDF is calculated as follows:
    CDF_z(Z) = Pr(z < Z) = Pr(x_1 < Z) * Pr(x_2 < Z) * ...
                      = CDF_x(Z) ** repeats
    """

    def __init__(
        self,
        distributions: tfd.Distribution,
        repeats: Union[int, tf.Tensor, np.ndarray],
    ):
        self._distributions = distributions
        self._repeats = tf.convert_to_tensor(repeats, dtype=tf.int64)
        self._max_repeats = tf.math.reduce_max(self._repeats)
        self._repeats_float = tf.cast(self._repeats, self._distributions.dtype)
        self._repeats.shape.assert_has_rank(self._distributions.batch_shape.ndims)
        assert self._repeats.shape.ndims in (0, 1)
        super().__init__(
            dtype=distributions.dtype,
            reparameterization_type=distributions.reparameterization_type,
            validate_args=False,
            parameters=dict(locals()),
            allow_nan_stats=distributions.allow_nan_stats,
            name="Max{}".format(distributions.name),
        )

    def _sample_n(self, n, seed=None, **kwargs):
        if self._repeats.shape.ndims == 0:
            samples = self._distributions.sample(
                n, self._max_repeats, seed=seed, **kwargs
            )
        else:
            samples = self._distributions.sample(
                (n, self._max_repeats), seed=seed, **kwargs
            )
            samples = tf.RaggedTensor.from_tensor(samples, self._repeats)
        return tf.math.reduce_max(samples, axis=1)

    def _cdf(self, value, **kwargs):
        cdf = self._distributions.cdf(value, **kwargs)
        return cdf ** self._repeats_float

    def _log_cdf(self, value, **kwargs):
        log_cdf = self._distributions.cdf(value, **kwargs)
        return log_cdf * self._shaped_repeats(log_cdf)

    def _prob(self, value, **kwargs):
        pdfs = self._distributions.prob(value, **kwargs)
        cdfs = self._distributions.cdf(value, **kwargs)
        repeats = self._repeats_float
        return repeats * pdfs * cdfs ** (repeats - 1)


class ReduceMax(tfd.Distribution):
    """
    The distribution resulting from maximizing other independent distributions.

    If x ~ X, y ~ Y and z = max(x, y) ~ Z, then this represents Z.

    Sampling is simply the max of sampling X and Y.

    CDF is calculated as follows:

    CDF(Z) = Pr(z < Z) = Pr(z < X) * Pr(z < Y)
                       = CDF(X) * CDF(Y)

    prob is calculated by differentiating CDF.

    This implementation generalizes this to a maximum of an arbitrary number
    of independent distributions.

    Only `prob`, `cdf` and `sample` currently implemented.
    """

    def __init__(self, distributions: tfd.Distribution, axis: int = -1):
        self._distributions = distributions
        self._axis = axis
        super().__init__(
            dtype=distributions.dtype,
            reparameterization_type=distributions.reparameterization_type,
            validate_args=False,
            parameters=dict(locals()),
            allow_nan_stats=distributions.allow_nan_stats,
            name="Max{}".format(distributions.name),
        )

    def _batch_shape_tensor(self):
        base = self._distributions.batch_shape_tensor()
        axis = self._axis if self._axis >= 0 else self._axis + base.shape[0]
        return tf.concat([base[:axis], base[axis + 1 :]], axis=0)

    def _batch_shape(self):
        base = self._distributions.batch_shape
        if base is None:
            return None
        axis = self._axis if self._axis >= 0 else self._axis + base.ndims
        return tf.TensorShape((*base[:axis], *base[axis + 1 :]))

    def _sample_n(self, n, seed=None, **kwargs):
        axis = self._axis if self._axis < 0 else (self._axis + 1)
        sample = self._distributions.sample(n, seed=seed, **kwargs)
        return tf.math.reduce_max(sample, axis=axis)

    def _cdf(self, value, **kwargs):
        value = tf.expand_dims(value, axis=self._axis)
        return tf.math.reduce_prod(
            self._distributions.cdf(value, **kwargs), axis=self._axis,
        )

    def _log_cdf(self, value, **kwargs):
        value = tf.expand_dims(value, axis=self._axis)
        return tf.math.reduce_sum(
            self._distributions.log_cdf(value, **kwargs), axis=self._axis
        )

    def _prob(self, value, **kwargs):
        value = tf.expand_dims(value, axis=self._axis)
        pdfs = self._distributions.prob(value, **kwargs)
        cdfs = self._distributions.cdf(value, **kwargs)
        quotient = pdfs / cdfs
        probs = tf.math.reduce_prod(cdfs, axis=self._axis) * tf.math.reduce_sum(
            quotient, axis=self._axis
        )
        probs = tf.where(tf.math.is_nan(probs), tf.zeros_like(probs), probs)
        return probs
