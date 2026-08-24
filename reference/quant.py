"""Fixed-point quantization helpers."""

import numpy as np


def fold(entry, eps=1e-6):
    """Fold batch norm into a conv layer's weights and bias.

    `entry` is one layer's dict from weights.load(). Returns
    (weights, biases) that can go straight into layers.conv2d() with no
    batch-norm step afterwards -- the normalization has been absorbed
    into the numbers themselves.

    Darknet applies batch norm to the raw conv output, then adds the
    bias, so the full unfolded path is

        y = (conv(x, w) - mean) / (sqrt(var) + eps) * gamma + beta

    Every term except conv(x, w) is per-filter and constant at inference
    time, which is what makes the fold possible: rearranged, the whole
    expression is just another conv with different weights and a
    different bias. The hardware never has to know batch norm existed.

    On a layer with no batch norm the arrays pass through untouched.
    """
    # weights.load() only writes the batch-norm triple when the layer
    # has one, so the absent key is the test. This has to come before
    # the lookups below, not after: on the 1x1 output conv 'scales' is
    # missing and reading it raises. There is nothing to fold on such a
    # layer -- its biases are already ordinary conv biases, not beta.
    if 'scales' not in entry:
        return entry['weights'], entry['biases']

    gamma = entry['scales']
    beta = entry['biases']
    mean = entry['rolling_mean']
    var = entry['rolling_var']

    # Darknet writes sqrt(var) + eps, not the usual sqrt(var + eps).
    # The difference is small but it is a real divergence from every
    # other framework, and it matters once you are chasing
    # bit-exactness against darknet's own output.
    scale = gamma / (np.sqrt(var) + eps)

    # One scale per filter, so it needs three trailing axes to clear
    # c, k, k -- the same broadcasting-from-the-right rule as conv2d's
    # bias, two axes deeper. Every kernel belonging to filter f comes
    # out multiplied by scale[f].
    folded_weights = entry['weights'] * scale[:, None, None, None]

    # What is left of the docstring's equation once the division and
    # the gamma multiply have been pushed into the weights: the mean is
    # still subtracted, and it picks up the same scale on the way
    # through.
    folded_biases = beta - mean * scale

    return folded_weights, folded_biases
