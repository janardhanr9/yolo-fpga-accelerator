"""Layer operations for the reference model."""

import numpy as np


def conv2d(x, weights, biases, stride=1, pad=0):
    """Convolve `x` with `weights` and add `biases`.

    x is (c, h, w) float32. weights is (n, c, k, k) exactly as
    weights.load() returns it. Output is (n, h_out, w_out).

    `pad` is a pixel count the caller has already resolved. Darknet's cfg
    `pad=1` is a flag meaning "pad to same", not one pixel: it works out
    to size // 2, so the 1x1 output conv pads nothing despite saying
    pad=1.
    """
    n, c, k, _ = weights.shape

    c_in, h_in, w_in = x.shape

    h_out = (h_in + 2*pad - k) // stride + 1
    w_out = (w_in + 2*pad - k) // stride + 1

    padded = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))

    # im2col: flatten every receptive field into a column so the whole
    # convolution collapses into one matrix multiply -- the same GEMM the
    # PE array will run in hardware.
    cols = np.zeros((c, k, k, h_out, w_out), dtype=np.float32)
    for i in range(k):
        for j in range(k):
            # Kernel tap (i, j) sees one shifted, strided plane of the
            # input -- the same plane for every output position, so the
            # whole tap is one slice rather than a loop over pixels.
            cols[:, i, j] = padded[:, i : i + h_out*stride : stride, j : j + w_out*stride : stride]

    # Both reshapes are free views: cols already carries c, k, k as its
    # leading axes in the same order the kernels do, so flattening them
    # lands both operands on a shared c*k*k axis for the multiply.
    reshaped_cols = cols.reshape(c*k*k, h_out*w_out)
    reshaped_weights = weights.reshape(n, c*k*k)

    out = (reshaped_weights @ reshaped_cols).reshape(n, h_out, w_out)

    # Broadcasting aligns from the right, so a bare (n,) bias would try to
    # match w_out. The trailing 1s put n back on the filter axis.
    reshaped_biases = biases.reshape(n, 1, 1)

    out = out + reshaped_biases

    return out


def maxpool(x, size, stride):
    """Max-pool `x` with a size x size window.

    Darknet pads by size - 1, entirely on the bottom and right edges,
    using -inf so a padded cell can never win a max. For the stride-2
    pools that padding is never reached; layer 12 (size 2, stride 1) is
    the one place it matters, and it is what holds that layer's output at
    13x13 instead of 12x12.
    """
    c, h, w = x.shape
    pad = size - 1

    # `pad` appears once here, not twice as in conv2d, because it is all
    # on one side.
    h_out = (h + pad - size) // stride + 1
    w_out = (w + pad - size) // stride + 1

    padded = np.pad(x, ((0, 0), (0, pad), (0, pad)), constant_values=-np.inf)

    # Same gather as conv2d, different reduction: channels never mix when
    # pooling, so c rides along untouched and only the two window axes
    # collapse.
    windows = np.zeros((c, size, size, h_out, w_out), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            # Window position (i, j) sees one shifted, strided plane of
            # the input -- the same plane for every output position.
            windows[:, i, j] = padded[:, i : i + h_out*stride : stride, j : j + w_out*stride : stride]

    out = windows.max(axis=(1, 2))

    return out


def leaky(x, slope=0.1):
    """Darknet's leaky ReLU: x where positive, slope * x where not."""
    return np.where(x > 0, x, slope * x)
