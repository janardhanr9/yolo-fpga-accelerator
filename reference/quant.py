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


def calibrate(collects, percentile=None):
    """Measure each layer's activation range across a set of images.

    `collects` is a list of the dicts forward() fills when handed a
    collect argument -- one per calibration image. Returns a dict
    mapping layer index to a single absolute range for that layer.

    Taking already-collected tensors rather than image paths is
    deliberate: model.py imports this module, so importing model here
    would be a cycle. It also keeps the expensive part -- the forward
    passes -- in the caller's hands, where it can be done once and
    reused.

    With `percentile` set to, say, 99.9, the range is that percentile
    of the absolute values instead of the true maximum. That
    deliberately clips the largest handful of activations in exchange
    for finer resolution on everything else. Whether that trade is
    worth it is a real question and the answer differs per layer; the
    measured version of the question is what this function exists to
    answer.

    One image is not a calibration set. Across darknet's eight test
    images, ten of the fifteen layers peak higher than any single
    image suggests, two of them by a factor of two. A range measured
    too low does not degrade gracefully -- it clips, and the clipped
    values are exactly the strong activations the detector keys on.
    """
    # TODO 1: every collect dict has the same keys, so take the layer
    # indices from the first one rather than assuming a range.
    layers = sorted(collects[0])

    # TODO 2: for each layer, combine that layer's tensors from every
    # image into one range. For percentile=None this is just the
    # largest absolute value seen anywhere. For a percentile it is the
    # percentile over all images' values pooled together -- not the
    # mean of per-image percentiles, which is a different and wrong
    # number.
    ranges = {}
    for index in layers:
        if percentile is None:
            # Each image collapses to a single float before anything is
            # compared, so the whole set is never held at once.
            ranges[index] = max(float(np.abs(c[index]).max()) for c in collects)
        else:
            # A percentile is a rank statistic and has to see every value
            # together, so this branch really does pool the set. ravel()
            # first: concatenate joins along an axis otherwise, which is a
            # different operation that happens to look similar.
            pooled = np.concatenate([np.abs(c[index]).ravel() for c in collects])
            ranges[index] = float(np.percentile(pooled, percentile))

    return ranges


def choose_scale(absmax, bits=8):
    """Return the scale factor for a tensor with range +-absmax.

    A quantized tensor stores integer codes; the scale is what turns a
    code back into a real value:  value = code * scale. Picking it is
    the whole game. Too large and the codes bunch up near zero, wasting
    resolution; too small and the extremes clip.

    Signed `bits`-bit codes run from -2^(bits-1) to 2^(bits-1) - 1. The
    positive limit is the smaller of the two, so sizing against it is
    what keeps the largest value representable.

    Note what this deliberately does NOT do: round the scale to a power
    of two. A power-of-two scale makes dequantization a shift, which is
    free in hardware, but it can only land on octave boundaries -- a
    tensor peaking at 16.04 gets a range of 32 and wastes half its
    codes. Measured on this network that costs about 30% of the code
    space on average and a great deal of recall. The multiplier it
    takes to avoid that is cheaper than it looks, because it runs once
    per output element rather than once per MAC.
    """
    # TODO 3: return the scale. Guard the degenerate all-zeros tensor,
    # which would otherwise hand back a scale of zero and turn every
    # later division into a NaN.
    if absmax == 0:
        return 1.0
    return absmax / (2**(bits-1) - 1)


def quantize(v, scale, bits=8):
    """Convert real values to integer codes, saturating at the limits.

    Returns an integer array, not floats. That distinction matters:
    these codes are what a .mem file or an AXI stream actually carries,
    and keeping them as integers here means a wrong scale shows up as
    obviously-wrong data rather than as slightly-off floats.

    Round rather than truncate. Truncation biases every value the same
    direction, and across a fifteen-layer stack that bias compounds
    into a drift large enough to erase the detections entirely.
    """
    # The limits are not symmetric: two's complement gives one more code
    # below zero than above. Using the same bound for both silently
    # throws away a code and nothing will ever flag it.
    lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1

    # Round first, then clip. A value landing at 127.6 must become 128
    # and then saturate to 127, which is the order the hardware applies
    # them. np.round is banker's rounding on exact halves, a hair
    # different from the round-half-up an adder implements; it matters
    # only for values sitting exactly on .5, and requantize() is where
    # the hardware-exact version lives.
    codes = np.round(v / scale)

    # int32 rather than the exact width: an int8 array would wrap on
    # anything the clip missed, turning a bounds bug into the silent
    # wraparound failure instead of an obviously out-of-range number.
    return np.clip(codes, lo, hi).astype(np.int32)


def dequantize(codes, scale):
    """Turn integer codes back into real values.

    The inverse of quantize() except for what quantize() threw away:
    everything below the scale's resolution, and anything that
    saturated. Round-tripping a tensor through both is the cheapest way
    to see what a format costs before running a whole network through
    it.
    """
    # float32 explicitly. codes is an integer array and scale a Python
    # float, so the natural result of the multiply is float64, which
    # would quietly promote every downstream array it touches.
    return (codes * np.float32(scale)).astype(np.float32)


def requant_params(in_scale, weight_scale, out_scale, bits=16):
    """Turn three float scales into the integer multiplier and shift.

    This is the bridge between the study and the hardware. Convolving
    quantized inputs with quantized weights gives an accumulator whose
    real value is  acc * in_scale * weight_scale. Emitting the next
    layer's input means expressing that in units of out_scale, so
    every accumulator has to be multiplied by

        M = in_scale * weight_scale / out_scale

    M is a real number, usually well below 1, and no FPGA multiplies by
    0.0037. So M is approximated as an integer over a power of two:

        M ~= mult / 2^shift

    which the hardware evaluates as one integer multiply followed by
    one arithmetic shift. Returns (mult, shift) with `mult` fitting in
    `bits` bits.

    Choosing shift as large as `mult` allows is what makes the
    approximation tight: every extra bit of shift is another bit of
    precision in M. Too large and mult overflows its width; too small
    and M is coarse enough to shift layer outputs visibly.
    """
    # TODO 6: form M from the three scales.
    M = in_scale * weight_scale / out_scale

    # TODO 7: find the largest shift for which round(M * 2^shift) still
    # fits in a signed `bits`-bit integer, then return that mult and
    # shift. Think about what happens if M is itself larger than 1 --
    # it can be, when a layer's output range is narrower than its
    # input's, and the shift search has to still terminate.
    hi = 2 ** (bits - 1) - 1
    shift = int(np.floor(np.log2(hi / M)))
    if shift < 0:
        raise ValueError(
            f'M={M:g} needs more than {bits} bits of multiplier; '
            f'out_scale is too fine relative to in_scale * weight_scale')
    mult = round(M * 2**shift)

    return mult, shift


def requantize(acc, mult, shift, bits=8):
    """The hardware requantization stage, in software.

    Takes a wide accumulator and produces the next layer's `bits`-bit
    input: multiply by `mult`, round, shift right by `shift`, saturate.
    This mirrors what the RTL does gate for gate, so a mismatch between
    this and the accelerator is a real bug in one of them rather than a
    modelling artefact.

    The two knobs here are both traps.

    Rounding: an arithmetic shift right floors toward negative
    infinity, so shifting alone biases every single value the same
    direction. Adding half an LSB before the shift is what turns it
    into rounding, and it is one adder.

    Saturation: a value that will not fit must be pinned at the limit,
    not allowed to wrap. Wrapping sends the largest activation in a
    channel to the most negative one, which is catastrophic but sparse
    -- the output stays plausible, and the network still detects
    things, just worse. It is invisible until a format is tightened
    enough to overflow, which is precisely when you are optimizing and
    least expecting a regression.
    """
    # int64 throughout: the accumulator is already ~45 bits at 16-bit
    # data and mult adds another 15, so anything narrower wraps. numpy
    # raises a RuntimeWarning when a SCALAR multiply overflows but says
    # nothing at all when an array one does -- and this is always an
    # array. So the width is asserted rather than assumed.
    acc = np.asarray(acc, dtype=np.int64)
    span = int(np.abs(acc).max()) if acc.size else 0
    assert span * abs(int(mult)) < 2**62, (
        f'acc*mult needs more than 63 bits (acc up to {span}, mult {mult}); '
        f'narrow the multiplier or truncate the accumulator first')

    scaled = acc * np.int64(mult)

    # Half an LSB before the shift is what makes this rounding rather
    # than truncation. numpy's >> on signed integers is an arithmetic
    # shift, so it floors toward negative infinity exactly as the RTL's
    # >>> does; adding half first turns that floor into round-half-up.
    #
    # Round-half-up, not round-half-even: an exact .5 goes toward
    # positive infinity for both signs. That differs from np.round in
    # quantize(), and deliberately so -- this function models the
    # hardware, and one adder plus a shift is what the hardware can
    # afford.
    if shift > 0:
        scaled = (scaled + (np.int64(1) << (shift - 1))) >> shift

    # Saturate, never wrap. A wrapped value sends the largest activation
    # in a channel to the most negative one, which stays plausible
    # enough to survive a smoke test and quietly costs real accuracy.
    lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return np.clip(scaled, lo, hi).astype(np.int32)
