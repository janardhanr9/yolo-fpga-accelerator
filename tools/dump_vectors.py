"""Write golden vectors for the RTL testbenches.

Runs the reference model on a calibration image, quantizes every tensor,
and writes the codes as .mem files that $readmemh can load. Also writes
a manifest with the per-layer scales and requantization constants, which
is what parameterizes the testbench.

    python tools/dump_vectors.py --layer 0

The point of these files is to make a hardware mismatch findable. When
the accelerator disagrees with the model, comparing per-layer turns "the
output is wrong" into "it diverges at layer 6", which is the difference
between an afternoon and a week.

Nothing here is the reference model -- it only arranges what the
reference model already computed. If a number looks wrong, check
reference/ first.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'reference'))

import cfg
import layers
import model
import quant

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, 'models', 'yolov2-tiny.cfg')
WEIGHTS = os.path.join(ROOT, 'models', 'yolov2-tiny.weights')
OUT = os.path.join(ROOT, 'vectors')

BITS = 16          # data width; 8 once the accelerator works
BIAS_BITS = 32     # bias lives in accumulator units, not output units
LEAKY_SLOPE = 0.1  # darknet's leaky ReLU
MULT_BITS = 16     # requantizer multiplier width


def build_config(spec, image_paths, bits=BITS):
    """Calibrate and pick every scale and requantization constant.

    Returns a dict keyed by conv layer index, each entry carrying the
    input, weight and output scales, and the (mult, shift) pair that
    converts that layer's accumulator into the next layer's input.
    """
    collects = []
    for path in image_paths:
        c = {}
        model.forward(model.preprocess(path, 416, 416), spec, c)
        collects.append(c)
    ranges = quant.calibrate(collects)

    config = {}
    # The network's input is the preprocessed image, which preprocess()
    # leaves in 0..1. After that, each conv's input range is whatever the
    # previous layer's output range was.
    prev = 1.0
    for entry in spec:
        if entry['type'] != 'convolutional':
            continue
        i = entry['index']
        in_scale = quant.choose_scale(prev, bits)
        w_scale = quant.choose_scale(float(np.abs(entry['weights']).max()), bits)
        out_scale = quant.choose_scale(ranges[i], bits)
        mult, shift = quant.requant_params(in_scale, w_scale, out_scale, MULT_BITS)

        # Leaky ReLU folds into the requantizer rather than costing its own
        # multiplier: the negative branch is the same M scaled by the slope,
        # so the datapath picks between two constants on the accumulator's
        # sign. Exact, because 0.1 lives in the constant rather than in a
        # shift approximation. The output conv is linear, so both branches
        # are the same there.
        slope = LEAKY_SLOPE if entry['activation'] == 'leaky' else 1.0
        mult_neg = int(round(mult * slope))

        config[i] = {
            'in_scale': in_scale, 'weight_scale': w_scale,
            'out_scale': out_scale, 'mult': mult, 'mult_neg': mult_neg,
            'shift': shift, 'range': ranges[i],
            'activation': entry['activation'],
            # The bias adds into the accumulator, so it is expressed in the
            # accumulator's units, not the output's.
            'bias_scale': in_scale * w_scale,
        }
        prev = ranges[i]
    return config


def to_words(codes, bits):
    """Turn signed integer codes into unsigned two's complement words.

    $readmemh reads hex digits, not signed decimals: a -1 in a 16-bit
    file has to appear as ffff, not -1. Verilog will happily load a
    file of negative-looking values and give you nonsense.
    """
    # A mask, not a conditional: two's complement already IS the low
    # `bits` bits of the value's infinite-precision representation, so
    # masking gives the right pattern for negatives and leaves
    # non-negatives alone. int64 first, because masking a narrower
    # signed type would wrap before the mask ever applied.
    return np.asarray(codes, dtype=np.int64) & ((1 << bits) - 1)


def write_mem(path, codes, bits):
    """Write `codes` as one hex word per line, for $readmemh."""
    words = to_words(np.asarray(codes).reshape(-1), bits)
    digits = (bits + 3) // 4
    with open(path, 'w') as f:
        for w in words.tolist():
            f.write(f'{w:0{digits}x}\n')
    return len(words)


def flatten_activation(x):
    """Flatten a (C, H, W) activation into the order hardware reads it.

    THIS IS AN ARCHITECTURE DECISION, not a formatting one. The order
    these bytes come out in is the order the DMA will stream them, and
    the order the line buffer expects to receive them.

    rtl/line_buffer.sv already commits to one answer: it documents its
    input as raster order with channel innermost,

        for row: for col: for channel

    which is C-innermost, or "HWC". A (C, H, W) numpy array is stored
    the other way round -- channel outermost -- so this is a real
    transpose, not a reshape.

    Getting it wrong produces a testbench that fails on the very first
    window with plausible-looking numbers, so it is worth being sure
    before generating a whole layer.
    """
    # (C, H, W) -> (H, W, C), so channel varies fastest. ascontiguousarray
    # forces the copy explicitly rather than leaving it to reshape to
    # decide whether it can return a view -- the whole point of this
    # function is the byte order, so it should not depend on numpy's
    # judgement about strides.
    return np.ascontiguousarray(x.transpose(1, 2, 0)).reshape(-1)


def flatten_weights(w):
    """Flatten an (N, C, K, K) weight tensor into the order hardware reads it.

    The other half of the same decision, and less constrained than the
    activation one because nothing has committed to an answer yet.

    What should drive it: the MAC array consumes one filter's kernels
    against one window at a time, and the window arrives with channel
    innermost. Whatever order makes the weights arrive in step with the
    window is the order that avoids a reorder buffer on the FPGA.

    Chosen: channel-major, [c][n][k][k].

    The line buffer emits one K x K window per (pixel, channel), walking
    channels innermost. So at any moment the array is working on a single
    input channel and needs that channel's kernels for every output
    filter -- N * K * K values. Channel-major puts exactly that block
    contiguously, so weights stream in lockstep with windows and each
    weight is read once per layer.

    Filter-major, which is how numpy stores them, would need the whole
    weight tensor re-read once per output filter, or a reorder buffer
    on the FPGA to hold what the stream delivered early.
    """
    # (N, C, K, K) -> (C, N, K, K). Only the leading two axes move; the
    # k x k kernel stays contiguous, which is what lets one filter's taps
    # land in the MAC array as a single burst.
    return np.ascontiguousarray(w.transpose(1, 0, 2, 3)).reshape(-1)


def dump_layer(spec, config, index, image, bits=BITS):
    """Write input, weights, bias and expected output for one conv layer."""
    entry = next(e for e in spec if e['index'] == index)
    conf = config[index]

    collect = {}
    model.forward(image, spec, collect)
    x = image if index == 0 else collect[index - 1]

    xq = quant.quantize(x, conf['in_scale'], bits)
    wq = quant.quantize(entry['weights'], conf['weight_scale'], bits)

    # The bias adds into the accumulator, so it is expressed in the
    # accumulator's units -- in_scale * weight_scale -- not the output's.
    # Quantizing it to `bits` would be wrong by the whole requantization
    # factor. 32 bits is ample: biases are small numbers, and the width
    # costs nothing because they load once per filter.
    bq = quant.quantize(entry['biases'], conf['bias_scale'], BIAS_BITS)

    # The integer forward pass, exactly as the datapath will run it.
    # conv2d is float64 here only as an arithmetic carrier: every operand
    # is a whole number and the widest partial sum is about 2^43, well
    # inside float64's 2^53 exact-integer range, so nothing is rounded.
    n = entry['weights'].shape[0]
    acc = layers.conv2d(xq.astype(np.float64), wq.astype(np.float64),
                        bq.astype(np.float64), entry['stride'],
                        entry['pad']).round().astype(np.int64)

    # Leaky ReLU is folded into the requantizer: same shift, a different
    # multiplier on the negative branch. Note this happens AFTER the bias,
    # which is where darknet applies the activation, and the sign test is
    # on the accumulator rather than the output -- requantization is
    # monotonic, so they agree, but the accumulator is what the hardware
    # has in hand at that point.
    pos = quant.requantize(acc, conf['mult'], conf['shift'], bits)
    neg = quant.requantize(acc, conf['mult_neg'], conf['shift'], bits)
    expected = np.where(acc < 0, neg, pos).astype(np.int32)

    os.makedirs(OUT, exist_ok=True)
    n_in = write_mem(os.path.join(OUT, f'layer{index:02d}_input.mem'),
                     flatten_activation(xq), bits)
    n_w = write_mem(os.path.join(OUT, f'layer{index:02d}_weights.mem'),
                    flatten_weights(wq), bits)
    n_b = write_mem(os.path.join(OUT, f'layer{index:02d}_bias.mem'),
                    bq, BIAS_BITS)
    n_o = write_mem(os.path.join(OUT, f'layer{index:02d}_expected.mem'),
                    flatten_activation(expected), bits)

    # How far the fixed-point path drifted from the float reference. This
    # is the number that says whether a hardware mismatch is worth
    # chasing: anything inside it is the format, anything outside is a bug.
    ref = layers.conv2d(x, entry['weights'], entry['biases'],
                        entry['stride'], entry['pad'])
    if entry['activation'] == 'leaky':
        ref = layers.leaky(ref)
    got = quant.dequantize(expected, conf['out_scale'])
    drift = float(np.abs(got - ref).max()) / float(np.abs(ref).max())

    print(f'  layer {index}: {n_in:,} input + {n_w:,} weight + {n_b:,} bias '
          f'+ {n_o:,} expected words')
    print(f'    codes used {len(np.unique(expected)):,} of {2**bits:,}, '
          f'drift from float {drift:.3%} of full scale')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--layer', type=int, default=0,
                    help='conv layer index to dump (default 0)')
    ap.add_argument('--image', default=os.path.join(ROOT, 'calib', 'dog.jpg'))
    ap.add_argument('--bits', type=int, default=BITS)
    args = ap.parse_args()

    net, spec = model.build(CFG, WEIGHTS)
    calib = sorted(glob.glob(os.path.join(ROOT, 'calib', '*.jpg')))
    if not calib:
        sys.exit('no calibration images in calib/')

    print(f'calibrating on {len(calib)} images at {args.bits} bits')
    config = build_config(spec, calib, args.bits)

    image = model.preprocess(args.image, int(net['width']), int(net['height']))
    dump_layer(spec, config, args.layer, image, args.bits)

    os.makedirs(OUT, exist_ok=True)
    manifest = os.path.join(OUT, 'manifest.json')
    with open(manifest, 'w') as f:
        json.dump({'bits': args.bits, 'mult_bits': MULT_BITS,
                   'bias_bits': BIAS_BITS,
                   'activation_layout': 'HWC, channel innermost',
                   'weight_layout': 'CNKK, channel major',
                   'leaky': 'folded into mult_neg, selected on acc sign',
                   'image': os.path.basename(args.image),
                   'layers': {str(k): v for k, v in config.items()}}, f, indent=2)
    print(f'  manifest -> {os.path.relpath(manifest, ROOT)}')


if __name__ == '__main__':
    main()
