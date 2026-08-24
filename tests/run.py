"""Self-contained test runner for the reference model.

Run with `python tests/run.py`. Exits nonzero if anything fails, so it
drops straight into a Makefile next to the RTL sims later.

Deliberately no pytest: this project's only dependencies are numpy and
pillow, and a golden reference model that needs a test framework
installed to prove itself is one more thing to get wrong on a fresh
machine.

Tests needing the cfg/weights (both gitignored, see README) skip rather
than fail when the files are absent.
"""

import math
import os
import sys
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'reference'))

import cfg
import layers
import model
import quant
import weights

CFG = os.path.join(ROOT, 'models', 'yolov2-tiny.cfg')
WEIGHTS = os.path.join(ROOT, 'models', 'yolov2-tiny.weights')

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


class Skip(Exception):
    """Raised to skip a test whose inputs are not present."""


def need_model():
    """Return a built (net, spec), or skip if the weights are missing."""
    if not (os.path.exists(CFG) and os.path.exists(WEIGHTS)):
        raise Skip('models/yolov2-tiny.{cfg,weights} not present')
    return model.build(CFG, WEIGHTS)


def close(got, want, tol, what):
    err = float(np.abs(np.asarray(got) - np.asarray(want)).max())
    assert err <= tol, f'{what}: max error {err:.3e} exceeds {tol:.3e}'
    return err


# --------------------------------------------------------------- cfg


@test
def cfg_parses_the_expected_stack():
    if not os.path.exists(CFG):
        raise Skip('cfg not present')
    net, lay = cfg.parse(CFG)
    assert (int(net['width']), int(net['height']), int(net['channels'])) == (416, 416, 3)
    kinds = [l['type'] for l in lay]
    assert kinds.count('convolutional') == 9, kinds.count('convolutional')
    assert kinds.count('maxpool') == 6, kinds.count('maxpool')
    assert kinds.count('region') == 1
    # The one stride-1 pool is what holds the last stage at 13x13.
    pools = [(int(l['size']), int(l['stride'])) for l in lay if l['type'] == 'maxpool']
    assert pools.count((2, 1)) == 1, pools


@test
def cfg_threads_channels_through_pools():
    if not os.path.exists(CFG):
        raise Skip('cfg not present')
    net, lay = cfg.parse(CFG)
    cfg.thread_channels(net, lay)
    convs = [l for l in lay if l['type'] == 'convolutional']
    assert convs[0]['channels_in'] == 3
    # Each conv's input count must be the previous conv's filter count,
    # regardless of how many pools sit between them.
    for prev, cur in zip(convs, convs[1:]):
        assert cur['channels_in'] == int(prev['filters']), \
            f"{cur['channels_in']} != {prev['filters']}"


# ----------------------------------------------------------- weights


@test
def weights_consume_the_file_exactly():
    # load() asserts on EOF internally; reaching here at all proves every
    # layer's shape arithmetic. This also pins the header.
    if not (os.path.exists(CFG) and os.path.exists(WEIGHTS)):
        raise Skip('weights not present')
    net, lay = cfg.parse(CFG)
    cfg.thread_channels(net, lay)
    header, params = weights.load(WEIGHTS, lay)
    assert (header['major'], header['minor']) == (0, 2), header
    assert len(params) == 9, len(params)
    for i, entry in params.items():
        n, c, k, k2 = entry['weights'].shape
        assert k == k2
        assert entry['biases'].shape == (n,)
        if 'scales' in entry:
            for key in ('scales', 'rolling_mean', 'rolling_var'):
                assert entry[key].shape == (n,), key
    # exactly one layer without batch norm: the 1x1 output conv
    plain = [i for i, e in params.items() if 'scales' not in e]
    assert plain == [14], plain


# ------------------------------------------------------------ layers


def brute_conv(x, w, b, stride, pad):
    """Nested-loop convolution. Slow, obvious, and independent of im2col."""
    n, c, k, _ = w.shape
    _, h_in, w_in = x.shape
    h_out = (h_in + 2*pad - k)//stride + 1
    w_out = (w_in + 2*pad - k)//stride + 1
    p = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    out = np.zeros((n, h_out, w_out), dtype=np.float32)
    for f in range(n):
        for oy in range(h_out):
            for ox in range(w_out):
                acc = 0.0
                for ch in range(c):
                    for ky in range(k):
                        for kx in range(k):
                            acc += p[ch, oy*stride+ky, ox*stride+kx] * w[f, ch, ky, kx]
                out[f, oy, ox] = acc + b[f]
    return out


@test
def conv2d_matches_nested_loops():
    rng = np.random.default_rng(11)
    # stride 2 is here even though no yolov2-tiny conv uses it: it is the
    # only thing that proves the *stride in the im2col slice is right.
    configs = [
        dict(c=5, k=3, n=4, stride=1, pad=1, h=9, w=9),
        dict(c=5, k=3, n=4, stride=2, pad=1, h=9, w=9),
        dict(c=3, k=1, n=7, stride=1, pad=0, h=6, w=6),
        dict(c=2, k=3, n=3, stride=1, pad=0, h=7, w=5),
        dict(c=4, k=3, n=2, stride=2, pad=0, h=8, w=8),
    ]
    for cf in configs:
        x = rng.standard_normal((cf['c'], cf['h'], cf['w']), dtype=np.float32)
        w = rng.standard_normal((cf['n'], cf['c'], cf['k'], cf['k']), dtype=np.float32)
        b = rng.standard_normal(cf['n'], dtype=np.float32)
        got = layers.conv2d(x, w, b, cf['stride'], cf['pad'])
        want = brute_conv(x, w, b, cf['stride'], cf['pad'])
        assert got.shape == want.shape, f"{cf}: {got.shape} != {want.shape}"
        close(got, want, 1e-4, f'conv2d {cf}')
        assert got.dtype == np.float32


@test
def conv2d_bias_lands_on_the_filter_axis():
    # A zero input isolates the bias: every output pixel of filter f must
    # be exactly b[f]. Catches a bias broadcast onto the wrong axis, which
    # a random input would hide in the noise.
    rng = np.random.default_rng(4)
    n, c, k = 6, 3, 3
    x = np.zeros((c, 8, 8), dtype=np.float32)
    w = rng.standard_normal((n, c, k, k), dtype=np.float32)
    b = rng.standard_normal(n, dtype=np.float32)
    out = layers.conv2d(x, w, b, 1, 1)
    for f in range(n):
        assert np.all(out[f] == b[f]), f'filter {f} bias not uniform'


def brute_maxpool(x, size, stride):
    c, h, w = x.shape
    pad = size - 1
    h_out = (h + pad - size)//stride + 1
    w_out = (w + pad - size)//stride + 1
    p = np.pad(x, ((0, 0), (0, pad), (0, pad)), constant_values=-np.inf)
    out = np.full((c, h_out, w_out), -np.inf, dtype=np.float32)
    for ch in range(c):
        for oy in range(h_out):
            for ox in range(w_out):
                for ky in range(size):
                    for kx in range(size):
                        out[ch, oy, ox] = max(out[ch, oy, ox],
                                              p[ch, oy*stride+ky, ox*stride+kx])
    return out


@test
def maxpool_matches_nested_loops():
    rng = np.random.default_rng(12)
    # All-negative inputs on purpose: a zero-filled pad would win a max
    # against real data and this is the only way to catch it. Post-leaky
    # activations really are often negative.
    for size, stride, h, w, negative in [
        (2, 2, 8, 8, False),
        (2, 2, 8, 8, True),
        (2, 1, 13, 13, True),     # the layer 11 case
        (3, 1, 13, 13, True),
        (2, 2, 7, 5, True),       # odd input, padding actually reached
    ]:
        x = rng.standard_normal((4, h, w), dtype=np.float32)
        if negative:
            x = -np.abs(x) - np.float32(1.0)
        got = layers.maxpool(x, size, stride)
        want = brute_maxpool(x, size, stride)
        assert got.shape == want.shape, f'{size}/{stride}: {got.shape} != {want.shape}'
        close(got, want, 0.0, f'maxpool size={size} stride={stride} neg={negative}')


@test
def maxpool_holds_13x13_through_the_stride1_pool():
    x = np.random.default_rng(1).standard_normal((512, 13, 13), dtype=np.float32)
    assert layers.maxpool(x, 2, 1).shape == (512, 13, 13)


@test
def leaky_slopes_only_the_negatives():
    x = np.array([-4.0, -0.5, 0.0, 0.5, 4.0], dtype=np.float32)
    got = layers.leaky(x)
    close(got, [-0.4, -0.05, 0.0, 0.5, 4.0], 1e-7, 'leaky')
    assert got.dtype == np.float32
    # slope is a parameter, not a constant baked into the branch
    close(layers.leaky(np.float32([-2.0]), 0.25), [-0.5], 1e-7, 'leaky slope')


# ------------------------------------------------------------- quant


@test
def fold_reproduces_the_unfolded_batch_norm():
    net, spec = need_model()
    _, lay = cfg.parse(CFG)
    cfg.thread_channels(net, lay)
    _, params = weights.load(WEIGHTS, lay)
    rng = np.random.default_rng(0)
    for i, entry in params.items():
        n, c, k, _ = entry['weights'].shape
        x = rng.standard_normal((c, 9, 9), dtype=np.float32)
        pad = k // 2
        fw, fb = quant.fold(entry)
        folded = layers.conv2d(x, fw, fb, 1, pad)
        if 'scales' in entry:
            z = layers.conv2d(x, entry['weights'], np.zeros(n, dtype=np.float32), 1, pad)
            a = entry['scales'] / (np.sqrt(entry['rolling_var']) + 1e-6)
            unfolded = ((z - entry['rolling_mean'].reshape(n, 1, 1)) * a.reshape(n, 1, 1)
                        + entry['biases'].reshape(n, 1, 1))
        else:
            unfolded = layers.conv2d(x, entry['weights'], entry['biases'], 1, pad)
        # Relative to layer RMS, not to individual values: activations cross
        # zero, and a near-zero denominator makes relative error meaningless.
        rms = float(np.sqrt((unfolded**2).mean()))
        err = float(np.abs(folded - unfolded).max()) / rms
        assert err < 1e-4, f'layer {i}: err/rms {err:.3e}'
        assert fw.dtype == np.float32 and fb.dtype == np.float32


@test
def fold_passes_through_layers_without_batch_norm():
    net, spec = need_model()
    _, lay = cfg.parse(CFG)
    cfg.thread_channels(net, lay)
    _, params = weights.load(WEIGHTS, lay)
    entry = params[14]
    assert 'scales' not in entry
    fw, fb = quant.fold(entry)
    # identity, not just equality: nothing was copied or rescaled
    assert fw is entry['weights']
    assert fb is entry['biases']


# ------------------------------------------------------------- model


@test
def resolve_pad_treats_pad_as_a_flag():
    assert model.resolve_pad({'size': '3', 'pad': '1'}) == 1
    # the trap: pad=1 on a 1x1 conv means zero padding, not one pixel
    assert model.resolve_pad({'size': '1', 'pad': '1'}) == 0
    assert model.resolve_pad({'size': '5', 'pad': '1'}) == 2
    assert model.resolve_pad({'size': '3', 'pad': '0'}) == 0
    assert model.resolve_pad({'size': '3'}) == 0
    # explicit padding wins over the flag
    assert model.resolve_pad({'size': '3', 'pad': '1', 'padding': '4'}) == 4


@test
def forward_produces_the_documented_shapes():
    net, spec = need_model()
    expected = [
        (0, (16, 416, 416)), (1, (16, 208, 208)), (2, (32, 208, 208)),
        (3, (32, 104, 104)), (4, (64, 104, 104)), (5, (64, 52, 52)),
        (6, (128, 52, 52)), (7, (128, 26, 26)), (8, (256, 26, 26)),
        (9, (256, 13, 13)), (10, (512, 13, 13)), (11, (512, 13, 13)),
        (12, (1024, 13, 13)), (13, (512, 13, 13)), (14, (425, 13, 13)),
    ]
    x = np.zeros((3, 416, 416), dtype=np.float32)
    collect = {}
    out = model.forward(x, spec, collect)
    assert out.shape == (425, 13, 13)
    assert out.dtype == np.float32 and np.isfinite(out).all()
    assert len(collect) == len(expected), f'{len(collect)} tensors collected'
    for idx, shape in expected:
        assert collect[idx].shape == shape, f'layer {idx}: {collect[idx].shape} != {shape}'


@test
def forward_rejects_unknown_layers():
    fake = [{'type': 'shortcut', 'index': 0}]
    try:
        model.forward(np.zeros((1, 4, 4), dtype=np.float32), fake)
    except ValueError:
        return
    raise AssertionError('unimplemented layer was silently skipped')


# ------------------------------------------------------------ decode


@test
def decode_matches_darknet_entry_indexing():
    net, spec = need_model()
    reg = next(e for e in spec if e['type'] == 'region')
    rng = np.random.default_rng(7)
    out = rng.standard_normal((425, 13, 13), dtype=np.float32) * 2
    flat = out.reshape(-1)
    W = H = 13
    stride = W*H
    block = reg['coords'] + 1 + reg['classes']
    thresh = 0.05
    sig = lambda v: 1.0/(1.0 + math.exp(-v))

    # np.nonzero over (anchor, class, y, x) walks C order, so generate the
    # reference in that same order and compare positionally.
    want = []
    for n in range(reg['num']):
        for cls in range(reg['classes']):
            for i in range(H):
                for j in range(W):
                    base = n*stride*block + i*W + j
                    logits = [flat[base + (5+q)*stride] for q in range(reg['classes'])]
                    m = max(logits)
                    e = [math.exp(v - m) for v in logits]
                    conf = sig(flat[base + 4*stride]) * e[cls] / sum(e)
                    if conf > thresh:
                        want.append((
                            (j + sig(flat[base]))/W,
                            (i + sig(flat[base + stride]))/H,
                            math.exp(flat[base + 2*stride])*reg['anchors'][n][0]/W,
                            math.exp(flat[base + 3*stride])*reg['anchors'][n][1]/H,
                            cls, conf))

    got = model.decode(out, spec, thresh=thresh)
    assert len(got) == len(want), f'{len(got)} detections vs {len(want)}'
    for g, w in zip(got, want):
        assert g['class'] == w[4], 'detection order diverged'
        close(g['box'], w[:4], 1e-5, 'decode box')
        close(g['score'], w[5], 1e-5, 'decode score')


@test
def sigmoid_is_stable_at_the_extremes():
    v = np.array([-120, -90, -20, -1, 0, 1, 20, 90, 120], dtype=np.float32)
    naive = 1.0 / (1.0 + np.exp(-v.astype(np.float64)))
    close(model._sigmoid(v), naive, 1e-7, 'sigmoid')
    # the naive float32 form overflows here; this one must not
    assert np.isfinite(model._sigmoid(v)).all()
    assert model._sigmoid(np.float32([-120]))[0] == 0.0
    assert model._sigmoid(np.float32([120]))[0] == 1.0


@test
def softmax_normalises_along_the_class_axis():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((5, 80, 13, 13), dtype=np.float32) * 30
    s = model._softmax(x, axis=1)
    close(s.sum(axis=1), np.ones((5, 13, 13)), 1e-5, 'softmax sum')
    assert np.isfinite(s).all(), 'softmax overflowed'
    assert (s >= 0).all()


@test
def iou_matches_mask_overlap():
    rng = np.random.default_rng(3)
    g = np.linspace(0, 1, 1500)
    X, Y = np.meshgrid(g, g, indexing='xy')
    cell = 1.0/1500
    for _ in range(60):
        def draw():
            w, h = rng.uniform(0.05, 0.4, 2)
            return (rng.uniform(w/2, 1-w/2), rng.uniform(h/2, 1-h/2), w, h)
        a, b = draw(), draw()
        m = lambda q: (np.abs(X-q[0]) <= q[2]/2) & (np.abs(Y-q[1]) <= q[3]/2)
        ma, mb = m(a), m(b)
        union = (ma | mb).sum()
        want = (ma & mb).sum()/union if union else 0.0
        # tolerance is the grid's own discretisation, a few cells' worth
        close(model.iou(a, b), want, 8*cell, f'iou {a} {b}')
    assert model.iou((0.1, 0.1, 0.1, 0.1), (0.9, 0.9, 0.1, 0.1)) == 0.0
    close(model.iou((0.5, 0.5, 0.2, 0.2), (0.5, 0.5, 0.2, 0.2)), 1.0, 1e-9, 'iou self')


@test
def nms_suppresses_within_a_class_only():
    dets = [
        {'box': (0.50, 0.50, 0.2, 0.2), 'class': 0, 'score': 0.9},
        {'box': (0.51, 0.50, 0.2, 0.2), 'class': 0, 'score': 0.8},   # suppressed
        {'box': (0.50, 0.50, 0.2, 0.2), 'class': 1, 'score': 0.7},   # other class
        {'box': (0.10, 0.10, 0.2, 0.2), 'class': 0, 'score': 0.6},   # disjoint
    ]
    kept = model.nms(dets, thresh=0.45)
    assert len(kept) == 3, [d['score'] for d in kept]
    scores = [d['score'] for d in kept]
    assert 0.8 not in scores, 'overlapping same-class box survived'
    assert scores == sorted(scores, reverse=True), 'output not score-ordered'
    # a perfectly overlapping pair in the same class must collapse to one
    pair = [{'box': (0.5, 0.5, 0.3, 0.3), 'class': 2, 'score': s} for s in (0.9, 0.5)]
    assert len(model.nms(pair)) == 1

    # Exercise the DEFAULT threshold, not just an explicit one: two boxes
    # overlapping well below it must both survive. Without this, changing
    # the default is invisible to the whole suite -- real objects in a
    # crowd overlap, and a default that merges them loses detections.
    touching = [{'box': (0.450, 0.5, 0.2, 0.2), 'class': 3, 'score': 0.9},
                {'box': (0.546, 0.5, 0.2, 0.2), 'class': 3, 'score': 0.8}]
    assert 0.20 < model.iou(touching[0]['box'], touching[1]['box']) < 0.45, \
        'test fixture no longer straddles the default threshold'
    assert len(model.nms(touching)) == 2, 'default threshold merged distinct boxes'

    # ...and pinned from the other side: a heavily overlapping pair, well
    # above the default but below 1.0, must merge. Together these two
    # fixtures bracket the default, so moving it in either direction
    # fails rather than silently changing how many boxes you report.
    duplicate = [{'box': (0.50, 0.5, 0.2, 0.2), 'class': 4, 'score': 0.9},
                 {'box': (0.52, 0.5, 0.2, 0.2), 'class': 4, 'score': 0.8}]
    assert 0.45 < model.iou(duplicate[0]['box'], duplicate[1]['box']) < 1.0, \
        'test fixture no longer straddles the default threshold'
    assert len(model.nms(duplicate)) == 1, 'default threshold failed to merge duplicates'


# -------------------------------------------------------- letterbox


@test
def letterbox_round_trips():
    for w_in, h_in in [(640, 480), (480, 640), (416, 416), (1920, 1080), (555, 553)]:
        net_w = net_h = 416
        if net_w/w_in < net_h/h_in:
            new_w, new_h = net_w, (h_in*net_w)//w_in
        else:
            new_h, new_w = net_h, (w_in*net_h)//h_in
        truth = (0.5, 0.5, 0.5, 0.5)
        dets = [{'box': ((truth[0]*new_w + (net_w-new_w)//2)/net_w,
                         (truth[1]*new_h + (net_h-new_h)//2)/net_h,
                         truth[2]*new_w/net_w, truth[3]*new_h/net_h),
                 'class': 0, 'score': 1.0}]
        model.correct_boxes(dets, (w_in, h_in), (net_w, net_h))
        # Tolerance is darknet's own half-pixel: it pastes at an integer
        # offset and undoes it at a fractional one. See correct_boxes.
        close(dets[0]['box'], truth, 0.5/new_w + 1e-6, f'letterbox {w_in}x{h_in}')


@test
def preprocess_letterboxes_without_distorting():
    from PIL import Image
    import tempfile
    px = np.zeros((480, 640, 3), dtype=np.uint8)
    px[:, :, 0] = 255
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'red.png')
        Image.fromarray(px).save(path)
        x = model.preprocess(path, 416, 416)
    assert x.shape == (3, 416, 416) and x.dtype == np.float32
    assert x.flags.c_contiguous, 'conv2d gather needs a contiguous array'
    # 640x480 scales to 416x312, leaving 52 border rows top and bottom
    assert np.allclose(x[:, 0, 208], 0.5), 'border is not mid grey'
    assert np.allclose(x[:, 208, 208], [1.0, 0.0, 0.0]), 'centre pixel wrong'
    border_rows = int((x[0, :, 208] == 0.5).sum())
    assert border_rows == 104, f'{border_rows} border rows, expected 104'
    assert 0.0 <= x.min() and x.max() <= 1.0


# ------------------------------------------------------------ end to end


@test
def the_labrador_image_still_detects_a_dog():
    """Guards accuracy, not just shapes: everything else here would pass
    on a model whose weights were subtly mis-parsed."""
    net, spec = need_model()
    path = os.path.join(ROOT, 'calib', 'labrador.jpg')
    if not os.path.exists(path):
        raise Skip('calib/labrador.jpg not present')
    from PIL import Image
    size = Image.open(path).size
    x = model.preprocess(path, int(net['width']), int(net['height']))
    dets = model.nms(model.decode(model.forward(x, spec), spec,
                                  thresh=0.24, shape=size))
    dog = [d for d in dets if d['class'] == 16]        # COCO 16 == dog
    assert dog, f'no dog; got classes {[d["class"] for d in dets]}'
    assert dog[0]['score'] > 0.8, f"dog score fell to {dog[0]['score']:.3f}"
    bx, by, bw, bh = dog[0]['box']
    assert 0.2 < bx < 0.8 and 0.2 < by < 0.8, f'dog centre drifted to {bx:.2f},{by:.2f}'
    assert 0.3 < bw < 0.9 and 0.3 < bh < 1.0, f'dog box size {bw:.2f}x{bh:.2f}'


@test
def darknet_canonical_images_detect_their_published_objects():
    """The multi-object cases. A single-subject photo cannot catch a
    class-index off-by-one or an NMS bug that collapses distinct
    objects, because there is only ever one right answer in frame.

    These are darknet's own test images and these are the objects its
    published yolov2-tiny output finds in them.
    """
    net, spec = need_model()
    from PIL import Image
    expected = {
        'person.jpg': {0: 'person', 16: 'dog', 17: 'horse'},
        'dog.jpg': {1: 'bicycle', 2: 'car', 16: 'dog'},
        'horses.jpg': {17: 'horse'},
        'eagle.jpg': {14: 'bird'},
    }
    for name, want in expected.items():
        path = os.path.join(ROOT, 'calib', name)
        if not os.path.exists(path):
            raise Skip(f'calib/{name} not present')
        size = Image.open(path).size
        x = model.preprocess(path, int(net['width']), int(net['height']))
        dets = model.nms(model.decode(model.forward(x, spec), spec,
                                      thresh=0.24, shape=size))
        found = {d['class'] for d in dets}
        missing = {c: n for c, n in want.items() if c not in found}
        assert not missing, f'{name}: missing {list(missing.values())}, got {sorted(found)}'
        # boxes must be inside the frame and non-degenerate
        for d in dets:
            bx, by, bw, bh = d['box']
            assert bw > 0 and bh > 0, f'{name}: degenerate box {d["box"]}'
            assert -0.1 < bx < 1.1 and -0.1 < by < 1.1, f'{name}: box off frame {d["box"]}'
    # horses.jpg has four; NMS must not collapse them into one
    path = os.path.join(ROOT, 'calib', 'horses.jpg')
    size = Image.open(path).size
    x = model.preprocess(path, 416, 416)
    dets = model.nms(model.decode(model.forward(x, spec), spec, thresh=0.24, shape=size))
    horses = [d for d in dets if d['class'] == 17]
    assert len(horses) >= 3, f'NMS collapsed the herd to {len(horses)}'


# ------------------------------------------------------------------ main


def main():
    passed = failed = skipped = 0
    for fn in TESTS:
        name = fn.__name__.replace('_', ' ')
        try:
            fn()
        except Skip as exc:
            print(f'  SKIP  {name}  ({exc})')
            skipped += 1
        except Exception:
            print(f'  FAIL  {name}')
            print('        ' + traceback.format_exc().replace('\n', '\n        ').rstrip())
            failed += 1
        else:
            print(f'  ok    {name}')
            passed += 1

    print(f'\n{passed} passed, {failed} failed, {skipped} skipped')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
