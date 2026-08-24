"""Reference YOLOv2-tiny model assembly."""

import numpy as np
from PIL import Image

import cfg
import layers
import quant
import weights


def build(cfg_path, weights_path):
    """Parse the cfg and weights into everything forward() needs.

    Returns (net, spec) where net is the [net] section and spec is an
    ordered list of per-layer dicts. Each entry carries its type plus
    the already-resolved numbers for that layer -- kernel size, stride,
    pad in pixels, activation, and for convs the folded weights and
    biases. Nothing in spec refers back to the cfg strings.

    Folding happens here, once, rather than inside forward(): the fold
    is a property of the weights, not of any particular input, and
    doing it per-call would redo the same arithmetic on every frame.
    """
    # `layer_list` and `params`, not `layers` and `weights` -- those
    # names belong to the two modules this function calls.
    net, layer_list = cfg.parse(cfg_path)
    cfg.thread_channels(net, layer_list)
    _, params = weights.load(weights_path, layer_list)

    spec = []
    for i, layer in enumerate(layer_list):
        kind = layer['type']

        if kind == 'convolutional':
            # params is keyed by cfg position, not by which conv this
            # is, so the enumerate index is the right lookup.
            folded_weights, folded_biases = quant.fold(params[i])
            spec.append({
                'type': kind,
                'index': i,
                'size': int(layer['size']),
                'stride': int(layer['stride']),
                'pad': resolve_pad(layer),
                'activation': layer['activation'],
                'weights': folded_weights,
                'biases': folded_biases,
            })

        elif kind == 'maxpool':
            spec.append({
                'type': kind,
                'index': i,
                'size': int(layer['size']),
                'stride': int(layer['stride']),
            })

        elif kind == 'region':
            # Anchors are stored as flat w,h pairs in grid-cell units,
            # not pixels -- decode() divides by the grid size, never by
            # the input size.
            flat = [float(v) for v in layer['anchors'].split(',')]
            num = int(layer['num'])
            spec.append({
                'type': kind,
                'index': i,
                'anchors': np.array(flat, dtype=np.float32).reshape(num, 2),
                'num': num,
                'classes': int(layer['classes']),
                'coords': int(layer['coords']),
                'softmax': bool(int(layer.get('softmax', '0'))),
            })

        else:
            raise ValueError(f'layer {i}: unsupported type {kind!r}')

    return net, spec


def resolve_pad(layer):
    """Return the padding for `layer` as a pixel count.

    Darknet's cfg `pad=1` is a flag meaning "pad to same", not one
    pixel of padding. It resolves to size // 2, which is why the 1x1
    output conv pads nothing at all despite its section also saying
    pad=1. A layer may instead give an explicit `padding=N`, and a
    layer with neither key pads zero.
    """
    if 'padding' in layer:
        return int(layer['padding'])
    if int(layer.get('pad', '0')):
        return int(layer['size']) // 2
    return 0


def preprocess(path, width, height, border=0.5):
    """Load an image and letterbox it to (3, height, width) float32.

    Letterboxing scales the image to fit inside the target box while
    preserving aspect ratio, then centres it on a neutral background.
    A plain resize would stretch the image and shift every box the
    network predicts.

    This function is the one place the reference model can silently
    disagree with darknet without any shape mismatch to catch it:
    darknet fills the border with 0.5 (in 0-1 units, i.e. mid grey) and
    resizes with its own bilinear implementation, which does not match
    PIL's pixel-for-pixel. Small differences here move the low-order
    bits of every activation downstream, so when the accelerator's
    output is compared against this model, feed both the same
    already-preprocessed tensor rather than the same JPEG.
    """
    img = Image.open(path).convert('RGB')
    w_in, h_in = img.size

    # Darknet picks the axis that saturates first and scales the other
    # by integer division. Written as a branch rather than a single
    # min() ratio so the truncation lands where darknet's does.
    if width / w_in < height / h_in:
        new_w, new_h = width, (h_in * width) // w_in
    else:
        new_h, new_w = height, (w_in * height) // h_in

    resized = np.asarray(img.resize((new_w, new_h), Image.BILINEAR),
                         dtype=np.float32) / np.float32(255.0)

    # Build the canvas in HWC to paste into, then transpose once. The
    # border is already in 0-1 units, matching the divided image.
    canvas = np.full((height, width, 3), border, dtype=np.float32)
    dy, dx = (height - new_h) // 2, (width - new_w) // 2
    canvas[dy:dy + new_h, dx:dx + new_w] = resized

    # copy() because conv2d's im2col gather wants a contiguous array;
    # a bare transpose leaves a view with permuted strides.
    return canvas.transpose(2, 0, 1).copy()


def forward(x, spec, collect=None):
    """Run `x` through the layer stack and return the final feature map.

    x is (3, h, w) float32 from preprocess(). The return value is the
    raw (425, 13, 13) output of the last conv -- no region decode, no
    boxes. Decoding anchors, sigmoids and class scores into detections
    is a separate step and does not belong in the layer walk.

    If `collect` is a dict it is filled with each layer's output tensor
    keyed by layer index. That is how per-layer vectors get dumped for
    the hardware to compare against, and it is the fastest way to find
    the first layer where an accelerator diverges.
    """
    for entry in spec:
        kind = entry['type']

        if kind == 'region':
            # The stack ends here; the region layer is host-side work.
            break

        elif kind == 'convolutional':
            x = layers.conv2d(x, entry['weights'], entry['biases'],
                              entry['stride'], entry['pad'])
            activation = entry['activation']
            if activation == 'leaky':
                x = layers.leaky(x)
            elif activation != 'linear':
                # 'linear' is the output conv and means do nothing;
                # anything else would silently skip a nonlinearity.
                raise ValueError(f"layer {entry['index']}: unsupported "
                                 f'activation {activation!r}')

        elif kind == 'maxpool':
            # layers.maxpool already applies darknet's one-sided
            # padding, so there is no pad to pass through.
            x = layers.maxpool(x, entry['size'], entry['stride'])

        else:
            raise ValueError(f"layer {entry['index']}: unsupported "
                             f'type {kind!r}')

        if collect is not None:
            collect[entry['index']] = x

    return x


def _sigmoid(x):
    """Logistic, written to survive large-magnitude inputs.

    Darknet writes 1 / (1 + exp(-x)), which overflows to inf for very
    negative x before collapsing back to the right answer. The tanh
    identity is the same function without the intermediate inf.
    """
    return np.float32(0.5) * (np.tanh(np.float32(0.5) * x) + np.float32(1.0))


def _softmax(x, axis):
    """Softmax along `axis`, max-subtracted for stability.

    Darknet's softmax_cpu subtracts the row max too, so this is not a
    divergence from it.
    """
    shifted = x - x.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=axis, keepdims=True)


def decode(out, spec, thresh=0.5, shape=None):
    """Turn the raw (425, 13, 13) feature map into detections.

    Returns a list of dicts, each with 'box' as (x, y, w, h) in
    centre-form, 'class' as an index into the 80 COCO classes, and
    'score'. No NMS yet -- run nms() on the result.

    The 425 channels are 5 anchors x 85 entries, laid out anchor-major:
    channel n*85 + e, where e is 0-3 for the box coords, 4 for
    objectness, and 5-84 for the class scores. That grouping is what
    lets the whole thing reshape to (5, 85, 13, 13) instead of needing
    a gather.

    If `shape` is the original image's (width, height), boxes are
    mapped back out of letterbox space into fractions of that image.
    Without it they stay in fractions of the 416x416 letterboxed input,
    which is not what you want to draw.
    """
    num, classes, coords = spec_region(spec, 'num', 'classes', 'coords')
    anchors = spec_region(spec, 'anchors')[0]
    _, h, w = out.shape
    entries = coords + 1 + classes

    pred = out.reshape(num, entries, h, w)

    # Cell offsets: x/y are predicted as a fraction of one cell, so the
    # cell index has to be added back before normalising by the grid.
    gx = np.arange(w, dtype=np.float32).reshape(1, 1, w)
    gy = np.arange(h, dtype=np.float32).reshape(1, h, 1)

    bx = (_sigmoid(pred[:, 0]) + gx) / np.float32(w)
    by = (_sigmoid(pred[:, 1]) + gy) / np.float32(h)

    # w/h are log-space multipliers on the anchor. The anchors are in
    # grid-cell units, so dividing by the grid size -- not the input
    # size -- lands the box in 0-1 image fractions.
    aw = anchors[:, 0].reshape(num, 1, 1)
    ah = anchors[:, 1].reshape(num, 1, 1)
    bw = np.exp(pred[:, 2]) * aw / np.float32(w)
    bh = np.exp(pred[:, 3]) * ah / np.float32(h)

    obj = _sigmoid(pred[:, 4])
    scores = _softmax(pred[:, 5:], axis=1) if spec_region(spec, 'softmax')[0] \
        else _sigmoid(pred[:, 5:])

    # Darknet gates on objectness * class probability, not on either
    # alone: a confident class inside an empty cell must not survive.
    conf = obj[:, None] * scores

    dets = []
    for n, k, i, j in zip(*np.nonzero(conf > thresh)):
        dets.append({
            'box': (float(bx[n, i, j]), float(by[n, i, j]),
                    float(bw[n, i, j]), float(bh[n, i, j])),
            'class': int(k),
            'score': float(conf[n, k, i, j]),
        })

    if shape is not None:
        correct_boxes(dets, shape, (w * 32, h * 32))

    return dets


def spec_region(spec, *keys):
    """Pull fields off the [region] entry in `spec`."""
    for entry in spec:
        if entry['type'] == 'region':
            return [entry[k] for k in keys]
    raise ValueError('spec has no region layer')


def correct_boxes(dets, shape, net_shape):
    """Map boxes out of letterbox space, in place.

    preprocess() centred a scaled image inside a larger canvas, so
    every predicted coordinate is a fraction of the canvas, not of the
    photo. Undoing it is the same scale-and-offset preprocess() applied,
    run backwards.

    Not quite exactly backwards, though, and deliberately so: darknet
    pastes the image at an integer offset (embed_image truncates
    (w - new_w) / 2) but undoes it at a fractional one
    (correct_region_boxes divides by 2.0), leaving half a pixel on the
    table. That is 0.12% of a 416-wide box and invisible in practice,
    but it grows as the letterboxed image gets narrower -- 1.2% on a
    100x999 input, whose image occupies only 41 columns. Reproduced
    here rather than corrected, because matching darknet is this
    model's job.
    """
    w_in, h_in = shape
    net_w, net_h = net_shape

    if net_w / w_in < net_h / h_in:
        new_w, new_h = net_w, (h_in * net_w) // w_in
    else:
        new_h, new_w = net_h, (w_in * net_h) // h_in

    for det in dets:
        x, y, bw, bh = det['box']
        # Subtract the border's share of the canvas, then rescale by
        # how much of the canvas the image actually occupied.
        x = (x - (net_w - new_w) / 2 / net_w) / (new_w / net_w)
        y = (y - (net_h - new_h) / 2 / net_h) / (new_h / net_h)
        det['box'] = (x, y, bw * net_w / new_w, bh * net_h / new_h)


def iou(a, b):
    """Intersection over union of two centre-form (x, y, w, h) boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    # Overlap on each axis independently: the gap between the inner
    # edges, floored at zero when they do not reach each other.
    dx = min(ax + aw / 2, bx + bw / 2) - max(ax - aw / 2, bx - bw / 2)
    dy = min(ay + ah / 2, by + bh / 2) - max(ay - ah / 2, by - bh / 2)
    if dx <= 0 or dy <= 0:
        return 0.0

    overlap = dx * dy
    return overlap / (aw * ah + bw * bh - overlap)


def nms(dets, thresh=0.45):
    """Greedy non-maximum suppression, per class.

    Darknet suppresses within a class and never across: an overlapping
    dog and person are both real, so the class index has to gate the
    comparison.
    """
    kept = []
    for cls in {det['class'] for det in dets}:
        # Highest score first, so whatever survives is always the best
        # box of its cluster.
        ranked = sorted((d for d in dets if d['class'] == cls),
                        key=lambda d: -d['score'])
        for det in ranked:
            if all(iou(det['box'], k['box']) <= thresh for k in kept
                   if k['class'] == cls):
                kept.append(det)
    return sorted(kept, key=lambda d: -d['score'])
