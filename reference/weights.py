"""Darknet .weights parsing."""

import struct

import numpy as np


def load(path, layers):
    """Read a darknet .weights file into per-layer arrays.

    `layers` comes from cfg.parse(), already annotated by
    cfg.thread_channels(). Returns (header, weights) where weights maps
    a layer index to a dict of arrays, stored exactly as the file has
    them: no BN folding, no transposes, no scaling.
    """
    with open(path, 'rb') as f:
        header = read_header(f)
        # f is now positioned at the first float; read the rest as float32.
        buf = np.fromfile(f, dtype='<f4')

    pos = 0

    def take(count):
        """Return the next `count` floats and advance the cursor."""
        nonlocal pos
        chunk = buf[pos:pos + count]
        pos += count
        return chunk

    weights = {}
    for i, layer in enumerate(layers):
        if layer['type'] != 'convolutional':
            continue

        n = int(layer['filters'])          # output channels
        c = layer['channels_in']           # input channels (already int)
        k = int(layer['size'])             # kernel size
        bn = int(layer.get('batch_normalize', '0'))

        # Record order is fixed: biases, then the batch-norm triple when
        # present, then the kernels. Biases are beta on batch-normed layers.
        entry = {}
        entry['biases'] = take(n)
        if bn:
            entry['scales'] = take(n)
            entry['rolling_mean'] = take(n)
            entry['rolling_var'] = take(n)
        entry['weights'] = take(n * c * k * k).reshape(n, c, k, k)

        weights[i] = entry

    # Landing exactly on EOF proves every count above was right; a wrong
    # size anywhere shifts the cursor and changes where this ends up.
    assert pos == len(buf), (
        f'consumed {pos:,} of {len(buf):,} floats '
        f'(off by {len(buf) - pos:,})'
    )

    return header, weights


def read_header(f):
    """Read the darknet weights header, leaving `f` at the first float."""
    major, minor, revision = struct.unpack('<3i', f.read(12))
    # The images-seen counter widened to int64 in format version 2.
    if major * 10 + minor >= 2:
        seen, = struct.unpack('<q', f.read(8))
    else:
        seen, = struct.unpack('<i', f.read(4))
    return {
        'major': major,
        'minor': minor,
        'revision': revision,
        'seen': seen,
    }
