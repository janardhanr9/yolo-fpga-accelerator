"""Darknet .cfg parsing."""


def parse(path):
    """Parse a darknet .cfg into (net, layers).

    net is the [net] section as a dict. layers is an ordered list of
    dicts, one per layer section, each carrying a 'type' key. Values are
    left as strings; callers convert as needed.
    """
    net = {}
    layers = []
    current = None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('['):
                section = line[1:-1]
                if section == 'net':
                    current = net
                else:
                    current = {'type': section}
                    layers.append(current)
                continue
            key, value = line.split('=', 1)
            current[key.strip()] = value.strip()

    return net, layers


def thread_channels(net, layers):
    """Annotate each convolutional layer with its input channel count.

    Adds 'channels_in' as an int. Non-conv layers pass channels through
    unchanged.
    """
    channels = int(net['channels'])
    for layer in layers:
        if layer['type'] == 'convolutional':
            layer['channels_in'] = channels
            channels = int(layer['filters'])
