# YOLOv2-tiny FPGA Accelerator

A fixed-point YOLOv2-tiny inference accelerator for a Zynq-7020, built
alongside a bit-exact float reference model that every hardware output is
checked against.

## Overview

Two halves that have to agree:

**`reference/`** is a float implementation of the whole network in numpy,
from parsing darknet's `.cfg` and `.weights` through to decoded
detections. It is the golden model — when the accelerator disagrees with
it, the accelerator is wrong. It reproduces darknet's published output:
`person.jpg` gives dog + person + horse at 0.91–0.92.

**`rtl/`** is the accelerator: a weight-streaming convolution datapath
driven from DDR, with region decode and preprocessing left on the ARM
core, where they cost microseconds and would otherwise put `exp` and
`softmax` in the fabric.

`tools/dump_vectors.py` bridges them, writing per-layer `.mem` files so a
mismatch is located at a layer rather than at the output.

Design decisions and the measurements behind them are in
[DECISIONS.md](DECISIONS.md).

### Status

| | |
|---|---|
| Reference model | complete — 33 tests, mutation-verified |
| Quantization | complete — calibration, scales, requantization |
| Golden vectors | complete — `.mem` files and a manifest |
| RTL | line buffer in progress |

```
make test     reference model tests
make sim      RTL testbenches under iverilog
make simv     same under verilator, stricter
make lint     verilator static checks
```

## Layer table

| #  | type    | in H×W×C     | k | stride | pad          | out H×W×C    | params     | MACs          |
|----|---------|--------------|---|--------|--------------|--------------|------------|---------------|
| 1  | conv    | 416×416×3    | 3 | 1      | 1            | 416×416×16   | 432        | 74,760,192    |
| 2  | maxpool | 416×416×16   | 2 | 2      | –            | 208×208×16   | 0          | –             |
| 3  | conv    | 208×208×16   | 3 | 1      | 1            | 208×208×32   | 4,608      | 199,360,512   |
| 4  | maxpool | 208×208×32   | 2 | 2      | –            | 104×104×32   | 0          | –             |
| 5  | conv    | 104×104×32   | 3 | 1      | 1            | 104×104×64   | 18,432     | 199,360,512   |
| 6  | maxpool | 104×104×64   | 2 | 2      | –            | 52×52×64     | 0          | –             |
| 7  | conv    | 52×52×64     | 3 | 1      | 1            | 52×52×128    | 73,728     | 199,360,512   |
| 8  | maxpool | 52×52×128    | 2 | 2      | –            | 26×26×128    | 0          | –             |
| 9  | conv    | 26×26×128    | 3 | 1      | 1            | 26×26×256    | 294,912    | 199,360,512   |
| 10 | maxpool | 26×26×256    | 2 | 2      | –            | 13×13×256    | 0          | –             |
| 11 | conv    | 13×13×256    | 3 | 1      | 1            | 13×13×512    | 1,179,648  | 199,360,512   |
| 12 | maxpool | 13×13×512    | 2 | 1      | effective 1  | 13×13×512    | 0          | –             |
| 13 | conv    | 13×13×512    | 3 | 1      | 1            | 13×13×1024   | 4,718,592  | 797,442,048   |
| 14 | conv    | 13×13×1024   | 3 | 1      | 1            | 13×13×512    | 4,718,592  | 797,442,048   |
| 15 | conv    | 13×13×512    | 1 | 1      | 1 (effective 0) | 13×13×425 | 217,600    | 36,774,400    |
| 16 | region  | 13×13×425    | – | –      | –            | detections   | –          | –             |
|    | **total** |            |   |        |              |              | **11,226,544** | **2,703,221,248** |

Params are convolution weights only; batch-norm scales/means/variances and
layer 15's biases are counted separately.

## Build

Python needs only numpy and pillow:

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

RTL simulation needs Icarus Verilog and Verilator (`brew install
icarus-verilog verilator`). Synthesis needs Vivado, which has no macOS
build — that step runs on Windows or Linux.

Weights (44,948,600 bytes) are not tracked in git; fetch into `models/`:

```
curl -o models/yolov2-tiny.weights https://data.pjreddie.com/files/yolov2-tiny.weights
curl -o models/yolov2-tiny.cfg https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2-tiny.cfg
```

## Reference model

```
reference/cfg.py       parse the .cfg; thread input channel counts through it
reference/weights.py   read the .weights into per-layer arrays
reference/layers.py    conv2d (im2col + GEMM), maxpool, leaky ReLU
reference/quant.py     batch-norm folding, calibration, fixed-point conversion
reference/model.py     assemble, run, and decode to detections
```

`conv2d` uses im2col rather than nested loops, and not only for speed:
the resulting matrix multiply is the shape the PE array runs. For layer 1
it is `(16, 27) @ (27, 173056)` — 27 is the systolic depth, 16 the number
of PEs, 173,056 the stream length. The nine-tap gather it builds is the
same structure `rtl/line_buffer.sv` implements in registers.

Three darknet quirks are reproduced rather than corrected, since matching
darknet is the point:

- `pad=1` in the cfg is a flag meaning "pad to same", resolving to
  `size // 2` — so the 1×1 output conv pads nothing despite saying
  `pad=1`.
- Batch norm normalises by `sqrt(var) + eps`, not the `sqrt(var + eps)`
  every other framework uses.
- Maxpool pads by `size - 1` entirely on the bottom and right with
  `-inf`. Only layer 12 reaches that padding, and it is what holds the
  network at 13×13 instead of 12×12.

### Golden vectors

```
python tools/dump_vectors.py --layer 0
```

Writes `vectors/layer00_{input,weights,bias,expected}.mem` for
`$readmemh`, plus `manifest.json` carrying every scale, multiplier and
shift. The byte order in those files is the memory layout the DMA reads —
activations are HWC, weights channel-major — and the manifest records it
so the RTL and the testbench cannot drift apart silently.

The fixed-point path drifts 0.007%–0.018% of full scale from the float
reference across all nine conv layers.

## RTL

```
rtl/line_buffer.sv     sliding-window generator      in progress
tb/tb_line_buffer.sv   self-checking testbench       complete
```

SystemVerilog, restricted to the subset Icarus also accepts so that both
simulators stay usable — see [DECISIONS.md](DECISIONS.md#platform).

Testbenches are self-checking and print `PASS` or `FAIL`. Each is
validated against a behavioural model and then mutation-tested before it
is trusted.

## Results

Reference model, on darknet's own test images, matching its published
output:

| image | detections |
|---|---|
| `person.jpg` | dog 0.92, person 0.91, horse 0.91 |
| `dog.jpg` | dog 0.71, car 0.70, bicycle 0.68 |
| `horses.jpg` | horse 0.81, 0.78, 0.66, 0.26 |
| `eagle.jpg` | bird 0.81, bird 0.64 |
| `kite.jpg` | kite 0.92, person 0.81, + 18 more |

Quantization, 8-bit, against a 41-detection float baseline over eight
images:

| configuration | recall | spurious |
|---|---|---|
| power-of-two scale, per-tensor weights | 58.5% | 21 |
| arbitrary scale, per-channel weights | 90.2% | 6 |
| 16-bit, any configuration | 100% | 0 |

Hardware results pending.
