# YOLOv2-tiny FPGA Accelerator

A fixed-point YOLOv2-tiny inference accelerator for a Zynq-7020, built
alongside a bit-exact float reference model that every hardware output is
checked against.

## Overview

Two halves that have to agree:

**`reference/`** is a float implementation of the whole network in numpy,
from parsing darknet's `.cfg` and `.weights` through to decoded
detections. It is the golden model -- when the accelerator disagrees with
it, the accelerator is wrong. It reproduces darknet's published output:
`person.jpg` gives dog + person + horse at 0.91-0.92.

**`rtl/`** is the accelerator: a weight-streaming convolution datapath
driven from DDR, with region decode and preprocessing left on the ARM
core, where they cost microseconds and would otherwise put `exp` and
`softmax` in the fabric.

`tools/dump_vectors.py` bridges them, writing per-layer `.mem` files so a
mismatch is located at a layer rather than at the output.

Design decisions are recorded alongside the code they govern -- the
module headers in `rtl/` and the docstrings in `reference/` carry the
reasoning and the measurements behind it.

### Status

| | |
|---|---|
| Reference model | complete -- 33 tests, mutation-verified |
| Quantization | complete -- calibration, scales, requantization |
| Golden vectors | complete -- `.mem` files and a manifest |
| RTL | 2 of 4 datapath modules done |

```
make test     reference model tests
make sim      RTL testbenches under iverilog
make simv     same under verilator, stricter
make lint     verilator static checks
```

## Layer table

| #  | type    | in HxWxC     | k | stride | pad          | out HxWxC    | params     | MACs          |
|----|---------|--------------|---|--------|--------------|--------------|------------|---------------|
| 1  | conv    | 416x416x3    | 3 | 1      | 1            | 416x416x16   | 432        | 74,760,192    |
| 2  | maxpool | 416x416x16   | 2 | 2      | -            | 208x208x16   | 0          | -             |
| 3  | conv    | 208x208x16   | 3 | 1      | 1            | 208x208x32   | 4,608      | 199,360,512   |
| 4  | maxpool | 208x208x32   | 2 | 2      | -            | 104x104x32   | 0          | -             |
| 5  | conv    | 104x104x32   | 3 | 1      | 1            | 104x104x64   | 18,432     | 199,360,512   |
| 6  | maxpool | 104x104x64   | 2 | 2      | -            | 52x52x64     | 0          | -             |
| 7  | conv    | 52x52x64     | 3 | 1      | 1            | 52x52x128    | 73,728     | 199,360,512   |
| 8  | maxpool | 52x52x128    | 2 | 2      | -            | 26x26x128    | 0          | -             |
| 9  | conv    | 26x26x128    | 3 | 1      | 1            | 26x26x256    | 294,912    | 199,360,512   |
| 10 | maxpool | 26x26x256    | 2 | 2      | -            | 13x13x256    | 0          | -             |
| 11 | conv    | 13x13x256    | 3 | 1      | 1            | 13x13x512    | 1,179,648  | 199,360,512   |
| 12 | maxpool | 13x13x512    | 2 | 1      | effective 1  | 13x13x512    | 0          | -             |
| 13 | conv    | 13x13x512    | 3 | 1      | 1            | 13x13x1024   | 4,718,592  | 797,442,048   |
| 14 | conv    | 13x13x1024   | 3 | 1      | 1            | 13x13x512    | 4,718,592  | 797,442,048   |
| 15 | conv    | 13x13x512    | 1 | 1      | 1 (effective 0) | 13x13x425 | 217,600    | 36,774,400    |
| 16 | region  | 13x13x425    | - | -      | -            | detections   | -          | -             |
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
build -- that step runs on Windows or Linux.

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
it is `(16, 27) @ (27, 173056)` -- 27 is the systolic depth, 16 the number
of PEs, 173,056 the stream length. The nine-tap gather it builds is the
same structure `rtl/line_buffer.sv` implements in registers.

Three darknet quirks are reproduced rather than corrected, since matching
darknet is the point:

- `pad=1` in the cfg is a flag meaning "pad to same", resolving to
  `size // 2` -- so the 1x1 output conv pads nothing despite saying
  `pad=1`.
- Batch norm normalises by `sqrt(var) + eps`, not the `sqrt(var + eps)`
  every other framework uses.
- Maxpool pads by `size - 1` entirely on the bottom and right with
  `-inf`. Only layer 12 reaches that padding, and it is what holds the
  network at 13x13 instead of 12x12.

### Golden vectors

```
python tools/dump_vectors.py --layer 0
```

Writes `vectors/layer00_{input,weights,bias,expected}.mem` for
`$readmemh`, plus `manifest.json` carrying every scale, multiplier and
shift. The byte order in those files is the memory layout the DMA reads --
activations are HWC, weights channel-major -- and the manifest records it
so the RTL and the testbench cannot drift apart silently.

The fixed-point path drifts 0.007%-0.018% of full scale from the float
reference across all nine conv layers.

## RTL

```
rtl/line_buffer.sv     3x3 window from a pixel stream        done
rtl/mac_array.sv       16 filters in parallel, accumulated   done
rtl/requantize.sv      accumulator -> next layer's input     4 TODOs
rtl/conv_layer.sv      ties the three together               stub

rtl/maxpool.sv         one-sided -inf padding, stride 2      5 TODOs
rtl/weight_loader.sv   DDR -> on-chip, double buffered       stub
rtl/layer_sequencer.sv walks the 15 layers                   stub
rtl/accel_top.sv       AXI wrapper                           stub
```

No `leaky_relu` module: the activation folds into the requantiser's
multiplier as a second constant selected on the accumulator's sign, so
it never becomes hardware of its own.

SystemVerilog, restricted to the subset Icarus also accepts so that both
simulators stay usable. Two constructs are avoided: variable indexing
into a packed multi-dimensional array, and named struct assignment
patterns. Both are Verilator-only.

### The datapath, one layer

```
DDR -> line_buffer -> mac_array -> requantize -> DDR
        holds 2 rows   144 mults    x mult, >> shift,
        emits a 3x3    16 accs      saturate
        window/cycle
```

One window per clock. `C_IN` windows accumulate into one output pixel;
`ceil(N / 16)` passes over the image produce all its channels. The layer
loop, the pass loop and the pixel loop all live outside `mac_array`,
which is why it takes `first_channel` / `last_channel` rather than
counting -- one array serves a 3-channel layer and a 1024-channel one
unchanged.

### Verification

Testbenches are self-checking and print `PASS` or `FAIL`. Each is
validated against a behavioural model, then mutation-tested -- deliberate
bugs injected to confirm the suite notices -- before it is trusted.

Two habits that came out of doing that, both from real failures here:

- **A testbench that can hang is worse than one that fails.** Verilator
  initialises undriven signals to zero, so an unimplemented DUT left
  `in_ready` low and the line buffer's testbench spun forever. Every
  wait is now bounded and every suite carries a watchdog.
- **A run that checked nothing is a failure, not a pass.** An undriven
  `out_valid` reads as `x`, and `if (x)` is false -- so the MAC array's
  suite silently skipped every check and reported PASS. Comparisons are
  written `!== 1'b1`, and a verdict with zero checks now fails.

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
