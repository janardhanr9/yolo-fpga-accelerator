# Architecture decisions

Every decision here came out of a measurement rather than a preference.
Each entry records what was chosen, why, and what it was chosen over, so
that none of them has to be relitigated at 2am six weeks from now.

Numbers are from the real yolov2-tiny weights and darknet's own test
images, reproducible with `make test` and `tools/dump_vectors.py`.

---

## Numerics

### Batch norm is folded into the weights, offline

At inference, batch norm is an affine function of the convolution output
with constants frozen at training time. Affine composed with linear is
another linear map, so it collapses:

```
y = a·(w·x) + b  =  (a·w)·x + b
```

`a` is the same for every pixel in a channel, so folding moves the
multiply from **5,710,848 runtime elements per frame to 2,544 offline
numbers**. The accelerator has no batch-norm unit and never learns that
batch norm existed.

Folding happens in `build()`, once, not per frame -- the fold is a
property of the weights, not of any input.

**Rejected:** a batch-norm stage in the datapath. One multiply and one
add per element, where the add is free (accumulator initialisation) and
the multiply costs a DSP per lane, to compute something already known.

*Note:* darknet writes `sqrt(var) + eps`, not the `sqrt(var + eps)` every
other framework uses. Reproduced exactly; it matters once you chase
bit-exactness.

### Fold before quantizing

Folding rescales each filter's kernels by its own factor, which changes
per-filter dynamic range -- precisely what the fixed-point format is
chosen to fit. Quantizing first would size formats against numbers that
are about to move.

### 16-bit first, 8-bit as an optimization

16-bit reaches full accuracy with naive calibration and costs almost
nothing on the target part: **52 KiB of 630 KiB block RAM, 214 MB/s against
DDR3's ~1 GB/s**, and the same one DSP48 per multiply, since a 16×16
signed product fits the 25×18 multiplier exactly as an 8×8 does.

The decisive argument is not resources but debugging: at 16-bit, a wrong
answer from the board is an RTL bug. At 8-bit it could be either, and
resolving that ambiguity on unfamiliar hardware is what turns a week into
three.

8-bit is worth about 2× throughput via DSP SIMD packing, and stays on the
map for after the board works.

**Measured, 8-bit, against a 41-detection float baseline over eight
images:**

| configuration | recall | spurious |
|---|---|---|
| power-of-two scale, per-tensor weights | 58.5% | 21 |
| power-of-two scale, per-channel weights | 65.9% | 14 |
| arbitrary scale, per-tensor weights | 73.2% | 13 |
| arbitrary scale, per-channel weights | 90.2% | 6 |

### Scales are arbitrary reals, not powers of two

A power-of-two scale dequantizes with a free shift, but can only land on
octave boundaries. A tensor peaking at 16.04 gets a range of 32 and
wastes half its codes -- **29.8% of the code space on average across this
network, 49.9% on layer 8**.

That costs **32 points of recall at 8 bits** (58.5% → 90.2%, with
per-channel weights). Buying it back needs one multiplier per output
lane -- but a 48×16 product does not fit one DSP48E1, whose multiplier is
25×18 signed, so each lane costs two slices: **32 DSPs at 16 lanes, 22%
on top of the array's 144**. It fires once per 9,216 MACs on layer 13.
Take that trade.

Truncating the accumulator to 25 bits before the multiply would halve
that to 16 DSPs, and the low bits are below the output format's
resolution anyway. Worth measuring before spending the slices.

**Rejected:** power-of-two scales. Free in hardware, and the most
expensive decision available.

### Per-channel weights, per-tensor activations

Layer 10's 512 channels need anywhere from −1 to 4 integer bits; a single
channel forces the whole tensor to the widest. Per-channel formats
recover that, but the cost is asymmetric:

- **Per-channel weights is nearly free.** Each output channel already
  gets its own bias added at the end of accumulation; a per-channel scale
  rides along in the same stage.
- **Per-channel activations is expensive.** The scale would have to vary
  per *input* channel mid-datapath, breaking the uniform MAC array.

Which is exactly why this is the industry default -- not convention, but
which side of the accumulator the rescale lands on.

### Requantization rounds half-up and saturates

The stage between the accumulator and the next layer's input is
`multiply → round → shift → saturate`. Both knobs are traps, and both
were measured:

| knob | wrong choice | consequence |
|---|---|---|
| rounding | truncate (`>>>` alone) | arithmetic shift floors toward −∞, so every layer biases the same direction; compounds to `+4.5` mean drift by layer 8 and **the network stops detecting anything** |
| overflow | wrap | `+max+1` becomes `−max`, sending a channel's strongest activation to its most negative; **0.683 → 0.454** on the dog, still detected |

The wrap bug is the dangerous one: it only appears once a format is
tightened enough to overflow, which is exactly when you are optimizing
and least expecting a regression, and the output stays plausible enough
to pass a smoke test.

#### Why adding half turns floor into round

`p >>> shift` is `floor(p / 2^shift)`. It steps up when the true value
crosses a **whole** number, so everything between `n.0` and `n.999` lands
on `n` -- always the value below.

Adding half the divisor first moves that step-up point back by half a
division, so it lands on `n` only up to `n.5` and on `n+1` after. Which
is what "nearest" means.

Dividing by 8, where half is 4:

| `p` | true `p/8` | `p >>> 3` | `(p + 4) >>> 3` |
|---|---|---|---|
| 3 | 0.375 | 0 | 0 |
| **4** | **0.500** | **0** | **1** |
| 7 | 0.875 | 0 | 1 |
| 8 | 1.000 | 1 | 1 |
| **12** | **1.500** | **1** | **2** |

The adder costs one gate delay. What it buys is not precision but the
absence of **bias** -- measured over `p = 0..63`:

| | average error | average bias |
|---|---|---|
| floor | 0.438 | **−0.438** |
| round | 0.250 | +0.062 |

The errors are comparable. The difference is that floor is wrong in the
*same direction every time*, so it accumulates: fifteen layers of −0.438
drifts about −6.6 LSB, with every activation in the network shifting
down together. Rounding scatters either side of zero and cancels.

That is the difference between "slightly less precise" and "the
detections disappear".

Rounding is half-up rather than half-even, because one adder plus a shift
is what the hardware can afford. `quantize()` uses numpy's half-even; the
difference appears only on exact halves, and `requantize()` is the
function that models hardware.

### The integer is a count; the scale is the unit

There is no fixed binary point anywhere in the datapath. A stored value
is an integer, and each stage carries a **scale** saying what one LSB is
worth. The same 16-bit pattern means different things at different
points:

| stage | scale | `12345` means |
|---|---|---|
| layer input | `3.052e-05` | 0.377 |
| a weight | `8.711e-04` | 10.754 |
| layer output | `1.685e-03` | 20.801 |

The accumulator's scale is `in_scale × weight_scale`, because every term
in it was one input code times one weight code -- **the units multiply**.

Requantisation is therefore a change of counting unit, not a change of
value: *"I hold 1,234,567 counts of 2.66e-08; how many counts of
1.68e-03 is that?"* The multiply-and-shift is how that division gets done
in integers.

This is what distinguishes the design from a Q-format one, where bit
*position* carries the value. Here position carries nothing -- you need
the scale, and the scale never appears in the RTL at all. It lives in
Python, collapsed into `mult` and `shift` before synthesis.

### The accumulator width is asserted, not assumed

numpy raises a `RuntimeWarning` when a scalar integer multiply overflows
and says **nothing at all** when an array one does. Since `requantize()`
always takes arrays, the width is checked explicitly.

Observed accumulator use is far below the worst case -- layer 13 reaches
33 bits against a 46-bit budget -- because the budget assumes all 9,216
terms at maximum magnitude with the same sign. Size to the budget; treat
the 13-bit gap as a known optimization, not a saving to bank now.

---

## Dataflow and memory

### Weights stream from DDR; activations round-trip through it

Not a choice. **11,229,513 parameters is 21.4 MiB at 16-bit against 630 KiB
of block RAM**; layer 13 alone is larger than the entire BRAM. Layer 0's
output is 5.5 MiB, so activations do not fit either, at any bit width.

The dataflow is therefore `DDR → line buffer → MACs → requantize → DDR`,
once per layer.

Bandwidth is comfortable: **51.3 MB per frame, 103 MB/s at 2 fps and
513 MB/s at 10**, against ~1.2 GB/s on one 64-bit HP AXI port, of which
the part has four.

### Activations are HWC -- channel innermost

```
for row: for col: for channel
```

This is what `line_buffer.sv` consumes. Numpy stores `(C, H, W)` --
channel outermost -- so `flatten_activation()` is a genuine transpose.

**Rejected:** CHW planes. Native to numpy, no transpose, but the line
buffer would gather each window from addresses a whole feature map apart.

### Weights are channel-major -- `[c][n][k][k]`

The line buffer emits one K×K window per (pixel, channel), walking
channels innermost. So at any moment the array holds a single input
channel and needs that channel's kernels for **every** output filter.
Channel-major places exactly that block contiguously: weights stream in
lockstep with windows, and each weight is read once per layer.

**Rejected:** filter-major `[n][c][k][k]`, numpy's own order. Zero
transpose in the dumper, but the whole tensor must be re-read once per
output filter, or held in a reorder buffer in the fabric.

### One line buffer, sized for the worst layer, serves all of them

A K×K window spans K rows, K−1 of which must be held while the newest
arrives: `(K−1) × W × C` samples. Across the whole network the worst case
is layer 13 at **2 × 13 × 1024 = 26 KiB (52 KiB at 16-bit), under six
BRAM36 blocks of 140** -- because image width halves exactly as channel
count doubles, holding the product nearly flat.

### Window history lives in LUT-RAM, not flip-flops

Channel is innermost in the stream, so between one channel's window at
column *c* and its window at *c+1* exactly `C_IN` samples go by. Every
horizontal stage is therefore a delay line of depth `C_IN`.

Held in flip-flops that is `K*(K-1)*C_IN` of them -- **98,304 at layer
13's 1024 channels, 92% of the device**. Written instead as a memory
indexed by `channel`, only one word changes per cycle, so synthesis
infers distributed RAM or an SRL: **the same bits for about 3% of the
LUTs**.

Generalises: when a design does not fit, check whether it is the
*amount* of state or the *resource holding it*. Long delay lines in
flip-flops is one of the easiest ways to burn an FPGA.

**Unverified.** This is inference, not a directive -- only Vivado reports
what it actually built, and Vivado has no macOS build. The first
synthesis run should check the utilization report at layer 13's
parameters. If it comes back as flip-flops, force it with
`(* ram_style = "distributed" *)` on the declaration.

### Windows stream channel-serially

`C_IN` reaches 1024 on layer 13. A channel-parallel window would be
**9,216 wires**.

### Leaky ReLU folds into the requantizer

0.1 is not a power of two, so leaky is either a multiplier or an
approximation. But the requantizer already has a multiplier per lane, so
the negative branch is the same `M` scaled by the slope:

```
mult_pos = round(M * 2**shift)
mult_neg = round(M * 0.1 * 2**shift)
out      = requantize(acc, acc < 0 ? mult_neg : mult_pos, shift)
```

Cost: one mux and one extra stored constant per layer. Error: none -- the
slope lives in a constant, not in a shift approximation. Layer 0's pair
is `16940 / 1694`, exactly a tenth.

**Rejected:** `>>4 + >>5 + >>8`, which gives 0.0977 -- 2.3% low on every
negative activation, and requires the reference model to be bent to match
the hardware rather than the other way round.

### Bias is 32-bit, in accumulator units

The bias adds into the accumulator, so it lives at
`in_scale × weight_scale`, not at the output scale. Quantizing it to the
data width would be wrong by the entire requantization factor. 32 bits is
ample and effectively free -- biases load once per filter.

---

## Partitioning

### Region decode, NMS and letterboxing stay on the host CPU

The last conv's output is 13×13×425 -- **0.2% of the network's MACs**, and
microseconds of scalar work. Accelerating it would add `exp` and
`softmax` to the datapath for no throughput gain.

On a Zynq the host is the ARM core on the same die, so this is not a
partition across a bus -- and it can run the same Python the reference
model uses.

---

## What ships

Python appears in the finished product in two places, and the reference
model is not one of them.

**Build time, once, on a laptop -- ships as data.** `cfg.py` and
`weights.py` parse darknet's files; `quant.py` folds batch norm,
calibrates, and computes every scale; `dump_vectors.py` writes `.mem`
files and `manifest.json`. What reaches the board is those files -- the
`mult = 16940, shift = 30` constants and the quantized weights. The
Python that produced them does not, and never runs again unless the
network is re-quantized.

**Runtime, every frame, on the ARM core -- ships as code.**
`model.preprocess()` letterboxes the image in, `model.decode()` and
`model.nms()` turn the last layer's output into boxes, and `host.py`
drives the DMA between them. On a Zynq the host is a Linux core on the
same die, so this is Python calling into PYNQ.

**Never ships.** `layers.py`'s float convolution, `model.forward()`, and
`tests/run.py` exist so the hardware can be told it disagrees with
something. They are a measuring instrument, not part of the product.

The dividing line is useful when deciding where work belongs: anything
that can be computed once, before the first frame, should be -- which is
why the accelerator has no batch-norm unit, no scale registers, and no
division.

## Platform

### Zynq-7020 as the design target

| | |
|---|---|
| DSP | 220 (~148 used: 132 MAC + 16 requant) |
| BRAM | 140 × 36 Kb = 630 KiB (~14% used at 16-bit) |
| DDR | 512 MB DDR3 |
| Estimate | ~7 fps at 132 MACs/cycle, 150 MHz |

Chosen because universities stock it, the ARM PS runs the host-side work
on-die, and a design that fits its budget ports upward for free -- the
reverse is a rewrite.

**The architecture does not depend on this choice.** Weight streaming and
activation tiling are required on every candidate part, so only the array
width and clock target move. The board question can be settled as late as
21 September and still leave a bought fallback time to ship.

### Mac for simulation, Windows for Vivado

Vivado has no macOS build. Icarus 13 and Verilator 5 cover all RTL
development and verification with second-scale iteration; synthesis,
timing closure and bitstream generation happen on the Windows machine.

### SystemVerilog, restricted to the Icarus-compatible subset

Two constructs are Verilator-only and are avoided:

- variable indexing into a packed multi-dimensional array (`w[i][j]`)
- named struct assignment patterns (`'{r: 4'd3}`)

So module ports carry flat packed vectors with a `WIN` macro for tap
access, and the design holds its window registers in an **unpacked**
array bridged to the port by a genvar loop -- genvars are constant, so
that indexing is legal everywhere.

Losing the fast simulator costs more than the nicer syntax is worth.

---

## Verification

### Every check compares against an independent implementation

Not against a stored expectation. `conv2d` and `maxpool` go against
nested loops; `fold` against the unfolded batch-norm path; `decode`
against a brute-force walk of darknet's own `entry_index` arithmetic;
`requantize` against the same operation in unbounded Python integers;
`iou` against mask overlap on a 1500² grid.

A stored expectation only proves the code still does what it did. An
independent implementation proves it does the right thing.

### Every suite is deliberately broken before it is trusted

A test that has never failed is unverified. Thirteen bugs were injected
into the reference model, thirteen into the quantization module, and
seven into the line buffer testbench; each must produce failures, and the
unmutated tree must pass clean.

This found a real hole: the NMS test passed its threshold explicitly, so
changing the default was invisible to the entire suite. Two fixtures now
bracket it at IoU 0.351 and 0.818.

**Corollary worth remembering:** if a test passes a parameter explicitly,
that parameter's default is untested.

### A testbench that can hang is worse than one that fails

Verilator initialises undriven signals to zero, so an unimplemented DUT
left `in_ready` low and the line buffer testbench spun forever -- the
first command anyone would run hung instead of reporting. It now carries
a watchdog and a bounded handshake wait. A hang gives you no information
about where to look.

### Calibration uses a set, never one image

Across darknet's eight test images, **thirteen of fifteen layers peak
higher than any single image suggests** -- layer 10 by 2.06×, enough to
cross a bit boundary. A range measured too low does not degrade
gracefully: it clips, and the clipped values are exactly the strong
activations the detector keys on.
