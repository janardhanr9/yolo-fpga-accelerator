// One convolution layer: line_buffer -> mac_array -> requantize.
//
// Takes a pixel stream and emits finished activations for N_PE output
// channels. Everything the three submodules need in the way of control
// comes from one place -- the line buffer already tracks which channel
// each window belongs to, so first_channel and last_channel fall out of
// its out_ch rather than needing counters of their own.
//
// Scope. This is one PASS over the image, producing N_PE output
// channels. A layer with more filters than N_PE runs this repeatedly,
// which is the sequencer's job, not this module's. Weights arrive on a
// port rather than being fetched: weight_loader comes later, and until
// it exists the testbench feeds them from a .mem file.
//
// Latency. line_buffer holds its window for one cycle, mac_array
// registers its accumulator, requantize registers its output. So an
// output lags the sample that completed it by three cycles, and
// out_valid has to be pipelined to match rather than recomputed.

module conv_layer #(
    parameter int DATA_W = 16,     // activation width, signed
    parameter int ACC_W  = 48,     // accumulator width
    parameter int MULT_W = 16,     // requantisation multiplier width
    parameter int IMG_W  = 416,
    parameter int IMG_H  = 416,
    parameter int C_IN   = 3,
    parameter int K      = 3,
    parameter int PAD    = 1,
    parameter int N_PE   = 16      // output channels produced per pass
)(
    input  logic clk,
    input  logic rst_n,

    // Input pixel stream: raster order, channel innermost.
    input  logic                     in_valid,
    input  logic signed [DATA_W-1:0] in_data,
    output logic                     in_ready,

    // Per-cycle weights: N_PE filters x K*K taps, for the channel the
    // line buffer is currently emitting. Held steady for the cycle.
    input  logic signed [N_PE*K*K*DATA_W-1:0] weights,

    // Per-pass constants, held steady for the whole pass.
    input  logic signed [N_PE*ACC_W-1:0] bias,
    input  logic signed [MULT_W-1:0]     mult,
    input  logic signed [MULT_W-1:0]     mult_neg,
    input  logic [$clog2(ACC_W+MULT_W)-1:0] shift,

    // N_PE finished activations, one per output channel.
    output logic                        out_valid,
    output logic signed [N_PE*DATA_W-1:0] out_data,
    output logic                        out_last
);

    // TODO 1: instantiate line_buffer. Its parameters are this module's,
    // and its window output feeds mac_array directly. Give it named port
    // connections rather than positional ones -- there are eight, and a
    // silent transposition is a very unpleasant afternoon.

    // TODO 2: derive first_channel and last_channel from the line
    // buffer's out_ch. This is the whole trick of the module: out_ch
    // says which channel the current window belongs to, so the first is
    // channel 0 and the last is C_IN-1. No counters needed.
    //
    // Both must be qualified by the line buffer's out_valid, or they
    // assert on cycles where there is no window and the accumulator
    // clears at the wrong time.

    // TODO 3: instantiate mac_array, fed by the window and the weights
    // port. Its ACC_W and K must match; its N_PE is this module's.

    // TODO 4: instantiate requantize, fed by the accumulators. Note the
    // output width: requantize's OUT_W is this module's DATA_W, because
    // its output IS the next layer's input.

    // TODO 5: out_last. The line buffer knows when the frame ends, but
    // its out_last fires three cycles before the corresponding
    // activation emerges. Pipeline it to match, rather than trying to
    // recompute it downstream.

endmodule
