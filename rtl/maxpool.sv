// Max-pooling over a POOL x POOL window.
//
// Six of yolov2-tiny's fifteen layers are pools. Five are stride 2 and
// halve the feature map; layer 11 is size 2 stride 1 and does not, which
// is what holds the network at 13x13 instead of 12x12.
//
// Darknet's padding is NOT the convolution's. It pads by POOL-1,
// entirely on the bottom and right, and fills with -inf so a padded cell
// can never win a max. Zero fill would be wrong: post-leaky activations
// are frequently negative, and a zero would beat them. In fixed point
// the -inf is the most negative representable value.
//
// Only the stride-1 pool ever reaches that padding. The stride-2 ones
// consume it exactly, which is why the bug would hide until layer 11.
//
// Structurally this is the line buffer again -- a sliding window over a
// streamed image -- with max instead of multiply-accumulate, and with
// channels riding along untouched because pooling never mixes them.
// Whether to reuse line_buffer or write a smaller dedicated window is a
// real decision: pooling needs POOL-1 rows rather than K-1, has no
// weights, and for stride 2 emits only one window in four.

module maxpool #(
    parameter int DATA_W = 16,
    parameter int IMG_W  = 416,
    parameter int IMG_H  = 416,
    parameter int C_IN   = 16,
    parameter int POOL   = 2,
    parameter int STRIDE = 2
)(
    input  logic clk,
    input  logic rst_n,

    input  logic                     in_valid,
    input  logic signed [DATA_W-1:0] in_data,
    output logic                     in_ready,

    output logic                     out_valid,
    output logic signed [DATA_W-1:0] out_data,
    output logic [$clog2(C_IN > 1 ? C_IN : 2)-1:0] out_ch,
    output logic                     out_last
);

    // Darknet's output formula. Note `pad` appears ONCE, unlike the
    // convolution's, because it is all on one side.
    localparam int PAD   = POOL - 1;
    localparam int W_OUT = (IMG_W + PAD - POOL) / STRIDE + 1;
    localparam int H_OUT = (IMG_H + PAD - POOL) / STRIDE + 1;

    // TODO 1: the row buffer. POOL-1 rows of IMG_W * C_IN, the same
    // shape and the same slot-wrapping address as line_buffer.

    // TODO 2: input counters and the stride. Unlike the convolution,
    // output positions are not one-per-input: at stride 2 only every
    // other row and column produces one. Decide whether to count input
    // positions and gate the output, or count output positions directly.

    // TODO 3: the reduction. Max over the POOL x POOL window, per
    // channel. Channels never mix, so this is the same per-channel
    // delay-line structure the line buffer needed -- for the same
    // reason, and with the same flip-flop cost if you get it wrong.

    // TODO 4: padding with the most negative value, bottom and right
    // only. The stride-2 layers never reach it; layer 11 does, and it is
    // the only place a zero-fill bug would ever show.

    // TODO 5: out_valid, out_ch, out_last, and the drain. Same shape as
    // line_buffer's, but the geometry differs -- one-sided padding and a
    // stride mean the warm-up and drain counts are not the convolution's.

endmodule
