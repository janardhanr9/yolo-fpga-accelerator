// Multiply-accumulate array for the convolution datapath.
//
// Takes one K x K window from the line buffer and multiplies it against
// N_PE filters' kernels for that channel, accumulating across channels
// until an output pixel is complete. This is the inner loop of
// reference/layers.py's conv2d: the (N, C*K*K) x (C*K*K, H*W) matrix
// multiply, one column of the right operand at a time.
//
// Dataflow. The line buffer emits one window per (output position,
// channel), channel innermost. So for a given output position the array
// sees C_IN windows in a row, and each must be multiplied against that
// channel's slice of every filter's kernel and added to a running total.
// After C_IN windows the accumulators hold N_PE finished sums.
//
//     for output position:
//         for channel:  <- C_IN windows arrive here
//             for pe:   <- N_PE accumulators updated in parallel
//                 acc[pe] += sum over K*K of window * weight
//
// N_PE filters are computed in parallel; layers with more than N_PE
// filters are walked in ceil(N / N_PE) passes over the same input, which
// is why the module takes first_channel and last_channel rather than
// counting for itself.
//
// Sizing. Each PE performs K*K multiplies per cycle, so the array is
// N_PE * K * K multipliers -- 16 * 9 = 144 for the default, against the
// 220 DSP48 slices on a Zynq-7020. A 16x16 signed product fits one
// DSP48E1 exactly as an 8x8 does, since its multiplier is 25x18.
//
// Accumulator width. The widest layer sums 9,216 products of two
// DATA_W-bit values, needing 2*DATA_W + ceil(log2(9216)) = 46 bits at
// DATA_W=16. The DSP48 accumulator is 48 bits, so the natural width
// covers every layer with room to spare. Real data never approaches
// that bound -- layer 13 reaches 33 bits on a calibration image -- but
// the bound is what the hardware must survive, not the average.
//
// The bias is added here rather than downstream because it belongs in
// accumulator units: tools/dump_vectors.py quantizes it at
// in_scale * weight_scale, which is the accumulator's own scale.

module mac_array #(
    parameter int DATA_W = 16,     // sample and weight width, signed
    parameter int ACC_W  = 48,     // accumulator width, signed
    parameter int K      = 3,      // kernel size
    parameter int N_PE   = 16      // filters computed in parallel
)(
    input  logic clk,
    input  logic rst_n,

    // Window stream, straight from line_buffer. All K*K taps of one
    // channel, valid together.
    input  logic                        in_valid,
    input  logic signed [K*K*DATA_W-1:0] in_window,
    output logic                        in_ready,

    // Where this window sits in the accumulation. first_channel clears
    // the accumulators and loads the bias; last_channel marks the cycle
    // whose result is finished and should be emitted.
    input  logic                        first_channel,
    input  logic                        last_channel,

    // Weights for the current channel: N_PE filters x K*K taps. Held
    // steady by the weight loader for the whole cycle.
    input  logic signed [N_PE*K*K*DATA_W-1:0] weights,

    // One bias per PE, in accumulator units. Read only when
    // first_channel is asserted.
    input  logic signed [N_PE*ACC_W-1:0] bias,

    // Finished accumulators, one per PE, valid together.
    output logic                        out_valid,
    output logic signed [N_PE*ACC_W-1:0] out_acc
);

    // Tap accessors. Same flat-vector convention as line_buffer: Icarus
    // cannot index a packed multi-dimensional array with a variable, so
    // the ports are flat and the slicing is spelled out here once.
    `define TAP(i)      in_window[(i)*DATA_W +: DATA_W]
    `define WGT(pe, i)  weights[((pe)*K*K + (i))*DATA_W +: DATA_W]
    `define BIAS(pe)    bias[(pe)*ACC_W +: ACC_W]
    `define ACC(pe)     out_acc[(pe)*ACC_W +: ACC_W]

    // One accumulator per PE, unpacked so the loops below can index it
    // with a variable, bridged to the flat port by a genvar loop.
    logic signed [ACC_W-1:0] acc [N_PE];

    for (genvar i = 0; i < N_PE; i++) begin : acc_bridge
        assign `ACC(i) = acc[i];
    end

    // Nine multiplies per PE, summed to one value. Combinational, so
    // synthesis sees an adder tree it can pipeline later.
    //
    // $signed() on both operands is load-bearing: Verilog treats a
    // part-select of a signed vector as UNSIGNED, so a bare multiply
    // gets roughly half the data wrong and nothing warns.

    logic signed [ACC_W-1:0] prod [N_PE];

    always_comb begin
        for (int pe = 0; pe < N_PE; pe++) begin
            prod[pe] = 0;
            for (int tap = 0; tap < K * K; tap++) begin
                prod[pe] = prod[pe] + $signed(`WGT(pe, tap)) * $signed(`TAP(tap));
            end
        end
    end

    // Bias loads on the first channel rather than being added at the
    // end, which is free -- the accumulator has to be initialised
    // anyway -- and keeps it in accumulator units.
    always_ff @(posedge clk) begin
        if (in_valid && in_ready) begin
            for (int pe = 0; pe < N_PE; pe++) begin
                if (first_channel)  acc[pe] <= $signed(`BIAS(pe)) + $signed(prod[pe]);
                else                acc[pe] <= $signed(acc[pe]) + $signed(prod[pe]);
            end
        end
    end

    // out_acc is valid the cycle after a window arrives with
    // last_channel set, so out_valid is registered alongside the data.
    // in_ready is tied high -- the array never stalls on its own -- but
    // the port stays so the requantiser does not need rewiring when it
    // gains a FIFO.
    always_ff @(posedge clk) begin
        if (!rst_n) out_valid <= 1'b0;
        else        out_valid <= in_valid && last_channel;
    end

    assign in_ready = 1'b1;

    `undef TAP
    `undef WGT
    `undef BIAS
    `undef ACC

endmodule
