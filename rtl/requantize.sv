// Accumulator-to-activation conversion: the stage between the MAC array
// and the next layer's input.
//
// The accumulator holds a value in units of in_scale * weight_scale.
// The next layer expects out_scale. So every accumulator is multiplied
// by M = in_scale * weight_scale / out_scale, a real number well below
// one -- and no FPGA multiplies by 0.0000158. reference/quant.py's
// requant_params() approximates it as mult / 2^shift, which the hardware
// evaluates as one integer multiply and one arithmetic shift.
//
// Leaky ReLU folds in here rather than costing its own multiplier: the
// negative branch is the same M scaled by the slope, so the datapath
// picks between two constants on the accumulator's sign. Exact, because
// 0.1 lives in a constant rather than a shift approximation. On the
// output conv, which is linear, the two constants are equal.
//
// This module must match reference/quant.py's requantize() bit for bit
// -- it is the one place where a mismatch against the golden vectors is
// a real bug in one of them rather than a modelling artefact. Both
// traps it contains are documented there and repeated here because they
// are worth meeting twice.

module requantize #(
    parameter int ACC_W  = 48,     // accumulator width, signed
    parameter int OUT_W  = 16,     // output width, signed
    parameter int MULT_W = 16,     // requantisation multiplier width
    parameter int N_PE   = 16      // lanes, matching mac_array
)(
    input  logic clk,
    input  logic rst_n,

    input  logic                        in_valid,
    input  logic signed [N_PE*ACC_W-1:0] in_acc,
    output logic                        in_ready,

    // Per-layer constants, held steady by the sequencer. mult_neg is
    // mult scaled by the activation slope; for a linear layer the two
    // are equal.
    input  logic signed [MULT_W-1:0]    mult,
    input  logic signed [MULT_W-1:0]    mult_neg,
    input  logic [$clog2(ACC_W+MULT_W)-1:0] shift,

    output logic                        out_valid,
    output logic signed [N_PE*OUT_W-1:0] out_data
);

    `define ACC(pe)  in_acc[(pe)*ACC_W +: ACC_W]
    `define OUT(pe)  out_data[(pe)*OUT_W +: OUT_W]

    // TODO 1: the product. Each lane multiplies its accumulator by mult
    // or mult_neg, chosen on the accumulator's sign.
    //
    // Width matters here and nothing will warn you. ACC_W + MULT_W bits
    // is 64 for the defaults, and a product held in anything narrower
    // wraps silently -- the same failure reference/quant.py asserts
    // against, because numpy is equally quiet about it.
    //
    // Note the sign test is on the ACCUMULATOR, not on the result.
    // Requantisation is monotonic so they agree, but the accumulator is
    // what the hardware has in hand at that point.

    // TODO 2: rounding. An arithmetic shift right floors toward negative
    // infinity, so shifting alone biases every value the same direction
    // -- and fifteen layers of that erases the detections entirely.
    // Adding half an LSB before the shift is what turns the floor into
    // a round, and it is one adder.
    //
    // Half an LSB is 1 << (shift-1). Guard shift == 0, where that
    // expression is a negative shift amount.

    // TODO 3: saturation. A value that will not fit must pin at the
    // limit, never wrap. A wrapped value sends a channel's strongest
    // activation to its most negative and stays plausible enough to
    // survive a smoke test -- measured, it costs about a third of the
    // detection confidence while still finding the object.
    //
    // The limits are asymmetric: -2^(OUT_W-1) and +2^(OUT_W-1)-1. Using
    // the same bound for both silently discards a code.

    // TODO 4: the output handshake. out_data is valid the cycle after an
    // accepted accumulator. Register out_valid alongside the data or it
    // will lead by a cycle. in_ready can be tied high for now.

    `undef ACC
    `undef OUT

endmodule
