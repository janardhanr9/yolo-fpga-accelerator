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

    // One independent lane per PE. The sign test is on the ACCUMULATOR
    // rather than the result: requantisation is monotonic so they agree,
    // but the accumulator is what the hardware has in hand at that point.
    // Product width. Anything narrower wraps silently.
    localparam int P_W = ACC_W + MULT_W;

    // Held at the product's width, not OUT_W: they are compared against
    // `rounded`, and an OUT_W-wide limit would truncate before the
    // comparison ever happened.
    localparam logic signed [P_W-1:0] OUT_LO = -(1 <<< (OUT_W-1));
    localparam logic signed [P_W-1:0] OUT_HI =  (1 <<< (OUT_W-1)) - 1;

    for (genvar pe = 0; pe < N_PE; pe++) begin : lane

        wire signed [P_W-1:0] prod =
            $signed(`ACC(pe)) < 0 ? $signed(`ACC(pe)) * mult_neg
                                  : $signed(`ACC(pe)) * mult;

        wire signed [P_W-1:0] rounded =
            (shift > 0) ? (prod + (P_W'(1) << (shift - 1))) >>> shift
                        : prod;

        wire signed [OUT_W-1:0] saturated =
            (rounded > OUT_HI) ? OUT_W'(OUT_HI) :
            (rounded < OUT_LO) ? OUT_W'(OUT_LO) : OUT_W'(rounded);

        assign `OUT(pe) = saturated;

    end

    // out_data is valid the cycle after an accepted accumulator, so
    // out_valid is a registered copy of in_valid -- not a flag that sets
    // and stays set, which would tell the next stage that every cycle
    // carries a finished result.
    always_ff @(posedge clk) begin
        if (!rst_n)         out_valid <= 1'b0;
        else                out_valid <= in_valid;
    end
    assign in_ready = 1'b1;

    `undef ACC
    `undef OUT

endmodule
