`timescale 1ns / 1ps

// Testbench for requantize.
//
// This module has to match reference/quant.py's requantize() exactly --
// it is the one place where a mismatch against the golden vectors is a
// real bug rather than a modelling artefact. So the reference here is
// that function, transcribed.
//
//   make sim        iverilog
//   make simv       verilator

module tb_requantize;

    // Small on purpose: 4 lanes and 8-bit output keep failure messages
    // readable and make the saturation limits easy to reason about
    // (-128 .. 127 rather than -32768 .. 32767).
    localparam int ACC_W  = 48;
    localparam int OUT_W  = 8;
    localparam int MULT_W = 16;
    localparam int N_PE   = 4;

    localparam int SH_W = $clog2(ACC_W + MULT_W);

    localparam longint OUT_HI =  (1 << (OUT_W-1)) - 1;   //  127
    localparam longint OUT_LO = -(1 << (OUT_W-1));       // -128

    logic clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    logic                         in_valid = 0;
    logic signed [N_PE*ACC_W-1:0] in_acc   = '0;
    logic                         in_ready;
    logic signed [MULT_W-1:0]     mult     = '0;
    logic signed [MULT_W-1:0]     mult_neg = '0;
    logic [SH_W-1:0]              shift    = '0;
    logic                         out_valid;
    logic signed [N_PE*OUT_W-1:0] out_data;

    requantize #(.ACC_W(ACC_W), .OUT_W(OUT_W),
                 .MULT_W(MULT_W), .N_PE(N_PE)) dut (
        .clk(clk), .rst_n(rst_n),
        .in_valid(in_valid), .in_acc(in_acc), .in_ready(in_ready),
        .mult(mult), .mult_neg(mult_neg), .shift(shift),
        .out_valid(out_valid), .out_data(out_data)
    );

    int errors = 0;
    int checks = 0;

    // A testbench that can hang reports nothing. An unimplemented DUT
    // leaves out_valid at x, and comparisons below are written !== 1'b1
    // rather than ! for the same reason: `if (x)` is false, so a bare
    // test would silently skip every check and still say PASS.
    initial begin
        #20ms;
        $display("");
        $display("FAIL  timed out after %0d checks -- out_valid never fired?", checks);
        $finish;
    end

    // -----------------------------------------------------------------
    // TODO 1: the reference.
    //
    // Transcribe reference/quant.py's requantize(), for ONE lane, in
    // unbounded longint so a too-narrow datapath in the DUT shows up as
    // a mismatch rather than as matching truncation.
    //
    //   function automatic longint ref_requant(longint acc);
    //
    // Four steps, in this order:
    //   1. pick mult or mult_neg on the sign of acc  (not of the result)
    //   2. multiply
    //   3. if shift > 0, add half an LSB then arithmetic-shift right
    //   4. clip to OUT_LO .. OUT_HI
    //
    // Watch the widths: acc reaches 2^47 and mult 2^15, so the product
    // is up to 2^62. longint is 64-bit signed, so it fits -- but only
    // just, and it would not if ACC_W grew.
    // -----------------------------------------------------------------
    function automatic longint ref_requant(longint acc);
        longint m;      // which multiplier
        longint p;      // the product

        // 1. pick, on the sign of ACC (not of the result)
        //    mult and mult_neg are module-level signals -- the function can
        //    read them directly, no need to pass them in
        m = acc < 0 ? longint'($signed(mult_neg))
                : longint'($signed(mult));

        // 2. multiply
        p = acc * m;

        // 3. if shift > 0, add half an LSB then >>> shift
        //    half an LSB is  1 << (shift-1)
        //    guard shift == 0, where (shift-1) is a negative shift a
        if (shift > 0)
            p = (p + (longint'(1) << (shift - 1))) >>> shift;

        // 4. clip to OUT_LO .. OUT_HI, then return
        if (p < OUT_LO)         return OUT_LO;
        else if (p > OUT_HI)    return OUT_HI;
        return p;
    endfunction

    // -----------------------------------------------------------------
    // TODO 2: drive one cycle and check every lane.
    //
    //   task automatic run(string label, longint a [N_PE]);
    //
    // Pack the N_PE accumulator values into in_acc, raise in_valid for
    // one clock, wait for out_valid, then compare each lane's OUT_W bits
    // against ref_requant() and count.
    //
    // Two things that will bite:
    //   - out_valid is registered, so it lands the cycle AFTER the
    //     accumulator was accepted. Wait for it, bounded, rather than
    //     assuming a fixed delay.
    //   - out_data lanes are unsigned part-selects. $signed() them
    //     before comparing or every negative result looks wrong.
    // -----------------------------------------------------------------

    // -----------------------------------------------------------------
    // TODO 3: the cases. Aim at the things that are actually hard here,
    // not at coverage for its own sake:
    //
    //   ROUNDING   values landing exactly on .5 after the shift, both
    //              signs. This is the difference between round and
    //              truncate, and truncation is what erases detections
    //              across fifteen layers.
    //
    //   SATURATION accumulators far past what OUT_W can hold, both
    //              directions. The limits are asymmetric: -128 and +127.
    //              Wrapping instead of clamping is the bug that stays
    //              plausible, so make sure a wrap would fail loudly.
    //
    //   SHIFT = 0  the guard case. `1 << (shift-1)` is a negative shift
    //              amount, which is undefined.
    //
    //   NEGATIVE   accumulators below zero, where mult_neg is selected.
    //   ACC        Set mult_neg to something clearly different from mult
    //              (a tenth, as leaky ReLU does) so a DUT that ignores it
    //              cannot accidentally agree.
    //
    //   REAL       mult and shift from vectors/manifest.json -- layer 0
    //   CONSTANTS  is mult 16940, shift 30. Numbers that come from your
    //              own network, not made up.
    //
    //   RANDOM     the combinations you would not think to write.
    // -----------------------------------------------------------------

    initial begin
        $dumpfile("tb_requantize.vcd");
        $dumpvars(0, tb_requantize);

        repeat (4) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        // TODO 4: call your cases here.

        $display("");
        if (errors == 0 && checks > 0)
            $display("PASS  %0d lane values checked", checks);
        else if (checks == 0)
            $display("FAIL  nothing was checked -- the DUT never produced output");
        else
            $display("FAIL  %0d errors over %0d checks", errors, checks);
        $finish;
    end

endmodule
