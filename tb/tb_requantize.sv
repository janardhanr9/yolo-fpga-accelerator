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

    // reference/quant.py's requantize(), one lane, in unbounded longint
    // so a too-narrow datapath in the DUT shows up as a mismatch rather
    // than as matching truncation. acc reaches 2^47 and mult 2^15, so
    // the product is up to 2^62 -- inside a signed longint, but only
    // just, and it would not be if ACC_W grew.
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

    // Stimulus at module scope: Icarus takes neither dynamic arrays nor
    // unpacked subroutine ports, so run() reads these rather than
    // receiving them.

    longint accs [N_PE];
    task automatic run(string label);
        longint got, want;
        begin
            @(negedge clk); //drive signals on negedge
            in_valid = 1;
            for (int pe = 0; pe < N_PE; pe++)
                in_acc[pe*ACC_W +:ACC_W] = ACC_W'(accs[pe]);
            @(posedge clk);
            @(negedge clk); // wait one cycle for posedge then on next negedge wait
            in_valid = 0;
            if (out_valid !== 1'b1) begin
                $display("  FAIL %s: out_valid not asserted one cycle after in_valid", label);
                errors++;
            end

            if (out_valid === 1'b1) begin
                for (int pe = 0; pe < N_PE; pe++) begin
                    got = longint'($signed(out_data[pe*OUT_W +: OUT_W]));
                    want = ref_requant(accs[pe]);
                    checks++;
                    if (got !== want) begin
                        $display("  FAIL %s, lane %0d: got %0d, expected %0d",
                                label, pe, got, want);
                        errors++;
                    end
                end
            end

            // out_valid must fall again. Without this the suite passes a
            // DUT whose out_valid sets and never clears, which would tell
            // the next stage every cycle carries a finished result.
            @(posedge clk); #1;
            if (out_valid !== 1'b0) begin
                $display("  FAIL %s: out_valid still high a cycle later", label);
                errors++;
            end
            checks++;
        end
    endtask

    initial begin
        $dumpfile("tb_requantize.vcd");
        $dumpvars(0, tb_requantize);

        repeat (4) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        // Each case is chosen so that one specific plausible bug gives a
        // DIFFERENT number. A case where the buggy and correct versions
        // agree tests nothing.

        // ---- rounding -------------------------------------------------
        // mult 1, shift 3 means out = acc/8, so the arithmetic is checkable
        // by hand. 4/8 and 12/8 land exactly on .5, which is where floor
        // and round disagree.
        mult = 16'sd1; mult_neg = 16'sd1; shift = 6'd3;
        accs[0] = 3;  accs[1] = 4;  accs[2] = 11;  accs[3] = 12;
        run("rounding at .5");           // truncating gives 0 and 1, not 1 and 2

        accs[0] = -3; accs[1] = -4; accs[2] = -11; accs[3] = -12;
        run("rounding, negative");       // truncating gives -1 and -2, not 0 and -1

        // ---- saturation -----------------------------------------------
        // The limits are asymmetric: -128 and +127. A DUT that wraps stays
        // plausible, which is what makes it the dangerous failure.
        mult = 16'sd1; mult_neg = 16'sd1; shift = 6'd0;
        accs[0] = 127; accs[1] = 128; accs[2] = 1000; accs[3] = 99999;
        run("saturate high");            // wrapping gives -128 for acc 128

        accs[0] = -128; accs[1] = -129; accs[2] = -1000; accs[3] = -99999;
        run("saturate low");             // wrapping gives +127 for acc -129

        // ---- shift = 0 ------------------------------------------------
        // The guard case: half an LSB is 1 << (shift-1), which at shift 0
        // is a negative shift amount. Nothing is being discarded here, so
        // there is nothing to round.
        mult = 16'sd3; mult_neg = 16'sd3; shift = 6'd0;
        accs[0] = 0; accs[1] = 1; accs[2] = 40; accs[3] = -40;
        run("shift = 0");

        // ---- the leaky branch -----------------------------------------
        // mult_neg deliberately a tenth of mult, as leaky ReLU makes it,
        // and far enough apart that a DUT ignoring it cannot coincide.
        mult = 16'sd100; mult_neg = 16'sd10; shift = 6'd4;
        accs[0] = 16; accs[1] = -16; accs[2] = 32; accs[3] = -32;
        run("mult_neg on negative acc");  // ignoring it gives -100, not -10

        // ---- the network's own constants ------------------------------
        // From vectors/manifest.json, layer 0. Numbers that came out of
        // this network rather than being invented for the test.
        mult = 16'sd16940; mult_neg = 16'sd1694; shift = 6'd30;
        accs[0] = 1234567; accs[1] = -1234567;
        accs[2] = 10000000; accs[3] = -10000000;
        run("layer 0 constants");

        // ---- random ---------------------------------------------------
        // The combinations nobody writes by hand. mult stays positive
        // because real ones always are, and shift stays under 35 so the
        // REFERENCE cannot overflow: acc*mult reaches 2^62, and adding
        // 1 << (shift-1) on top has to stay inside a signed longint.
        for (int t = 0; t < 20; t++) begin
            mult     = 16'($urandom_range(1, 32767));
            mult_neg = 16'($urandom_range(1, 32767));
            shift    = 6'($urandom_range(0, 34));
            for (int pe = 0; pe < N_PE; pe++)
                accs[pe] = longint'($random) * longint'($urandom_range(1, 1000));
            run($sformatf("random %0d", t));
        end

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
