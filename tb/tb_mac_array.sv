`timescale 1ns / 1ps

// Self-checking testbench for mac_array.
//
// Two independent checks, because they catch different things.
//
// A behavioural reference computed in the testbench catches arithmetic
// bugs: sign handling, a dropped tap, an accumulator that clears at the
// wrong time. It runs on directed edge cases and on random data, and it
// is independent of the DUT in the way that matters -- written as an
// obvious nested loop rather than as a parallel array.
//
// Golden vectors read from vectors/mac_*.mem catch a different class:
// divergence from the reference MODEL. A DUT can be perfectly
// self-consistent and still not compute yolov2-tiny. These vectors come
// from tools/dump_mac_vectors.py, which runs the real quantized
// arithmetic from reference/quant.py on real layer weights. They are
// skipped if the files are absent, so `make sim` works on a fresh
// checkout; run `make vectors` to generate them.
//
//   make sim        iverilog
//   make simv       verilator

module tb_mac_array;

    localparam int DATA_W = 16;
    localparam int ACC_W  = 48;
    localparam int K      = 3;
    localparam int N_PE   = 4;      // small, so failure messages stay readable

    localparam int TAPS   = K*K;

    logic clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    logic                          in_valid = 0;
    logic signed [TAPS*DATA_W-1:0] in_window = '0;
    logic                          in_ready;
    logic                          first_channel = 0;
    logic                          last_channel = 0;
    logic signed [N_PE*TAPS*DATA_W-1:0] weights = '0;
    logic signed [N_PE*ACC_W-1:0]  bias = '0;
    logic                          out_valid;
    logic signed [N_PE*ACC_W-1:0]  out_acc;

    mac_array #(.DATA_W(DATA_W), .ACC_W(ACC_W), .K(K), .N_PE(N_PE)) dut (
        .clk(clk), .rst_n(rst_n),
        .in_valid(in_valid), .in_window(in_window), .in_ready(in_ready),
        .first_channel(first_channel), .last_channel(last_channel),
        .weights(weights), .bias(bias),
        .out_valid(out_valid), .out_acc(out_acc)
    );

    int errors = 0;
    int checks = 0;

    // Guard against a DUT that never asserts out_valid. A testbench that
    // hangs reports nothing; one that fails says where to look.
    localparam time TIMEOUT_NS = 20ms;
    initial begin
        #TIMEOUT_NS;
        $display("");
        $display("FAIL  timed out after %0d checks -- out_valid never fired?", checks);
        $finish;
    end

    // ---------------------------------------------------------------
    // Behavioural reference: the same arithmetic, written the obvious
    // way. Unbounded `longint` so a wrong accumulator width in the DUT
    // shows up as a mismatch rather than as matching truncation.
    // ---------------------------------------------------------------

    localparam int MAX_CH = 1024;

    // Stimulus lives at module scope: Icarus supports neither dynamic
    // arrays nor unpacked subroutine ports, so the tasks read these
    // directly rather than taking them as arguments.
    longint win        [MAX_CH][TAPS];
    longint wgt        [MAX_CH][N_PE][TAPS];
    longint bs         [N_PE];
    longint expect_acc [N_PE];

    // ---------------------------------------------------------------
    // Drive one complete output pixel: C channels of windows, then
    // check the accumulators against the reference.
    // ---------------------------------------------------------------

    task automatic run_pixel(input string label, input int n_ch);
        longint got;
        begin
            for (int pe = 0; pe < N_PE; pe++) expect_acc[pe] = bs[pe];

            for (int ch = 0; ch < n_ch; ch++) begin
                @(negedge clk);
                in_valid      = 1;
                first_channel = (ch == 0);
                last_channel  = (ch == n_ch-1);

                for (int i = 0; i < TAPS; i++)
                    in_window[i*DATA_W +: DATA_W] = DATA_W'(win[ch][i]);
                for (int pe = 0; pe < N_PE; pe++)
                    for (int i = 0; i < TAPS; i++)
                        weights[(pe*TAPS + i)*DATA_W +: DATA_W] = DATA_W'(wgt[ch][pe][i]);
                for (int pe = 0; pe < N_PE; pe++)
                    bias[pe*ACC_W +: ACC_W] = ACC_W'(bs[pe]);

                // The reference, computed the obvious way: a nested loop
                // in unbounded longint, so a too-narrow accumulator in
                // the DUT shows up as a mismatch rather than as matching
                // truncation.
                for (int pe = 0; pe < N_PE; pe++)
                    for (int i = 0; i < TAPS; i++)
                        expect_acc[pe] += win[ch][i] * wgt[ch][pe][i];

                @(posedge clk);
                while (in_ready !== 1'b1) @(posedge clk);
            end

            @(negedge clk);
            in_valid = 0; first_channel = 0; last_channel = 0;

            // out_valid is registered, so it lands on the edge after the
            // last window was accepted. Bounded, so an unimplemented DUT
            // fails instead of hanging.
            begin
                int spin;
                spin = 0;
                while ((out_valid !== 1'b1) && spin < 8) begin
                    @(posedge clk);
                    spin++;
                end
                if (out_valid !== 1'b1) begin
                    $display("  FAIL %s: out_valid never asserted", label);
                    errors++;
                end
            end

            if (out_valid === 1'b1) begin
                for (int pe = 0; pe < N_PE; pe++) begin
                    got = longint'($signed(out_acc[pe*ACC_W +: ACC_W]));
                    checks++;
                    if (got !== expect_acc[pe]) begin
                        $display("  FAIL %s, pe %0d: got %0d, expected %0d",
                                 label, pe, got, expect_acc[pe]);
                        errors++;
                    end
                end
            end
            @(negedge clk);
        end
    endtask

    // ---------------------------------------------------------------
    // Cases
    // ---------------------------------------------------------------

    task automatic fill(input int n_ch, input longint wv, input longint gv);
        for (int ch = 0; ch < n_ch; ch++) begin
            for (int i = 0; i < TAPS; i++) win[ch][i] = wv;
            for (int pe = 0; pe < N_PE; pe++)
                for (int i = 0; i < TAPS; i++) wgt[ch][pe][i] = gv;
        end
    endtask

    task automatic fill_rand(input int n_ch);
        for (int ch = 0; ch < n_ch; ch++) begin
            for (int i = 0; i < TAPS; i++)
                win[ch][i] = longint'($signed(DATA_W'($random)));
            for (int pe = 0; pe < N_PE; pe++)
                for (int i = 0; i < TAPS; i++)
                    wgt[ch][pe][i] = longint'($signed(DATA_W'($random)));
        end
    endtask

    int counts [5];
    initial begin
        counts[0] = 1; counts[1] = 2; counts[2] = 3;
        counts[3] = 8; counts[4] = 64;
    end

    localparam longint HI = (1 << (DATA_W-1)) - 1;   //  32767
    localparam longint LO = -(1 << (DATA_W-1));      // -32768

    initial begin
        $dumpfile("tb_mac_array.vcd");
        $dumpvars(0, tb_mac_array);

        repeat (4) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        for (int pe = 0; pe < N_PE; pe++) bs[pe] = 0;

        // A zero window must leave exactly the bias -- this is the check
        // that catches a bias added at the wrong time or scaled wrong.
        fill(1, 0, 12345);
        for (int pe = 0; pe < N_PE; pe++) bs[pe] = 1000 + pe;
        run_pixel("zero window, nonzero bias", 1);

        // Signs. Verilog part-selects are unsigned, so a missing
        // $signed() gets three of these four wrong and nothing warns.
        for (int pe = 0; pe < N_PE; pe++) bs[pe] = 0;
                fill(1,  7,  11); run_pixel("pos x pos", 1);
        fill(1, -7,  11); run_pixel("neg x pos", 1);
        fill(1,  7, -11); run_pixel("pos x neg", 1);
        fill(1, -7, -11); run_pixel("neg x neg", 1);

        // Extremes, where a too-narrow accumulator truncates.
        fill(1, LO, LO); run_pixel("min x min", 1);
        fill(1, HI, HI); run_pixel("max x max", 1);
        fill(1, LO, HI); run_pixel("min x max", 1);

        // Accumulation across channels. If the accumulator clears on the
        // wrong cycle, only the multi-channel cases notice.
        for (int c = 0; c < 5; c++) begin
            int n_ch;
            n_ch = counts[c];
            fill(n_ch, 3, 5);
            for (int pe = 0; pe < N_PE; pe++) bs[pe] = -50 * pe;
            run_pixel($sformatf("accumulate %0d channels", n_ch), n_ch);
        end

        // Deep accumulation at full magnitude: 1024 channels x 9 taps is
        // layer 13's shape, and the point where a 32-bit accumulator
        // would silently wrap.
        fill(1024, LO, HI);
        for (int pe = 0; pe < N_PE; pe++) bs[pe] = 0;
        run_pixel("1024 channels at full scale", 1024);

        // Random, which finds the combinations hand-written cases miss.
        for (int t = 0; t < 20; t++) begin
            int n_ch = 1 + (t % 7);
            fill_rand(n_ch);
            for (int pe = 0; pe < N_PE; pe++)
                bs[pe] = longint'($signed(32'($random)));
            run_pixel($sformatf("random %0d", t), n_ch);
        end

        $display("");
        if (errors == 0 && checks > 0)
            $display("PASS  %0d accumulator values checked", checks);
        else if (checks == 0)
            $display("FAIL  nothing was checked -- the DUT never produced output");
        else
            $display("FAIL  %0d errors over %0d checks", errors, checks);
        $finish;
    end

endmodule
