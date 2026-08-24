`timescale 1ns / 1ps

// Self-checking testbench for line_buffer.
//
// Drives a counting pattern through the DUT and checks every emitted
// window against one computed independently here. The pattern makes
// each sample's value a function of its position, so a window that
// reads the wrong address produces a visibly wrong number rather than
// plausible-looking data -- the same reason the Python tests compare
// against nested loops instead of stored expectations.
//
// Deliberately small: 8x6 with 2 channels runs in milliseconds and the
// failure messages stay readable. Scale the parameters up once it
// passes; the layer-0 geometry is IMG_W=416, IMG_H=416, C_IN=3.
//
//   make sim                  iverilog, fast iteration
//   make lint                 verilator's stricter static checks
//   make simv                 verilator simulation

module tb_line_buffer;

    localparam int DATA_W = 8;
    localparam int IMG_W  = 8;
    localparam int IMG_H  = 6;
    localparam int C_IN   = 2;
    localparam int K      = 3;
    localparam int PAD    = 1;

    localparam int W_OUT  = (IMG_W + 2*PAD - K) + 1;
    localparam int H_OUT  = (IMG_H + 2*PAD - K) + 1;
    localparam int N_OUT  = W_OUT * H_OUT * C_IN;
    localparam int CH_W   = $clog2(C_IN > 1 ? C_IN : 2);

    // Guards against a DUT that never handshakes. A testbench that
    // hangs tells you nothing; one that fails tells you where to look.
    localparam int STALL_MAX  = 1000;
    localparam time TIMEOUT_NS = 10ms;

    // Every sample is a function of where it is, so a misaddressed read
    // shows up as a wrong value instead of believable data.
    function automatic logic signed [DATA_W-1:0] pattern(int r, c, ch);
        return DATA_W'(((r * IMG_W + c) * C_IN + ch) % 127);
    endfunction

    // Zero outside the image: the padding convolution expects.
    function automatic logic signed [DATA_W-1:0] sample(int r, c, ch);
        if (r < 0 || r >= IMG_H || c < 0 || c >= IMG_W) return '0;
        else                                            return pattern(r, c, ch);
    endfunction

    logic clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    logic                     in_valid = 0;
    logic signed [DATA_W-1:0] in_data  = 0;
    logic                     in_ready;
    logic                     out_valid;
    logic signed [K*K*DATA_W-1:0] out_window;
    logic [CH_W-1:0]          out_ch;
    logic                     out_last;

    line_buffer #(
        .DATA_W(DATA_W), .IMG_W(IMG_W), .IMG_H(IMG_H),
        .C_IN(C_IN), .K(K), .PAD(PAD)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .in_valid(in_valid), .in_data(in_data), .in_ready(in_ready),
        .out_valid(out_valid), .out_window(out_window),
        .out_ch(out_ch), .out_last(out_last)
    );

    int errors = 0;
    int seen   = 0;

    // Expected windows are consumed in emission order: output row, then
    // column, then channel.
    task automatic check_window(int idx);
        int orow, ocol, och;
        logic signed [DATA_W-1:0] want, got;
        begin
            och  =  idx % C_IN;
            ocol = (idx / C_IN) % W_OUT;
            orow =  idx / (C_IN * W_OUT);

            if (out_ch !== CH_W'(och)) begin
                $display("  FAIL window %0d: out_ch %0d, expected %0d",
                         idx, out_ch, och);
                errors++;
            end

            for (int i = 0; i < K; i++)
                for (int j = 0; j < K; j++) begin
                    want = sample(orow - PAD + i, ocol - PAD + j, och);
                    got  = out_window[(i*K + j)*DATA_W +: DATA_W];
                    if (got !== want) begin
                        $display("  FAIL window %0d (row %0d col %0d ch %0d) tap [%0d][%0d]: got %0d, expected %0d",
                                 idx, orow, ocol, och, i, j, got, want);
                        errors++;
                    end
                end

            if (out_last !== (idx == N_OUT - 1)) begin
                $display("  FAIL window %0d: out_last %0b, expected %0b",
                         idx, out_last, (idx == N_OUT-1));
                errors++;
            end
        end
    endtask

    // Collector: every emitted window is checked as it appears.
    always_ff @(posedge clk) begin
        if (rst_n && out_valid) begin
            if (seen < N_OUT) check_window(seen);
            else begin
                $display("  FAIL: extra window emitted past the expected %0d", N_OUT);
                errors++;
            end
            seen++;
        end
    end

    // Watchdog. Nothing here should take milliseconds; if it does, the
    // DUT is stalled and the run must end with a verdict rather than
    // spinning until someone notices.
    initial begin
        #TIMEOUT_NS;
        $display("");
        $display("FAIL  timed out with %0d of %0d windows emitted", seen, N_OUT);
        $display("      (in_ready never asserted, or out_valid never fired)");
        $finish;
    end

    int cyc, stall;
    initial begin
        $dumpfile("tb_line_buffer.vcd");
        $dumpvars(0, tb_line_buffer);

        repeat (4) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        // stream the frame in, raster order, channel innermost
        for (int r = 0; r < IMG_H; r++)
            for (int c = 0; c < IMG_W; c++)
                for (int ch = 0; ch < C_IN; ch++) begin
                    @(negedge clk);
                    in_valid = 1;
                    in_data  = pattern(r, c, ch);
                    @(posedge clk);
                    // Honour backpressure, but bounded: an undriven
                    // in_ready reads as 0 under Verilator, and an
                    // unbounded wait would hang instead of failing.
                    stall = 0;
                    while (!in_ready && stall < STALL_MAX) begin
                        @(posedge clk);
                        stall++;
                    end
                end

        @(negedge clk);
        in_valid = 0;

        // let the pipeline drain
        cyc = 0;
        while (seen < N_OUT && cyc < 10*N_OUT + 1000) begin
            @(posedge clk);
            cyc++;
        end

        if (seen != N_OUT) begin
            $display("  FAIL: %0d windows emitted, expected %0d", seen, N_OUT);
            errors++;
        end

        $display("");
        if (errors == 0)
            $display("PASS  %0d windows checked, %0dx%0d image, %0d channels",
                     seen, IMG_W, IMG_H, C_IN);
        else
            $display("FAIL  %0d errors over %0d windows", errors, seen);
        $finish;
    end

endmodule
