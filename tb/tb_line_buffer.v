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
//   iverilog -g2012 -o tb.out tb/tb_line_buffer.v rtl/line_buffer.v
//   vvp tb.out

module tb_line_buffer;

    localparam integer DATA_W = 8;
    localparam integer IMG_W  = 8;
    localparam integer IMG_H  = 6;
    localparam integer C_IN   = 2;
    localparam integer K      = 3;
    localparam integer PAD    = 1;

    localparam integer W_OUT  = (IMG_W + 2*PAD - K) + 1;
    localparam integer H_OUT  = (IMG_H + 2*PAD - K) + 1;
    localparam integer N_OUT  = W_OUT * H_OUT * C_IN;

    // Every sample is a function of where it is, so a misaddressed read
    // shows up as a wrong value instead of believable data.
    function signed [DATA_W-1:0] pattern(input integer r, c, ch);
        pattern = ((r * IMG_W + c) * C_IN + ch) % 127;
    endfunction

    // Zero outside the image: the padding convolution expects.
    function signed [DATA_W-1:0] sample(input integer r, c, ch);
        if (r < 0 || r >= IMG_H || c < 0 || c >= IMG_W) sample = 0;
        else                                            sample = pattern(r, c, ch);
    endfunction

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    reg                      in_valid = 0;
    reg signed [DATA_W-1:0]  in_data  = 0;
    wire                     in_ready;
    wire                     out_valid;
    wire signed [K*K*DATA_W-1:0] out_window;
    wire [$clog2(C_IN > 1 ? C_IN : 2)-1:0] out_ch;
    wire                     out_last;

    line_buffer #(
        .DATA_W(DATA_W), .IMG_W(IMG_W), .IMG_H(IMG_H),
        .C_IN(C_IN), .K(K), .PAD(PAD)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .in_valid(in_valid), .in_data(in_data), .in_ready(in_ready),
        .out_valid(out_valid), .out_window(out_window),
        .out_ch(out_ch), .out_last(out_last)
    );

    integer errors = 0;
    integer seen   = 0;

    // Expected windows are consumed in emission order: output row, then
    // column, then channel.
    task check_window(input integer idx);
        integer orow, ocol, och, i, j, tap;
        reg signed [DATA_W-1:0] want, got;
        begin
            och  =  idx % C_IN;
            ocol = (idx / C_IN) % W_OUT;
            orow =  idx / (C_IN * W_OUT);

            if (out_ch !== och[$clog2(C_IN > 1 ? C_IN : 2)-1:0]) begin
                $display("  FAIL window %0d: out_ch %0d, expected %0d",
                         idx, out_ch, och);
                errors = errors + 1;
            end

            for (i = 0; i < K; i = i + 1)
                for (j = 0; j < K; j = j + 1) begin
                    tap  = i * K + j;
                    want = sample(orow - PAD + i, ocol - PAD + j, och);
                    got  = out_window[tap*DATA_W +: DATA_W];
                    if (got !== want) begin
                        $display("  FAIL window %0d (row %0d col %0d ch %0d) tap [%0d][%0d]: got %0d, expected %0d",
                                 idx, orow, ocol, och, i, j, got, want);
                        errors = errors + 1;
                    end
                end

            if (out_last !== (idx == N_OUT - 1)) begin
                $display("  FAIL window %0d: out_last %0b, expected %0b",
                         idx, out_last, (idx == N_OUT-1));
                errors = errors + 1;
            end
        end
    endtask

    // Collector: every accepted output window is checked as it appears.
    always @(posedge clk) begin
        if (rst_n && out_valid) begin
            if (seen < N_OUT) check_window(seen);
            else begin
                $display("  FAIL: extra window emitted past the expected %0d", N_OUT);
                errors = errors + 1;
            end
            seen = seen + 1;
        end
    end

    integer r, c, ch, cyc;
    initial begin
        $dumpfile("tb_line_buffer.vcd");
        $dumpvars(0, tb_line_buffer);

        repeat (4) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        // stream the frame in, raster order, channel innermost
        for (r = 0; r < IMG_H; r = r + 1)
            for (c = 0; c < IMG_W; c = c + 1)
                for (ch = 0; ch < C_IN; ch = ch + 1) begin
                    @(negedge clk);
                    in_valid = 1;
                    in_data  = pattern(r, c, ch);
                    @(posedge clk);
                    while (!in_ready) @(posedge clk);   // honour backpressure
                end

        @(negedge clk);
        in_valid = 0;

        // let the pipeline drain
        cyc = 0;
        while (seen < N_OUT && cyc < 10*N_OUT + 1000) begin
            @(posedge clk);
            cyc = cyc + 1;
        end

        if (seen != N_OUT) begin
            $display("  FAIL: %0d windows emitted, expected %0d", seen, N_OUT);
            errors = errors + 1;
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
