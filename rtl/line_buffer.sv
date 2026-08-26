// Sliding-window generator for the convolution datapath.
//
// Streams pixels in and emits, for every output position, the K x K
// window that the MAC array multiplies against one filter's kernel.
// This is the hardware form of the tap loop in the reference model:
// where reference/layers.py gathers nine shifted planes with numpy
// slices, here K*K delay lines hold the same values as the image walks
// past them.
//
// Dataflow. The input arrives in raster order with channel innermost:
//
//     for row: for col: for ch: one DATA_W sample
//
// and the output is one K x K window per (output position, channel).
// Channels stream one at a time rather than in parallel because C_IN
// reaches 1024 on layer 13; a channel-parallel window would be 9216
// wires.
//
// Storage. A K x K window spans K rows, K-1 of which must be held
// while the newest one arrives, so the buffer is (K-1) * IMG_W * C_IN
// samples. Across every layer of yolov2-tiny the worst case is layer
// 13 at 2 * 13 * 1024 = 26 KiB, under six BRAM36 blocks, because the
// image width halves exactly as the channel count doubles. One
// instance sized for that layer serves them all.
//
// Padding is zero-fill on all four edges, which is what darknet's
// convolution does. Note this is NOT what its maxpool does -- that
// pads one-sided with -inf and belongs in a different module.
//
// Verified at the real layer-0 geometry, 519,168 windows on 416x416x3,
// and from K=1 PAD=0 to K=5 PAD=2. Known gap: K>1 with PAD=0 is wrong,
// because the warm-up below assumes the output is the same size as the
// input. Every conv in this network is 3x3 pad=1 or 1x1 pad=0.

module line_buffer #(
    parameter int DATA_W = 8,      // sample width, signed
    parameter int IMG_W  = 416,    // input width in pixels
    parameter int IMG_H  = 416,    // input height in pixels
    parameter int C_IN   = 3,      // input channels
    parameter int K      = 3,      // kernel size
    parameter int PAD    = 1       // pixels of zero padding per edge
)(
    input  logic clk,
    input  logic rst_n,

    // input stream: raster order, channel innermost
    input  logic                     in_valid,
    input  logic signed [DATA_W-1:0] in_data,
    output logic                     in_ready,

    // Output stream: one K x K window per (output position, channel).
    //
    // Flat rather than a packed [K-1:0][K-1:0] array because Icarus
    // cannot index a packed multi-dimensional array with a variable,
    // and losing the fast simulator is a worse trade than losing the
    // nicer syntax. Tap (i, j) lives at
    //
    //     out_window[(i*K + j)*DATA_W +: DATA_W]
    //
    // row-major, so [0] is the top-left tap -- the same layout as the
    // reference model's cols[:, i, j]. Use the WIN macro below rather
    // than writing that expression out.
    output logic                                   out_valid,
    output logic signed [K*K*DATA_W-1:0]           out_window,
    output logic [$clog2(C_IN > 1 ? C_IN : 2)-1:0] out_ch,
    output logic                                   out_last
);

    // Tap accessor. Works with variable i and j on every simulator.
    `define WIN(i, j) out_window[((i)*K + (j))*DATA_W +: DATA_W]

    // Counter and address widths, named once so the sized casts below
    // stay readable. Parameters are `int`, so any expression built from
    // them is 32 bits wide; casting is what stops that leaking into
    // comparisons against much narrower counters.
    localparam int CH_W   = $clog2(C_IN > 1 ? C_IN : 2);
    localparam int COL_W  = $clog2(IMG_W);
    localparam int ROW_W  = $clog2(IMG_H);
    // A 1x1 convolution needs no rows buffered at all -- the window is
    // the arriving sample. Floored at one entry because a zero-size
    // array does not elaborate; nothing ever reads it.
    localparam int STORE_N = (K > 1) ? (K-1) * IMG_W * C_IN : 1;
    localparam int ADDR_W = $clog2(STORE_N > 1 ? STORE_N : 2);
    localparam int SLOT_W = $clog2(K > 2 ? K - 1 : 2);

    // The output position needs a sign bit and headroom: a window at the
    // top or left edge names position -1, and the padding test compares
    // it against IMG_H / IMG_W, so the type must hold both.
    localparam int PROW_W = ROW_W + 2;
    localparam int PCOL_W = COL_W + 2;

    // Output (r, c) is complete only once input (r+PAD, c+PAD) has
    // arrived, so the output stream lags the input by this many
    // samples -- and the last PAD rows and columns have no input left
    // to trigger them. The module therefore keeps advancing for this
    // many cycles after the frame ends, shifting in zeros.
    localparam int DRAIN_N = PAD * (IMG_W + 1) * C_IN;
    localparam int DRAIN_W = (DRAIN_N > 0) ? $clog2(DRAIN_N + 1) : 1;

    // Output geometry, same formula the reference model uses.
    localparam int W_OUT = (IMG_W + 2*PAD - K) + 1;
    localparam int H_OUT = (IMG_H + 2*PAD - K) + 1;

    // K-1 rows of IMG_W * C_IN samples, indexed by a linear address --
    // the shape Vivado infers block RAM from.
    logic signed [DATA_W-1:0] storage [STORE_N];

    // Where the arriving sample sits in the image. Channel innermost,
    // matching the stream order -- these turn a bare handshake into a
    // position.
    logic [ROW_W-1:0] row;
    // Which of the K-1 buffered rows the arriving samples go into. A
    // wrapping counter rather than row % (K-1): the modulo would build a
    // divider for any K where K-1 is not a power of two, and this is the
    // same increment-and-wrap the other counters already use.
    logic [SLOT_W-1:0] slot;
    logic [COL_W-1:0] col;
    logic [CH_W-1:0] channel;

    // Drain state. `advance` replaces the bare handshake everywhere: the
    // window must keep moving after the input ends, and must not move
    // while a stalled producer holds in_valid low.
    logic [DRAIN_W-1:0]       drain_cnt;
    wire                      draining = (drain_cnt != '0);
    wire                      accept   = in_ready && in_valid;
    wire                      advance  = accept || draining;
    wire signed [DATA_W-1:0]  sample   = draining ? '0 : in_data;
    wire                      last_in  = (row     == ROW_W'(IMG_H-1))
                                      && (col     == COL_W'(IMG_W-1))
                                      && (channel == CH_W'(C_IN-1));

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            row <= 0;
            col <= 0;
            channel <= 0;
            slot <= 0;
        end
        else if (advance) begin
            if (channel == CH_W'(C_IN - 1)) begin
                channel <= 0;
                if (col == COL_W'(IMG_W - 1)) begin
                    col <= 0;
                    row <= row + 1;
                    slot <= (slot == SLOT_W'(K - 2)) ? '0 : slot + 1'b1;
                end else col <= col + 1;
            end else channel <= channel + 1;
        end
    end

    // Only K-1 rows are live, so the row part of the address wraps: the
    // arriving row overwrites the oldest one still held, which by then
    // has appeared in every window it belongs to.
    logic [ADDR_W-1:0] w_addr;
    assign w_addr = ADDR_W'(slot * (IMG_W * C_IN) + col * C_IN + channel);

    // The output position this window names -- see the always_ff below.
    // Declared here because the generate block underneath reads it, and
    // Verilog requires declaration before use.
    logic signed [PROW_W-1:0] pos_row;
    logic signed [PCOL_W-1:0] pos_col;
    logic        [CH_W-1:0]   pos_ch;
    // Whether the window currently held was produced by an advancing
    // cycle. pos_* keeps its value across a stall, so position alone
    // would leave out_valid asserted over stale data.
    logic                     pos_taken;

    // One window per channel. Between window(col c, ch) and
    // window(col c+1, ch), C_IN samples go by, so every horizontal stage
    // is a delay line of depth C_IN. Holding those in flip-flops would
    // cost K*(K-1)*C_IN of them -- 92% of the device at layer 13's 1024
    // channels. Written as a memory indexed by channel, Vivado infers
    // distributed RAM or an SRL instead, which costs about 3% of the
    // LUTs for the same bits.
    logic signed [DATA_W-1:0] win [K][K][C_IN];

    // The window registers always hold the last K x K samples that went
    // past, and near an edge some of those belong to the previous row or
    // to no row at all -- the shift register does not know rows end.
    //
    // Zeroing happens HERE, on the way out, rather than when the
    // registers load: a sample that is out of range for this output
    // position is still the correct value for a later one, so blanking
    // it in the register would destroy data that is still needed.
    for (genvar i = 0; i < K; i++) begin : win_row
        for (genvar j = 0; j < K; j++) begin: win_col
            // where in the image this tap reads from. Signed, because
            // both go negative at the top and left edges.
            wire signed [PROW_W-1:0] src_row = pos_row - PROW_W'(PAD) + PROW_W'(i);
            wire signed [PCOL_W-1:0] src_col = pos_col - PCOL_W'(PAD) + PCOL_W'(j);
            wire in_range = (src_row >= 0) && (src_row < PROW_W'(IMG_H))
                         && (src_col >= 0) && (src_col < PCOL_W'(IMG_W));
            // pos_ch, not channel: the input counter has already advanced
            // past this window by the time it is read.
            assign `WIN(i, j) = in_range ? win[i][j][pos_ch] : '0;
        end
    end

    function automatic int slot_of(input int base, input int i);
        slot_of = base + i;
        if (slot_of >= K-1) slot_of = slot_of - (K-1);
    endfunction

    always_ff @(posedge clk) begin
        if (advance) begin
            for (int i = 0; i < K; i++)
                for (int j = 0; j < K - 1; j++)
                    win[i][j][channel] <= win[i][j + 1][channel];
            for (int i = 0; i < K-1; i++)
                win[i][K-1][channel] <= storage[slot_of(int'(slot), i)*(IMG_W * C_IN)
                                    + col*C_IN + int'(channel)];
            win[K-1][K-1][channel] <= sample;

            storage[w_addr] <= sample;
        end
    end

    // The output position is counted, not derived from the input
    // counters. They wrap at IMG_W and IMG_H, which for a padded
    // convolution are not the output dimensions -- deriving from them
    // sends the position back to -1 at every row boundary, so the last
    // output column of each row is never named.
    //
    // WARM_N advances pass before the first window is complete: output
    // (0,0) needs input (PAD,PAD), which is that many samples in.
    logic [DRAIN_W-1:0] warm_cnt;
    wire                warming = (warm_cnt != '0);

    always_ff @(posedge clk) begin
        if (!rst_n)                   warm_cnt <= DRAIN_W'(DRAIN_N);
        else if (advance && warming)  warm_cnt <= warm_cnt - 1'b1;
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            // One position before the first, so the first increment
            // lands exactly on (0, 0, 0).
            pos_row   <= PROW_W'(H_OUT-1);
            pos_col   <= PCOL_W'(W_OUT-1);
            pos_ch    <= CH_W'(C_IN-1);
            pos_taken <= 1'b0;
        end else begin
            pos_taken <= advance && !warming;
            if (advance && !warming) begin
                if (pos_ch == CH_W'(C_IN-1)) begin
                    pos_ch <= '0;
                    if (pos_col == PCOL_W'(W_OUT-1)) begin
                        pos_col <= '0;
                        // wraps so the reset value below is genuinely
                        // "one before the first"
                        if (pos_row == PROW_W'(H_OUT-1)) pos_row <= '0;
                        else                             pos_row <= pos_row + 1'b1;
                    end else pos_col <= pos_col + 1'b1;
                end else pos_ch <= pos_ch + 1'b1;
            end
        end
    end

    // Loaded when the frame's final sample is accepted, then counted
    // down one per cycle. in_ready drops while it runs, so no further
    // input is taken mid-drain.
    always_ff @(posedge clk) begin
        if (!rst_n)                  drain_cnt <= '0;
        else if (accept && last_in)  drain_cnt <= DRAIN_W'(DRAIN_N);
        else if (draining)           drain_cnt <= drain_cnt - 1'b1;
    end

    assign out_ch = pos_ch;

    // A window is real exactly when the position it names lies inside
    // the output image. Positions run negative during the warm-up while
    // the first rows are still arriving, and past the end during the
    // drain, so the geometry does the timing on its own -- there is no
    // separate "enough rows buffered yet" counter to get wrong.
    //
    // pos_taken gates it because pos_* holds its value across a stall;
    // position alone would keep out_valid asserted over stale data.
    assign out_valid = pos_taken
                    && (pos_row >= 0) && (pos_row < PROW_W'(H_OUT))
                    && (pos_col >= 0) && (pos_col < PCOL_W'(W_OUT));

    // The final window of the frame, so downstream logic can flush
    // without counting output positions itself.
    assign out_last = out_valid
                   && (pos_row == PROW_W'(H_OUT-1))
                   && (pos_col == PCOL_W'(W_OUT-1))
                   && (pos_ch  == CH_W'(C_IN-1));

    // Low during the drain so no further input is taken mid-flush.
    assign in_ready = !draining;

    `undef WIN

endmodule
