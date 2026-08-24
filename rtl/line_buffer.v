// Sliding-window generator for the convolution datapath.
//
// Streams pixels in and emits, for every output position, the K x K
// window that the MAC array multiplies against one filter's kernel.
// This is the hardware form of the tap loop in the reference model:
// where reference/layers.py gathers nine shifted planes with numpy
// slices, here nine registers hold the same nine values as the image
// walks past them.
//
// Dataflow. The input arrives in raster order with channel innermost:
//
//     for row: for col: for ch: one DATA_W sample
//
// and the output is one K*K window per (output position, channel).
// Channels stream one at a time rather than in parallel because C_IN
// reaches 1024 on layer 13; a channel-parallel window would be 9216
// wires.
//
// Storage. A K x K window spans K rows, K-1 of which must be held
// while the newest one arrives, so the buffer is (K-1) * IMG_W * C_IN
// samples. Across every layer of yolov2-tiny the worst case is layer
// 13 at 2 * 13 * 1024 = 26 KB, under six BRAM36 blocks, because the
// image width halves exactly as the channel count doubles. One
// instance sized for that layer serves them all.
//
// Padding is zero-fill on all four edges, which is what darknet's
// convolution does. Note this is NOT what its maxpool does -- that
// pads one-sided with -inf and belongs in a different module.

module line_buffer #(
    parameter integer DATA_W = 8,      // sample width, signed
    parameter integer IMG_W  = 416,    // input width in pixels
    parameter integer IMG_H  = 416,    // input height in pixels
    parameter integer C_IN   = 3,      // input channels
    parameter integer K      = 3,      // kernel size
    parameter integer PAD    = 1       // pixels of zero padding per edge
)(
    input  wire                          clk,
    input  wire                          rst_n,

    // input stream: raster order, channel innermost
    input  wire                          in_valid,
    input  wire signed [DATA_W-1:0]      in_data,
    output wire                          in_ready,

    // output stream: one K*K window per (output position, channel)
    output wire                          out_valid,
    output wire signed [K*K*DATA_W-1:0]  out_window,  // row-major, [0] = top-left
    output wire [$clog2(C_IN > 1 ? C_IN : 2)-1:0] out_ch,
    output wire                          out_last     // final window of the frame
);

    // Output geometry, same formula the reference model uses.
    localparam integer W_OUT = (IMG_W + 2*PAD - K) + 1;
    localparam integer H_OUT = (IMG_H + 2*PAD - K) + 1;

    // TODO 1: declare the storage. It needs to hold K-1 rows of
    // IMG_W * C_IN samples. Write it as a flat reg array indexed by a
    // linear address rather than a 3-D array: Vivado infers block RAM
    // from the flat form and will fall back to distributed LUT RAM,
    // which at 26 KB will not fit, if you nest the dimensions.

    // TODO 2: input counters. Track which (row, col, channel) the
    // incoming sample belongs to, incrementing channel innermost so
    // they match the stream order. These are what turn a bare valid
    // handshake into a position.

    // TODO 3: the write address. Only K-1 rows are live at a time, so
    // the row part wraps: the arriving row overwrites the oldest one
    // still held. Work out the address from the counters in TODO 2.
    // An off-by-one here reads a neighbouring row's data and produces
    // a window that looks plausible, so get it right before trusting
    // any output.

    // TODO 4: the window registers. K*K of them, DATA_W wide. On each
    // accepted sample they shift left by one column, and the rightmost
    // column takes the newly available values: the K-1 buffered rows
    // plus in_data itself as the bottom row. Note the window's rows
    // come from different sources -- the buffer for the old rows, the
    // input for the new one.

    // TODO 5: padding. A window position whose source row or column
    // falls outside the image must read zero rather than whatever the
    // buffer happens to hold. Derive the out-of-range condition from
    // the output position, not the input one -- the two differ by the
    // pipeline delay below.

    // TODO 6: output position and valid. A window for output row r is
    // only complete once row r+PAD has arrived, so out_valid lags
    // in_valid by roughly one row plus one pixel. Track the output
    // (row, col, channel) separately from the input counters and
    // assert out_valid exactly when a full window is available.
    // out_last marks the final window of the frame, which is what lets
    // downstream logic flush without counting.

    // TODO 7: backpressure. in_ready can be tied high for now -- the
    // module never stalls on its own -- but leave the port so the DMA
    // side does not need rewiring when a downstream FIFO fills.

endmodule
