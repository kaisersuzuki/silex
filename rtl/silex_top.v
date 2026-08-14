// SILEX-1D top level.
//
// Software-free inference chip instance: 196->48->10 MLP, weights in ROM,
// per-layer hardwired FSMs, elastic valid/ready pipeline, power-on BIST.
//
// External contract:
//   - wait for READY (BIST passed)
//   - stream 196 pixel bytes on in_data/in_valid (in_ready flow-controls)
//   - class_valid pulses with class_out = argmax digit
//   - FAULT high = self-test failed, chip refuses input
module silex_top (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       in_valid,
    output wire       in_ready,
    input  wire [7:0] in_data,
    output wire       class_valid,
    output wire [3:0] class_out,
    output wire       ready,
    output wire       fault
);
    `include "silex_params.vh"
    // BIST drives the pipeline until READY; external port gated off before.
    wire        g_valid, l1_in_ready;
    wire [7:0]  g_data;
    wire        mux_valid = ready ? (in_valid & ~fault) : g_valid;
    wire [7:0]  mux_data  = ready ? in_data  : g_data;
    assign in_ready = ready & ~fault & l1_in_ready;

    wire        l1_ov, l1_or;
    wire [31:0] l1_od;
    silex_layer #(
        .NIN(NIN), .NOUT(NHID), .REQUANT(1), .M(M1), .SH(SH1),
        .WFILE("w1.hex"), .BFILE("b1.hex"),
        .AW_IN(AW_IN), .AW_OUT(AW_HID), .AW_ROM(AW_W1)
    ) u_l1 (
        .clk(clk), .rst_n(rst_n),
        .in_valid(mux_valid), .in_ready(l1_in_ready), .in_data(mux_data),
        .out_valid(l1_ov), .out_ready(l1_or), .out_data(l1_od)
    );

    wire        l2_ov, l2_or;
    wire [31:0] l2_od;
    silex_layer #(
        .NIN(NHID), .NOUT(NOUT), .REQUANT(0), .M(32'sd1), .SH(1),
        .WFILE("w2.hex"), .BFILE("b2.hex"),
        .AW_IN(AW_HID), .AW_OUT(4), .AW_ROM(AW_W2)
    ) u_l2 (
        .clk(clk), .rst_n(rst_n),
        .in_valid(l1_ov), .in_ready(l1_or), .in_data(l1_od[7:0]),
        .out_valid(l2_ov), .out_ready(l2_or), .out_data(l2_od)
    );

    silex_argmax #(.NOUT(NOUT), .AW(4)) u_amax (
        .clk(clk), .rst_n(rst_n),
        .in_valid(l2_ov), .in_ready(l2_or), .in_data(l2_od),
        .out_valid(class_valid), .out_ready(1'b1), .out_class(class_out)
    );

    silex_bist #(
        .NIN(NIN), .AW(AW_IN), .GFILE("golden.hex"), .EXPECT(GOLDEN_CLASS)
    ) u_bist (
        .clk(clk), .rst_n(rst_n),
        .g_valid(g_valid), .g_ready(~ready & l1_in_ready), .g_data(g_data),
        .res_valid(class_valid), .res_class(class_out),
        .ready(ready), .fault(fault)
    );
endmodule
