// SILEX-1C top level: CNN variant, still zero runtime software.
//
// conv 3x3 s2 x8 (ReLU, requant) -> dense 288->48 (P=4 MAC lanes, ReLU,
// requant) -> dense 48->10 (P=4, raw logits) -> argmax. Power-on BIST
// self-enables the chip exactly as in SILEX-1D.
module silex_top_cnn (
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
    `include "silex_cnn_params.vh"

    wire        g_valid, cv_in_ready;
    wire [7:0]  g_data;
    wire        mux_valid = ready ? (in_valid & ~fault) : g_valid;
    wire [7:0]  mux_data  = ready ? in_data  : g_data;
    assign in_ready = ready & ~fault & cv_in_ready;

    wire        cv_ov, cv_or;
    wire [31:0] cv_od;
    silex_conv #(
        .HIN(14), .WIN(14), .K(3), .S(2), .COUT(C_COUT),
        .HOUT(6), .WOUT(6), .M(C_MC), .SH(C_SH),
        .WFILE("wc.hex"), .BFILE("bc.hex"), .AW_IN(C_AW_IN)
    ) u_conv (
        .clk(clk), .rst_n(rst_n),
        .in_valid(mux_valid), .in_ready(cv_in_ready), .in_data(mux_data),
        .out_valid(cv_ov), .out_ready(cv_or), .out_data(cv_od)
    );

    wire        l1_ov, l1_or;
    wire [31:0] l1_od;
    silex_layer #(
        .NIN(C_NFLAT), .NOUT(C_H1), .P(4), .REQUANT(1), .M(C_M1), .SH(C_SH),
        .WFILE("w1c.hex"), .BFILE("b1c.hex"),
        .AW_IN(C_AW_FLAT), .AW_OUT(C_AW_H1), .AW_ROM(C_AW_W1)
    ) u_l1 (
        .clk(clk), .rst_n(rst_n),
        .in_valid(cv_ov), .in_ready(cv_or), .in_data(cv_od[7:0]),
        .out_valid(l1_ov), .out_ready(l1_or), .out_data(l1_od)
    );

    wire        l2_ov, l2_or;
    wire [31:0] l2_od;
    silex_layer #(
        .NIN(C_H1), .NOUT(C_NOUT), .P(4), .REQUANT(0), .M(32'sd1), .SH(1),
        .WFILE("w2c.hex"), .BFILE("b2c.hex"),
        .AW_IN(C_AW_H1), .AW_OUT(4), .AW_ROM(C_AW_W2)
    ) u_l2 (
        .clk(clk), .rst_n(rst_n),
        .in_valid(l1_ov), .in_ready(l1_or), .in_data(l1_od[7:0]),
        .out_valid(l2_ov), .out_ready(l2_or), .out_data(l2_od)
    );

    silex_argmax #(.NOUT(C_NOUT), .AW(4)) u_amax (
        .clk(clk), .rst_n(rst_n),
        .in_valid(l2_ov), .in_ready(l2_or), .in_data(l2_od),
        .out_valid(class_valid), .out_ready(1'b1), .out_class(class_out)
    );

    silex_bist #(
        .NIN(C_NIN), .AW(C_AW_IN), .GFILE("golden_cnn.hex"), .EXPECT(C_GOLDEN)
    ) u_bist (
        .clk(clk), .rst_n(rst_n),
        .g_valid(g_valid), .g_ready(~ready & cv_in_ready), .g_data(g_data),
        .res_valid(class_valid), .res_class(class_out),
        .ready(ready), .fault(fault)
    );
endmodule
