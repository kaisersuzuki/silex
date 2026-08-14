// SILEX conv-layer engine: KxK convolution, stride S, valid padding,
// ReLU + requantize, HWC output order (oy, ox, oc).
//
// The full input frame is buffered on-chip (tiny at this scale; a line
// buffer is the drop-in replacement for larger frames). Control is a
// hardwired FSM with nested window counters — the sliding-window address
// generator IS the schedule. No code, no configuration.
//
// Weight ROM layout: w[oc*K*K + ky*K + kx]  (must match tools/train_cnn.py)
// Arithmetic: identical contract to silex_layer (int8 x uint8 -> int32,
// round-half-up requant, clamp to uint8).
module silex_conv #(
    parameter HIN  = 14,
    parameter WIN  = 14,
    parameter K    = 3,
    parameter S    = 2,
    parameter COUT = 8,
    parameter HOUT = 6,
    parameter WOUT = 6,
    parameter signed [31:0] M = 32'sd1,
    parameter SH = 1,
    parameter WFILE = "wc.hex",
    parameter BFILE = "bc.hex",
    parameter AW_IN = 8            // >= clog2(HIN*WIN)
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    output wire        in_ready,
    input  wire [7:0]  in_data,
    output reg         out_valid,
    input  wire        out_ready,
    output reg  [31:0] out_data
);
    reg [7:0]  wrom [0:COUT*K*K-1];
    reg [31:0] brom [0:COUT-1];
    initial begin
        $readmemh(WFILE, wrom);
        $readmemh(BFILE, brom);
    end

    reg [7:0] fbuf [0:HIN*WIN-1];   // full-frame buffer (SRAM in silicon)

    localparam S_LOAD = 2'd0, S_MAC = 2'd1, S_ACT = 2'd2, S_OUT = 2'd3;
    reg [1:0]         state;
    reg [AW_IN:0]     icnt;
    reg [3:0]         oy, ox, oc, ky, kx;
    reg signed [31:0] acc;

    assign in_ready = (state == S_LOAD);

    // sliding-window operand addressing (constant-coefficient multiplies)
    wire [AW_IN-1:0] row  = oy * S + ky;
    wire [AW_IN-1:0] col  = ox * S + kx;
    wire [AW_IN-1:0] addr = row * WIN + col;
    wire signed [8:0]  xin  = {1'b0, fbuf[addr]};
    wire signed [7:0]  wcur = wrom[oc * (K * K) + ky * K + kx];
    wire signed [16:0] prod = xin * wcur;

    wire signed [31:0] relu    = (acc < 0) ? 32'sd0 : acc;
    wire signed [63:0] scaled  = relu * M + (64'sd1 <<< (SH - 1));
    wire signed [63:0] shifted = scaled >>> SH;
    wire [31:0] q8 = (shifted < 0)        ? 32'd0   :
                     (shifted > 64'sd255) ? 32'd255 : shifted[31:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_LOAD; icnt <= 0;
            oy <= 0; ox <= 0; oc <= 0; ky <= 0; kx <= 0;
            acc <= 0; out_valid <= 1'b0; out_data <= 0;
        end else begin
            case (state)
                S_LOAD: begin
                    if (in_valid) begin
                        fbuf[icnt[AW_IN-1:0]] <= in_data;
                        if (icnt == HIN * WIN - 1) begin
                            icnt <= 0;
                            oy <= 0; ox <= 0; oc <= 0; ky <= 0; kx <= 0;
                            acc <= $signed(brom[0]);
                            state <= S_MAC;
                        end else
                            icnt <= icnt + 1;
                    end
                end
                S_MAC: begin
                    acc <= acc + prod;
                    if (kx == K - 1) begin
                        kx <= 0;
                        if (ky == K - 1) begin
                            ky <= 0;
                            state <= S_ACT;
                        end else
                            ky <= ky + 1;
                    end else
                        kx <= kx + 1;
                end
                S_ACT: begin
                    out_data  <= q8;
                    out_valid <= 1'b1;
                    state     <= S_OUT;
                end
                S_OUT: begin
                    if (out_ready) begin
                        out_valid <= 1'b0;
                        if (oc == COUT - 1) begin
                            oc <= 0;
                            if (ox == WOUT - 1) begin
                                ox <= 0;
                                if (oy == HOUT - 1)
                                    state <= S_LOAD;   // frame done
                                else begin
                                    oy <= oy + 1;
                                    acc <= $signed(brom[0]);
                                    state <= S_MAC;
                                end
                            end else begin
                                ox <= ox + 1;
                                acc <= $signed(brom[0]);
                                state <= S_MAC;
                            end
                        end else begin
                            oc <= oc + 1;
                            acc <= $signed(brom[oc + 1]);
                            state <= S_MAC;
                        end
                    end
                end
            endcase
        end
    end
endmodule
