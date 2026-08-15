// SILEX-1D dense-layer engine.
//
// One physical pipeline stage = one network layer. Control is a four-state
// hardwired FSM; there is no instruction stream. Weights and biases are ROM
// contents fixed at design time ($readmemh stands in for the mask ROM).
//
// Arithmetic contract (must stay bit-exact with tools/train.py int_forward):
//   acc32   = b[n] + sum_i( int8(w[n*NIN+i]) * uint8(x[i]) )
//   REQUANT=1: relu -> a8 = clamp((acc32*M + 2^(SH-1)) >> SH, 0, 255), out = a8
//   REQUANT=0: out = acc32 (raw logits)
//
// Interface: elastic byte-stream in, word-stream out (valid/ready both sides).
// Cost: ceil(NIN/P)*NOUT MAC cycles per inference. P parallel MAC lanes are a
// design-time area/latency knob; integer addition order is irrelevant, so any
// P produces bit-identical results.

module silex_layer #(
    parameter NIN  = 196,
    parameter NOUT = 48,
    parameter P    = 1,
    parameter REQUANT = 1,
    parameter signed [31:0] M = 32'sd1,
    parameter SH = 1,
    parameter WFILE = "w.hex",
    parameter BFILE = "b.hex",
    parameter AW_IN  = 8,   // >= clog2(NIN)
    parameter AW_OUT = 6,   // >= clog2(NOUT)
    parameter AW_ROM = 14   // >= clog2(NIN*NOUT)
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
    // ---- ROMs (mask ROM in silicon; hex image in simulation) ----
    reg [7:0]  wrom [0:NIN*NOUT-1];
    reg [31:0] brom [0:NOUT-1];
    initial begin
        $readmemh(WFILE, wrom);
        $readmemh(BFILE, brom);
    end

    // ---- input vector buffer (single-port SRAM in silicon) ----
    reg [7:0] ibuf [0:NIN-1];

    localparam S_LOAD = 2'd0, S_MAC = 2'd1, S_ACT = 2'd2, S_OUT = 2'd3;
    reg [1:0]           state;
    reg [AW_IN:0]       icnt;   // input element index (steps by P)
    reg [AW_OUT:0]      ncnt;   // neuron index
    reg [AW_ROM:0]      wbase;  // current neuron's base address in weight ROM
    reg signed [31:0]   acc;

    assign in_ready = (state == S_LOAD);

    // Synchronous operand fetch (one-cycle ROM/SRAM read, BRAM/macro
    // compatible), then P parallel signed MAC lanes over the registered
    // operands. Lanes past the end of the vector (NIN not a multiple of P)
    // fetch zero weight and contribute nothing.
    reg [7:0]         xr [0:P-1];
    reg [7:0]         wr [0:P-1];
    reg               mac_v;   // registered operands are valid
    reg               drain;   // last chunk fetched; one accumulate left
    reg signed [31:0] psum;
    integer k;   // comb psum loop only
    integer j;   // clocked fetch loop only (separate variable: sharing one
                 // across two processes creates a multi-driven net in synthesis)
    always @* begin
        psum = 32'sd0;
        for (k = 0; k < P; k = k + 1)
            psum = psum + $signed({1'b0, xr[k]}) * $signed(wr[k]);
    end

    // ReLU + fixed-point requantize (round-half-up), clamp to uint8
    wire signed [31:0] relu    = (acc < 0) ? 32'sd0 : acc;
    wire signed [63:0] scaled  = relu * M + (64'sd1 <<< (SH - 1));
    wire signed [63:0] shifted = scaled >>> SH;
    wire [31:0] q8 = (shifted < 0)      ? 32'd0   :
                     (shifted > 64'sd255) ? 32'd255 : shifted[31:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_LOAD; icnt <= 0; ncnt <= 0; wbase <= 0;
            acc <= 0; out_valid <= 1'b0; out_data <= 0;
            mac_v <= 1'b0; drain <= 1'b0;
        end else begin
            case (state)
                S_LOAD: begin
                    if (in_valid) begin
                        ibuf[icnt[AW_IN-1:0]] <= in_data;
                        if (icnt == NIN - 1) begin
                            icnt <= 0; ncnt <= 0; wbase <= 0;
                            acc  <= $signed(brom[0]);
                            mac_v <= 1'b0; drain <= 1'b0;
                            state <= S_MAC;
                        end else
                            icnt <= icnt + 1;
                    end
                end
                S_MAC: begin
                    if (mac_v)
                        acc <= acc + psum;
                    if (drain) begin
                        mac_v <= 1'b0; drain <= 1'b0;
                        icnt  <= 0;
                        state <= S_ACT;
                    end else begin
                        for (j = 0; j < P; j = j + 1)
                            if (j == 0) begin
                                xr[0] <= ibuf[icnt[AW_IN-1:0]];
                                wr[0] <= wrom[wbase[AW_ROM-1:0] + icnt[AW_IN-1:0]];
                            end else begin
                                xr[j] <= (icnt + j < NIN)
                                         ? ibuf[icnt[AW_IN-1:0] + j] : 8'd0;
                                wr[j] <= (icnt + j < NIN)
                                         ? wrom[wbase[AW_ROM-1:0] + icnt[AW_IN-1:0] + j]
                                         : 8'd0;
                            end
                        mac_v <= 1'b1;
                        if (icnt + P >= NIN)
                            drain <= 1'b1;
                        else
                            icnt <= icnt + P;
                    end
                end
                S_ACT: begin
                    out_data  <= REQUANT ? q8 : acc;
                    out_valid <= 1'b1;
                    state     <= S_OUT;
                end
                S_OUT: begin
                    if (out_ready) begin
                        out_valid <= 1'b0;
                        if (ncnt == NOUT - 1)
                            state <= S_LOAD;
                        else begin
                            ncnt  <= ncnt + 1;
                            wbase <= wbase + NIN;
                            icnt  <= 0;
                            acc   <= $signed(brom[ncnt[AW_OUT-1:0] + 1]);
                            state <= S_MAC;
                        end
                    end
                end
            endcase
        end
    end
endmodule
