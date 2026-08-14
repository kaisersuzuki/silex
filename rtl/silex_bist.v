// SILEX-1D power-on built-in self-test.
//
// Pure-hardware boot: after reset the BIST owns the pipeline input, streams
// one golden image from ROM through the full network, and compares the
// resulting class against the design-time expectation. Pass -> READY pin
// asserts and the external port is unmuxed. Fail -> FAULT latches, chip
// stays offline. No host, no code.
module silex_bist #(
    parameter NIN = 196,
    parameter AW  = 8,
    parameter GFILE = "golden.hex",
    parameter [3:0] EXPECT = 4'd0
) (
    input  wire       clk,
    input  wire       rst_n,
    // to pipeline input mux
    output reg        g_valid,
    input  wire       g_ready,
    output wire [7:0] g_data,
    // observed pipeline result
    input  wire       res_valid,
    input  wire [3:0] res_class,
    // chip status pins
    output reg        ready,
    output reg        fault
);
    reg [7:0] grom [0:NIN-1];
    initial $readmemh(GFILE, grom);

    localparam B_FEED = 2'd0, B_WAIT = 2'd1, B_DONE = 2'd2;
    reg [1:0]  state;
    reg [AW:0] cnt;

    assign g_data = grom[cnt[AW-1:0]];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= B_FEED; cnt <= 0;
            g_valid <= 1'b1; ready <= 1'b0; fault <= 1'b0;
        end else begin
            case (state)
                B_FEED: begin
                    if (g_valid && g_ready) begin
                        if (cnt == NIN - 1) begin
                            g_valid <= 1'b0;
                            state   <= B_WAIT;
                        end else
                            cnt <= cnt + 1;
                    end
                end
                B_WAIT: begin
                    if (res_valid) begin
                        if (res_class == EXPECT) ready <= 1'b1;
                        else                     fault <= 1'b1;
                        state <= B_DONE;
                    end
                end
                B_DONE: ; // latched until next power cycle
            endcase
        end
    end
endmodule
