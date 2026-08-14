// SILEX-1D argmax unit: consumes NOUT signed 32-bit logits as a stream,
// emits the winning class index. Ties resolve to the lowest index
// (strict > comparison), matching numpy argmax.
module silex_argmax #(
    parameter NOUT = 10,
    parameter AW   = 4
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    output wire        in_ready,
    input  wire [31:0] in_data,
    output reg         out_valid,
    input  wire        out_ready,
    output reg  [3:0]  out_class
);
    reg [AW:0]        cnt;
    reg signed [31:0] best;
    reg [3:0]         best_i;
    reg               busy_out;

    assign in_ready = !busy_out;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cnt <= 0; best <= 0; best_i <= 0;
            out_valid <= 1'b0; out_class <= 0; busy_out <= 1'b0;
        end else begin
            if (out_valid && out_ready) begin
                out_valid <= 1'b0;
                busy_out  <= 1'b0;
            end
            if (in_valid && in_ready) begin
                if (cnt == 0 || $signed(in_data) > best) begin
                    best   <= in_data;
                    best_i <= cnt[3:0];
                end
                if (cnt == NOUT - 1) begin
                    cnt       <= 0;
                    out_class <= (cnt == 0 || $signed(in_data) > best)
                                 ? cnt[3:0] : best_i;
                    out_valid <= 1'b1;
                    busy_out  <= 1'b1;
                end else
                    cnt <= cnt + 1;
            end
        end
    end
endmodule
