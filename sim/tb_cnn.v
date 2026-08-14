// SILEX-1D testbench: boot (BIST) then stream 1000 MNIST test images,
// log the chip's class decisions for bit-exact comparison against the
// integer reference model.
//
// TB drives and samples on negedge (mid-cycle, signals stable); the DUT
// clocks on posedge. If in_ready is high at a negedge while we present a
// byte, that byte is consumed at the following posedge.
`timescale 1ns/1ps
module tb;
    localparam NIN = 196;
    localparam NVEC = 1000;

    reg clk = 0, rst_n = 0;
    reg        in_valid = 0;
    reg  [7:0] in_data = 0;
    wire       in_ready, class_valid, ready, fault;
    wire [3:0] class_out;

    silex_top_cnn dut (
        .clk(clk), .rst_n(rst_n),
        .in_valid(in_valid), .in_ready(in_ready), .in_data(in_data),
        .class_valid(class_valid), .class_out(class_out),
        .ready(ready), .fault(fault)
    );

    always #5 clk = ~clk;   // 100 MHz

    reg [7:0] tv [0:NIN*NVEC-1];
    initial $readmemh("tv.hex", tv);

    integer f, i, j, boot_cycles, got;
    initial begin
        f = $fopen("results_cnn.txt", "w");
        rst_n = 0; repeat (4) @(posedge clk);
        rst_n = 1;

        // ---- hardware boot: wait for BIST verdict ----
        boot_cycles = 0;
        while (!ready && !fault) begin
            @(posedge clk); boot_cycles = boot_cycles + 1;
        end
        if (fault) begin
            $display("FAULT asserted during BIST -- chip offline");
            $fclose(f); $finish;
        end
        $display("READY after %0d cycles (BIST passed)", boot_cycles);

        // ---- stream test set ----
        for (i = 0; i < NVEC; i = i + 1) begin
            j = 0;
            while (j < NIN) begin
                @(negedge clk);
                in_valid = 1'b1;
                in_data  = tv[i * NIN + j];
                if (in_ready) j = j + 1;  // consumed at next posedge
            end
            @(negedge clk);
            in_valid = 1'b0;

            got = 0;
            while (!got) begin
                @(negedge clk);
                if (class_valid) begin
                    $fdisplay(f, "%0d", class_out);
                    got = 1;
                end
            end
            if (i % 100 == 99)
                $display("  %0d / %0d inferences", i + 1, NVEC);
        end
        $display("%0d inferences done", NVEC);
        $fclose(f);
        $finish;
    end

    initial begin
        #200_000_000;
        $display("TIMEOUT");
        $fclose(f);
        $finish;
    end
endmodule
