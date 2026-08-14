RTL     = rtl/silex_layer.v rtl/silex_argmax.v rtl/silex_bist.v rtl/silex_top.v
RTL_CNN = rtl/silex_layer.v rtl/silex_conv.v rtl/silex_argmax.v rtl/silex_bist.v rtl/silex_top_cnn.v

.PHONY: all train train-cnn compile sim verify sim-cnn verify-cnn synth fpga gatesim clean

all: verify verify-cnn

train:
	python3 tools/train.py

train-cnn:
	python3 tools/train_cnn.py

build/silex_params.vh: build/model.npz
	python3 tools/silexc.py

build/silex_sim: build/silex_params.vh $(RTL) sim/tb.v
	cd build && iverilog -g2005 -I. -o silex_sim $(addprefix ../,$(RTL)) ../sim/tb.v

build/silex_cnn_sim: build/silex_params.vh $(RTL_CNN) sim/tb_cnn.v
	cd build && iverilog -g2005 -I. -o silex_cnn_sim $(addprefix ../,$(RTL_CNN)) ../sim/tb_cnn.v

compile: build/silex_sim build/silex_cnn_sim

sim: build/silex_sim
	cd build && vvp silex_sim

verify: sim
	python3 -c "\
	rtl = [int(l) for l in open('build/results.txt')]; \
	ref = [int(l) for l in open('build/test_expected.txt')]; \
	m = sum(a != b for a, b in zip(rtl, ref)); \
	print(f'MLP bit-exact: {len(rtl)-m}/{len(rtl)} match'); \
	assert m == 0"

sim-cnn: build/silex_cnn_sim
	cd build && vvp silex_cnn_sim

verify-cnn: sim-cnn
	python3 -c "\
	rtl = [int(l) for l in open('build/results_cnn.txt')]; \
	ref = [int(l) for l in open('build/test_expected_cnn.txt')]; \
	m = sum(a != b for a, b in zip(rtl, ref)); \
	print(f'CNN bit-exact: {len(rtl)-m}/{len(rtl)} match'); \
	assert m == 0"

# generic (ASIC-style) synthesis statistics
synth: build/silex_params.vh
	cd build && head -200 w1.hex > w.hex && head -48 b1.hex > b.hex && \
	yosys -p "read_verilog -I. $(addprefix ../,$(RTL)); hierarchy -top silex_top; synth; stat"

# iCE40 HX8K bitstream (SILEX-1D)
fpga: build/silex_params.vh
	cd build && head -200 w1.hex > w.hex && head -48 b1.hex > b.hex && \
	yosys -p "read_verilog -I. $(addprefix ../,$(RTL)); synth_ice40 -top silex_top -json silex_up5k.json" && \
	nextpnr-ice40 --hx8k --package ct256 --json silex_up5k.json --asc silex_hx8k.asc --pcf-allow-unconstrained && \
	icepack silex_hx8k.asc silex_hx8k.bin && ls -la silex_hx8k.bin

# gate-level simulation of the synthesized netlist (20 vectors + BIST)
gatesim:
	cd build && yosys -q -p "read_json silex_up5k.json; write_verilog -noattr silex_gate.v" && \
	sed 's/NVEC = 1000/NVEC = 20/' ../sim/tb.v > tb_gate.v && \
	iverilog -g2012 -o silex_gate_sim silex_gate.v tb_gate.v "$$(yosys-config --datdir)/ice40/cells_sim.v" && \
	vvp silex_gate_sim

clean:
	rm -f build/silex_sim build/silex_cnn_sim build/silex_gate_sim \
	      build/results.txt build/results_cnn.txt
