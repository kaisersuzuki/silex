# SILEX — a software-free AI inference chip

An AI chip that runs neural-network inference with **zero runtime software**:
no CPU, no instruction set, no firmware, no driver. The model is compiled
offline into hardware structure — weights become ROM contents, control
becomes hardwired per-layer finite-state machines, and the chip boots itself
through a hardware self-test before asserting a READY pin.

Two chip instances are implemented and verified bit-exact end to end:

| | SILEX-1D | SILEX-1C |
|---|---|---|
| Network | dense 196→48→10 | conv3×3 s2 ×8 → 288→48→10 |
| MNIST accuracy (chip logic) | 97.0% | 98.0% |
| RTL vs integer reference | 1000/1000 bit-exact | 1000/1000 bit-exact |
| Cycles per inference | ≈10,200 | ≈7,100 (P=4 MAC lanes) |

Silicon proof points: 23.5k cells (generic synthesis), iCE40 HX8K bitstream
at 74% utilization / 42 MHz (yosys + nextpnr + icepack), and gate-level
netlist simulation passing BIST plus 20/20 test vectors.

## Documents

- [DESIGN.md](DESIGN.md) — architecture concept: why no-software is feasible,
  compute-in-memory vs digital tradeoff, Tier A (mask ROM) vs Tier B
  (one-time NVM), risks, precedents.
- [SPEC.md](SPEC.md) — as-built microarchitecture: arithmetic contract,
  FSM design, measured results, scaling path.

## Layout

```
rtl/     silex_layer.v (P-parallel dense engine), silex_conv.v,
         silex_argmax.v, silex_bist.v, silex_top.v, silex_top_cnn.v
tools/   train.py, train_cnn.py (pure-numpy training + integer freeze),
         silexc.py (model -> RTL parameter/ROM compiler)
sim/     tb.v, tb_cnn.v
```

## Reproduce

Requires: python3 + numpy, icarus-verilog, yosys, nextpnr-ice40, icestorm.
Fetch MNIST idx files into `data/` (e.g. from
`storage.googleapis.com/cvdf-datasets/mnist/`), then:

```
make train train-cnn   # train + freeze both integer models
make verify verify-cnn # RTL simulation, 1000 vectors each, bit-exact assert
make synth             # generic synthesis gate counts
make fpga              # place-and-route + bitstream (build/silex_hx8k.bin)
make gatesim           # gate-level netlist simulation
```
