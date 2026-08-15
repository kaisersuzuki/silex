# SILEX-1D Microarchitecture Specification

Implemented and verified instance of the SILEX architecture (see DESIGN.md):
a digit-classification inference chip with **zero runtime software** — no CPU,
no instruction set, no firmware. All behavior is gate structure plus ROM
contents. This document describes the design as built in `rtl/`.

## 1. Chosen approach and why

| Decision | Choice | Rationale |
|---|---|---|
| Compute style | Digital fixed-point dataflow | Analog CIM offers better TOPS/W but requires foundry hard IP (ReRAM/eFlash macros) that cannot be designed from scratch at RTL; digital is fully self-contained, synthesizable, and bit-exact verifiable. Best buildable approach. |
| Control | One hardwired FSM per layer, elastic valid/ready handshakes | Eliminates instruction fetch entirely; schedule is the state graph. Proven discipline (systolic/dataflow, hls4ml-class designs). |
| Weights | Mask ROM (`$readmemh` models it in sim) | Physical state, not data loaded by software. Tier B swap-in: one-time-programmable NVM behind the same read port. |
| Numerics | int8 weights, uint8 activations, int32 accumulators, fixed-point requantization (integer multiply + arithmetic shift) | Industry-standard int8 inference; measured **zero accuracy loss** vs float here (96.97% → 97.02%). |
| Boot | Hardware BIST with golden vector in ROM | Chip proves itself functional and self-enables. No host involvement. |
| Model | MLP 196→48→10 on 14×14 avg-pooled MNIST | Small enough to simulate the full silicon behavior end to end; the layer engine is size-parameterized, so scaling is a parameter change plus more ROM. |

## 2. Frozen network and arithmetic contract

```
input : 196 × uint8                     (14×14 image, raw pixels)
L1    : acc32[n] = b1[n] + Σ_i  w1[n,i]·x[i]      (int8 × uint8 MACs)
        relu    : acc32 = max(acc32, 0)
        requant : a8 = clamp((acc32·M1 + 2^(SH1-1)) >> SH1, 0, 255)
L2    : logit32[c] = b2[c] + Σ_n  w2[c,n]·a8[n]
out   : class = argmax_c logit32[c]     (ties → lowest index)
```

M1 = 31444, SH1 = 24 (computed at freeze time from calibrated activation
scale; round-half-up). Biases are pre-scaled into accumulator units at freeze
time, so the datapath contains **no floating point and no division**.

The exact same arithmetic is implemented twice: in `tools/train.py`
(`int_forward`, the golden reference) and in `rtl/silex_layer.v`. Verified
bit-exact over 1000 vectors: 1000/1000 identical decisions.

## 3. Block-level implementation

### 3.1 `silex_layer` — dense-layer engine
- ROMs: weight ROM (neuron-major, flat-addressed) + bias ROM.
- SRAM: one input vector buffer (NIN × 8b).
- Datapath: one signed 9×8-bit multiplier, 32-bit accumulator, ReLU mux,
  32×32→64 requant multiplier + rounding shifter + clamp.
- FSM (4 states):

```
        all NIN bytes in            NIN MACs done
  LOAD ───────────────────▶ MAC ───────────────▶ ACT
   ▲                         ▲                    │ result latched
   │ last neuron emitted     │ next neuron        ▼
   └──────────────────────── OUT ◀────────────────┘
                              (waits out_ready — elastic)
```

- Parameters: `NIN, NOUT, REQUANT, M, SH, WFILE, BFILE` — the layer is
  model-agnostic; `silexc` specializes instances.
- P=1 MAC per cycle as built. Parallelism P is a design-time knob: widen to P
  multipliers and cut MAC-state cycles by P (area↔latency trade, no control
  change).

### 3.2 `silex_argmax`
Streams NOUT signed logits, keeps running max and index (strict `>` matches
numpy tie-breaking), emits 4-bit class with a valid pulse.

### 3.3 `silex_bist` — hardware boot
Reset → BIST owns the pipeline input mux → streams the golden image from ROM
→ compares resulting class to the design-time expected class → asserts
`READY` (pass, external port unmuxed) or latches `FAULT` (chip refuses
input). Verdict is reached in 10,201 cycles.

### 3.4 `silex_top`
BIST/external input mux → L1 (196→48, requant) → L2 (48→10, raw logits) →
argmax. Pins: byte-stream in (valid/ready), `class_out[3:0]` + `class_valid`,
`READY`, `FAULT`, clock, reset. Clock can come from an on-die RC oscillator;
nothing about the design needs a precise frequency.

## 4. Measured behavior (RTL simulation, Icarus Verilog)

Two chip instances, same architecture:

| Metric | SILEX-1D (MLP) | SILEX-1C (CNN) |
|---|---|---|
| Network | 196→48→10 dense | conv3x3 s2 ×8 → 288→48→10 |
| Boot (reset → READY, full BIST inference) | 10,259 cycles | 7,115 cycles |
| Cycles per inference | ≈ 10,200 | ≈ 7,100 (P=4 dense lanes) |
| Test-set agreement with integer reference | 1000/1000 bit-exact | 1000/1000 bit-exact |
| MNIST accuracy (chip logic) | 97.0% | **98.0%** |
| Runtime software required | none | none |

Weight/activation SRAM reads are synchronous (one-cycle latency, matching
real ROM/SRAM macros and FPGA BRAM); the MAC pipeline hides the latency with
a registered-operand stage.

## 4b. Silicon results (synthesis and FPGA implementation)

| Metric | Value |
|---|---|
| Generic synthesis (yosys `synth`) | **23,542 cells** total, ≈2,200 FFs — weights-as-logic included |
| iCE40 HX8K place-and-route | 5,725 / 7,680 LCs (74%), 20 / 32 BRAMs (62%) |
| Timing (nextpnr) | fmax ≈ 42 MHz → ≈ 4,100 inferences/s |
| Bitstream | `build/silex_hx8k.bin` (135 KB, icepack) |
| Gate-level netlist simulation | BIST passed, 20/20 test vectors bit-exact |

Verification chain, every link bit-exact: float model → frozen integer model
→ RTL simulation (1000/1000) → synthesized iCE40 netlist (gate-level sim,
20/20 + BIST) → bitstream.

## 4c. Sky130 ASIC implementation (OpenLane 2, signoff-clean GDSII)

Full-size SILEX-1D failed detailed routing on sky130: 9.4 KB of weight ROM
synthesized to mux logic exceeds what five metal layers can route (three
runs: congestion at global route twice, then a 60k→1.4k violation stall in
detailed route). The industry-standard fix is a ROM macro (OpenRAM/DFFRAM);
adopting one is future work. To close the flow end to end, **SILEX-1S** was
frozen: same architecture, BIST, and arithmetic, sized 49→24→10 (1.4 KB ROM;
7×7 avg-pooled MNIST, 94.2% int8 accuracy, 1000/1000 bit-exact RTL, boots in
1,568 cycles).

| SILEX-1S signoff metric (sky130 HD, 50 MHz target) | Value |
|---|---|
| Magic DRC / KLayout DRC | **0 / 0** |
| LVS (netgen) | **0 errors, fully matched** |
| Routing DRC convergence | 5,916 → 0 in 7 iterations |
| Setup / hold worst slack | +7.05 ns / +0.107 ns (fmax ≈ 77 MHz) |
| Die area / std cells | 0.394 mm² / 14,981 |
| Estimated power | ≈ 4.6 mW |
| Antenna violations | 26 nets (100 diodes auto-inserted; MPW-acceptable) |
| Output | `asic/silex_1s_sky130.gds` |

Flow: `make asic-small` (OpenLane 2.3.10, Docker backend, config in
`asic/config.json` — 20% core utilization, 25% placement density,
GRT_ADJUSTMENT 0.05, AREA 3 synthesis; settings chosen during the SILEX-1D
congestion fight and kept).

## 5. Resource estimate (synthesis-ready structure)

- Weight ROM: 196·48 + 48·10 = 9,888 bytes; bias ROM 232 bytes.
- SRAM: 196 + 48 bytes of activation buffering.
- Multipliers: 2 small (MAC) + 1 requant per layer instance.
- Order 10–30k gates + ROM macros — trivially fits any process; a 130 nm
  shuttle run or an iCE40/Artix FPGA both hold it with room to spare.

## 6. Scaling path — status

**Future work (tracked in [issue #1](https://github.com/kaisersuzuki/silex/issues/1)):**
replace SILEX-1D's synthesized weight ROM with a hard memory macro
(OpenRAM ROM / DFFRAM / sky130 SRAM + boot-copy) so the full-size chip
routes on sky130. The layer engine's synchronous one-cycle read port already
matches a macro interface, so no control-path change is expected.

1. **Conv layers** — DONE: `silex_conv` (sliding-window FSM over a frame
   buffer, HWC output order), used by SILEX-1C. Line buffer is the drop-in
   replacement for frames too large to buffer whole.
2. **P-parallel MACs** — DONE: `silex_layer` parameter `P` (SILEX-1C uses
   P=4); integer addition order is irrelevant, so any P is bit-identical.
3. **More layers**: instantiate more stages; handshakes compose.
4. **Capacity**: swap mask ROM for one-time-programmable NVM (Tier B) —
   same synchronous read interface, field-updatable weights, still zero
   runtime software.

## 7. Reproduce

```
make train train-cnn   # train + freeze both models (data/ already fetched)
make verify            # SILEX-1D: RTL sim, 1000 vectors, bit-exact assert
make verify-cnn        # SILEX-1C: same
make synth             # generic synthesis, gate-count stats
make fpga              # yosys + nextpnr + icepack -> build/silex_hx8k.bin
make gatesim           # gate-level netlist sim (BIST + 20 vectors)
```
