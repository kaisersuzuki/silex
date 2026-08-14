# SILEX-1: A Software-Free AI Inference Chip

An ASIC that executes neural-network inference with **zero runtime software dependency**: no CPU, no instruction set, no firmware, no driver required for operation. Power on, stream data in, get results out.

---

## 1. Goal and honest scope

**Goal.** A chip whose entire inference behavior is defined by physical hardware state — logic structure and non-volatile device state — so that at runtime nothing fetches, decodes, or executes instructions.

**What "no software" can and cannot mean.** The model itself must originate somewhere: it is trained off-chip, then *compiled at design time* into hardware structure (or programmed once into on-die NVM). After that point the chip is autonomous. Changing the model means a new mask set (Tier A) or a one-time NVM reflash via JTAG (Tier B) — not software execution. This is the same sense in which a ROM-based arcade board or an analog filter "has no software."

**Non-goals.** Training on-chip, general programmability, multi-model hot-swap, OS integration beyond a dumb data port.

---

## 2. Two implementation tiers

| | Tier A: Mask-hardwired | Tier B: NVM-configured dataflow |
|---|---|---|
| Weights | Baked into ROM / metal masks | ReRAM or eFlash cells, programmed once |
| Model change | New tapeout | JTAG reflash (offline, one-time) |
| Control | Fully synthesized FSM per layer | FSM with NVM-held config registers |
| Best for | Ultra-cheap high-volume (keyword spotting, wake-word, anomaly detection) | Field-updatable edge inference |
| Software at runtime | None | None |

Both tiers share the same microarchitecture below. Tier B just replaces ROM with one-time-programmable NVM.

---

## 3. Top-level architecture

```
            ┌─────────────────────────────────────────────────────┐
            │                     SILEX-1 die                     │
            │                                                     │
 sensor ───▶│ ┌─────────┐  ┌──────────────────────┐  ┌─────────┐ │
 (MIPI/     │ │ Input   │  │  Layer Pipeline       │  │ Output  │ │──▶ result
  I2S/      │ │ Front   │─▶│  (N hardwired stages) │─▶│ Encoder │ │    (GPIO/
  SPI/      │ │ End     │  │                       │  │ + FIFO  │ │     SPI/
  LVDS)     │ └─────────┘  │  ┌────┐ ┌────┐ ┌────┐ │  └─────────┘ │     IRQ pin)
            │              │  │ L1 │▶│ L2 │▶│ …  │ │              │
            │              │  └────┘ └────┘ └────┘ │              │
            │              └──────────────────────┘               │
            │ ┌──────────────┐  ┌───────────────┐  ┌───────────┐  │
            │ │ Power-on     │  │ Clock/Reset   │  │ JTAG (Tier│  │
            │ │ Self-Init FSM│  │ Gen (on-die   │  │ B program │  │
            │ │ + BIST       │  │ RC osc, no    │  │ port only)│  │
            │ └──────────────┘  │ ext. crystal  │  └───────────┘  │
            │                   │ required)     │                 │
            │                   └───────────────┘                 │
            └─────────────────────────────────────────────────────┘
```

Boot sequence is pure hardware: power-on reset → bandgap + oscillator settle → self-init FSM loads NVM config into shadow registers (Tier B) or does nothing (Tier A) → built-in self-test walks a known test vector through the pipeline and compares against a stored golden CRC → `READY` pin asserts. No host involvement; the host (if any) is just a data producer/consumer.

---

## 4. Layer pipeline — the core

Each network layer is a **physical pipeline stage**, not a time-multiplexed program:

### 4.1 Compute: mixed-signal compute-in-memory (CIM)

- Weights live in **ReRAM crossbar macros** (Tier B) or **ROM-coefficient MAC arrays** (Tier A).
- A matrix-vector multiply is one analog operation: input activations drive DAC word-lines, currents sum on bit-lines per Kirchhoff's law, column ADCs read out partial sums. ~1–10 TOPS/W class efficiency at 4–8 bit precision, consistent with published CIM silicon.
- Digital fallback per macro: a bit-serial MAC array for layers needing exact arithmetic (e.g., final logits), selected at design time per layer.

### 4.2 Activation functions

Piecewise-linear approximations in **ROM lookup tables** (256-entry, 8-bit). ReLU degenerates to a sign-mux and costs nothing. No transcendental units, no microcode.

### 4.3 Sequencing without instructions

Each stage has a small **hardwired FSM** (typically 10–40 states, synthesized from the layer's loop nest at design time). The FSMs are chained with valid/ready handshakes — an elastic pipeline. There is no global program counter and no shared instruction memory; the "schedule" is the state graph itself. This is the classic systolic/dataflow discipline: control is *structure*, not *code*.

### 4.4 Inter-layer buffering

Double-buffered SRAM line buffers sized at design time from the model's activation footprints. Sizes are computed by the offline compiler (§6) and frozen into RTL.

---

## 5. Numerics, robustness, safety rails

- **Quantization:** weights 4–8 bit, activations 8 bit, 24-bit accumulators. Chosen offline with quantization-aware training; the chip only ever sees fixed-point.
- **Analog drift (Tier B):** on-die temperature sensor drives a hardwired compensation DAC per crossbar; periodic self-calibration against stored reference columns (spare crossbar columns holding known conductances). All in hardware FSMs.
- **Fault detection:** golden-vector BIST at boot plus a continuous CRC over the output FIFO; a `FAULT` pin asserts on mismatch. No software watchdog needed because there is nothing to hang — an FSM either handshakes or the fault timer (a counter, not code) trips.

---

## 6. The offline toolchain (design-time, not runtime)

Software exists — but only on the designer's workstation, never on or near the chip:

1. Train model (any framework) → quantize → freeze.
2. **Model-to-RTL compiler**: emits per-layer FSMs, buffer sizes, and either ROM weight images (Tier A) or an NVM programming file (Tier B). Practical base: an HLS or HDL-generator flow (e.g., a Chisel/Amaranth generator, or an hls4ml-style flow, both proven for fixed-model accelerators).
3. Standard ASIC backend: synthesis, place-and-route, signoff. CIM macros come in as hard IP.
4. Tier B only: at the factory (or field, via JTAG) burn weights into ReRAM. Chip never executes this file; it is device state, like blowing fuses.

---

## 7. Representative spec (Tier B, edge-vision class)

| Parameter | Value |
|---|---|
| Process | 22 nm CMOS + ReRAM back-end module |
| Model capacity | ~8 M parameters at 4-bit (≈4 MB NVM) |
| Throughput | 30 fps at 224×224×3 CNN, or always-on audio at <1 mW |
| Peak efficiency | ~5 TOPS/W (CIM layers) |
| Power | 10–150 mW depending on duty cycle |
| Interfaces | MIPI CSI-2 or I2S in; SPI/GPIO/IRQ out; JTAG (program/test only) |
| Boot time | <2 ms from power-on to READY |
| Runtime software required | None |

---

## 8. Key risks

1. **Analog CIM variability** — mitigated by per-column calibration, redundancy columns, and keeping precision-critical layers digital. Worst case: ship Tier A/digital-only variant at ~3× area.
2. **Model obsolescence** — the flip side of no software. Tier B's one-time reflash is the only escape hatch; pick it unless unit economics demand masks.
3. **Compiler correctness burden** — a bug ships in silicon. Mitigate with formal equivalence checking between the quantized reference model and the generated RTL (bit-exact for digital layers, bounded-error proofs for analog).
4. **Ecosystem friction** — no driver means integrators treat it like a sensor, not an accelerator. That is the intended mental model; document it that way.

---

## 9. Precedents proving feasibility

- **IBM NorthPole / TrueNorth** — fully on-chip weights, no external memory traffic, minimal runtime control.
- **Mythic AMP, IBM HERMES / analog-AI research chips** — flash/PCM compute-in-memory with weights as device state.
- **hls4ml FPGA triggers at CERN** — fixed models compiled to pure dataflow logic, nanosecond latency, no processor in the loop.
- Every mask-ROM-era fixed-function DSP — the "software-free" deployment model is old; applying it to modern NN workloads is the only new part.

---

## 10. Suggested build path

1. **Weeks 1–8:** model-to-RTL compiler MVP targeting a small CNN (MNIST/keyword-spotting scale); simulate.
2. **Weeks 8–16:** FPGA prototype of the full elastic pipeline incl. boot FSM and BIST — validates "no software" end to end (soft logic stands in for CIM).
3. **Then:** digital-only Tier A test chip on a shuttle run (~$20–50k for MPW); CIM macros in a second spin once the digital spine is proven.
