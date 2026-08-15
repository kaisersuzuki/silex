#!/usr/bin/env python3
"""Train and freeze SILEX-1S, the Sky130 test-chip instance.

Same architecture and arithmetic contract as SILEX-1D, sized so the weight
ROM (~1.4 KB) routes cleanly as synthesized logic on sky130's five metal
layers: 7x7 avg-pooled MNIST (49 inputs) -> 24 ReLU -> 10 logits.

Emits the same artifact set as train.py, into build/small/.
"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import (load_idx_images, load_idx_labels, train_float as _tf,  # noqa: E402
                   write_hex8, write_hex32, DATA)
import train as T  # noqa: E402

BUILD = os.path.join(os.path.dirname(DATA), "build", "small")
os.makedirs(BUILD, exist_ok=True)

# resize the shared training code's geometry
T.NIN, T.H = 49, 24
RNG = np.random.default_rng(3)
T.RNG = RNG


def pool7(imgs):
    """28x28 -> 7x7 by 4x4 mean, uint8."""
    x = imgs.reshape(-1, 7, 4, 7, 4).astype(np.uint32)
    return (x.sum(axis=(2, 4)) // 16).astype(np.uint8).reshape(-1, 49)


def main():
    xtr = pool7(load_idx_images(os.path.join(DATA, "train-images-idx3-ubyte")))
    ytr = load_idx_labels(os.path.join(DATA, "train-labels-idx1-ubyte"))
    xte = pool7(load_idx_images(os.path.join(DATA, "t10k-images-idx3-ubyte")))
    yte = load_idx_labels(os.path.join(DATA, "t10k-labels-idx1-ubyte"))

    w1f, b1f, w2f, b2f, facc = _tf(xtr.astype(np.float32) / 255.0, ytr,
                                   xte.astype(np.float32) / 255.0, yte)
    q = T.quantize(w1f, b1f, w2f, b2f, xtr[:10000])
    pred, _ = T.int_forward(q, xte)
    qacc = float((pred == yte).mean())
    print(f"SILEX-1S float acc {facc:.4f} -> int8 acc {qacc:.4f}")

    np.savez(os.path.join(BUILD, "model.npz"), **q)
    write_hex8(os.path.join(BUILD, "w1.hex"), q["w1"].T)
    write_hex32(os.path.join(BUILD, "b1.hex"), q["b1"])
    write_hex8(os.path.join(BUILD, "w2.hex"), q["w2"].T)
    write_hex32(os.path.join(BUILD, "b2.hex"), q["b2"])
    write_hex8(os.path.join(BUILD, "golden.hex"), xte[0].view(np.int8))
    with open(os.path.join(BUILD, "golden_class.txt"), "w") as f:
        f.write(str(int(pred[0])))
    n = 1000
    xte[:n].tofile(os.path.join(BUILD, "test_vectors.bin"))
    with open(os.path.join(BUILD, "test_expected.txt"), "w") as f:
        for c in pred[:n]:
            f.write(f"{int(c)}\n")
    print(f"SILEX-1S frozen to {BUILD}")


if __name__ == "__main__":
    main()
