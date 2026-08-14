#!/usr/bin/env python3
"""Train and freeze the SILEX-1C CNN model.

Network (14x14 uint8 input):
  conv 3x3, 8 filters, stride 2, valid  -> 6x6x8, ReLU, requant to uint8
  flatten HWC (oy, ox, oc)              -> 288
  dense 288 -> 48, ReLU, requant to uint8
  dense 48  -> 10 logits, argmax

Same integer arithmetic contract as the dense layers (int8 weights, uint8
activations, int32 accumulators, round-half-up requant). Conv weight ROM
layout: w[oc*9 + ky*3 + kx]; the RTL sliding-window FSM and the im2col
reference below must agree on this and on HWC output order.

Outputs to build/: model_cnn.npz, wc/bc/w1c/b1c/w2c/b2c hex ROMs,
golden_cnn.hex + golden_cnn_class.txt, test vectors shared with MLP flow.
"""
import numpy as np
import os
import struct

RNG = np.random.default_rng(7)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
BUILD = os.path.join(ROOT, "build")
os.makedirs(BUILD, exist_ok=True)

HIN, K, S, C = 14, 3, 2, 8
HOUT = (HIN - K) // S + 1            # 6
NFLAT = HOUT * HOUT * C              # 288
H1, NOUT = 48, 10


def load_idx_images(path):
    with open(path, "rb") as f:
        magic, n, r, c = struct.unpack(">IIII", f.read(16))
        assert magic == 2051
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r, c)


def load_idx_labels(path):
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049
        return np.frombuffer(f.read(), dtype=np.uint8)


def pool14(imgs):
    x = imgs.reshape(-1, 14, 2, 14, 2).astype(np.uint16)
    return (x.sum(axis=(2, 4)) // 4).astype(np.uint8)   # (N,14,14)


def im2col(x):
    """(N,14,14) -> (N, 36, 9): sliding 3x3 windows, stride 2, row-major
    (oy,ox) window order, (ky,kx) element order."""
    n = x.shape[0]
    cols = np.empty((n, HOUT * HOUT, K * K), dtype=x.dtype)
    for oy in range(HOUT):
        for ox in range(HOUT):
            w = x[:, oy * S:oy * S + K, ox * S:ox * S + K]
            cols[:, oy * HOUT + ox, :] = w.reshape(n, K * K)
    return cols


def train_float(ctr, ytr, cte, yte):
    """conv-as-dense on im2col + 2 dense layers, SGD momentum."""
    wc = RNG.normal(0, np.sqrt(2.0 / (K * K)), (K * K, C)).astype(np.float32)
    bc = np.zeros(C, np.float32)
    w1 = RNG.normal(0, np.sqrt(2.0 / NFLAT), (NFLAT, H1)).astype(np.float32)
    b1 = np.zeros(H1, np.float32)
    w2 = RNG.normal(0, np.sqrt(2.0 / H1), (H1, NOUT)).astype(np.float32)
    b2 = np.zeros(NOUT, np.float32)
    vel = [np.zeros_like(p) for p in (wc, bc, w1, b1, w2, b2)]
    lr, mom, bs = 0.06, 0.9, 128
    onehot = np.eye(NOUT, dtype=np.float32)[ytr]

    def fwd(cols):
        h0p = cols @ wc + bc                      # (B,36,8)
        h0 = np.maximum(h0p, 0)
        flat = h0.reshape(len(cols), NFLAT)       # HWC flatten
        h1p = flat @ w1 + b1
        h1 = np.maximum(h1p, 0)
        return h0p, flat, h1p, h1, h1 @ w2 + b2

    for epoch in range(15):
        idx = RNG.permutation(len(ctr))
        for k in range(0, len(ctr), bs):
            j = idx[k:k + bs]
            cols, t = ctr[j], onehot[j]
            h0p, flat, h1p, h1, z = fwd(cols)
            z -= z.max(axis=1, keepdims=True)
            p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
            dz = (p - t) / len(j)
            dw2 = h1.T @ dz; db2 = dz.sum(0)
            dh1 = dz @ w2.T; dh1[h1p <= 0] = 0
            dw1 = flat.T @ dh1; db1 = dh1.sum(0)
            dflat = (dh1 @ w1.T).reshape(len(j), HOUT * HOUT, C)
            dflat[h0p <= 0] = 0
            dwc = np.einsum("bik,bic->kc", cols, dflat)
            dbc = dflat.sum((0, 1))
            for v, g, p_ in zip(vel, (dwc, dbc, dw1, db1, dw2, db2),
                                (wc, bc, w1, b1, w2, b2)):
                v *= mom; v -= lr * g; p_ += v
        z = fwd(cte)[4]
        acc = float((np.argmax(z, 1) == yte).mean())
        print(f"epoch {epoch:2d}  float test acc {acc:.4f}")
    return (wc, bc, w1, b1, w2, b2), acc


def quantize(params, ccal):
    wcf, bcf, w1f, b1f, w2f, b2f = params
    s_x = 1.0 / 255.0
    SH = 24

    def qw(wf):
        s = float(np.max(np.abs(wf))) / 127.0
        return np.clip(np.round(wf / s), -127, 127).astype(np.int8), s

    wc, s_wc = qw(wcf)
    bc = np.round(bcf / (s_x * s_wc)).astype(np.int32)
    h0f = np.maximum(ccal.astype(np.float32) * s_x @ wcf + bcf, 0)
    s_ac = float(np.percentile(h0f, 99.99)) / 255.0
    Mc = int(round(s_x * s_wc / s_ac * (1 << SH)))

    w1, s_w1 = qw(w1f)
    b1 = np.round(b1f / (s_ac * s_w1)).astype(np.int32)
    flatf = np.maximum(h0f, 0).reshape(len(ccal), NFLAT)
    h1f = np.maximum(flatf @ w1f + b1f, 0)
    s_a1 = float(np.percentile(h1f, 99.99)) / 255.0
    M1 = int(round(s_ac * s_w1 / s_a1 * (1 << SH)))

    w2, s_w2 = qw(w2f)
    b2 = np.round(b2f / (s_a1 * s_w2)).astype(np.int32)
    return dict(wc=wc, bc=bc, Mc=Mc, w1=w1, b1=b1, M1=M1,
                w2=w2, b2=b2, SH=SH)


def rq(acc, M, SH):
    acc = np.maximum(acc, 0)
    return np.clip((acc * M + (1 << (SH - 1))) >> SH, 0, 255)


def int_forward(q, cols_u8):
    """Bit-exact integer reference = RTL golden model (conv via im2col)."""
    h0 = cols_u8.astype(np.int64) @ q["wc"].astype(np.int64) + q["bc"]
    a0 = rq(h0, q["Mc"], q["SH"]).reshape(len(cols_u8), NFLAT)
    h1 = a0 @ q["w1"].astype(np.int64) + q["b1"]
    a1 = rq(h1, q["M1"], q["SH"])
    logits = a1 @ q["w2"].astype(np.int64) + q["b2"]
    return np.argmax(logits, axis=1)


def write_hex8(path, arr):
    with open(path, "w") as f:
        for v in np.asarray(arr).astype(np.int64).ravel():
            f.write(f"{int(v) & 0xFF:02x}\n")


def write_hex32(path, arr):
    with open(path, "w") as f:
        for v in np.asarray(arr).astype(np.int64).ravel():
            f.write(f"{int(v) & 0xFFFFFFFF:08x}\n")


def main():
    xtr = pool14(load_idx_images(os.path.join(DATA, "train-images-idx3-ubyte")))
    ytr = load_idx_labels(os.path.join(DATA, "train-labels-idx1-ubyte"))
    xte = pool14(load_idx_images(os.path.join(DATA, "t10k-images-idx3-ubyte")))
    yte = load_idx_labels(os.path.join(DATA, "t10k-labels-idx1-ubyte"))
    ctr, cte = im2col(xtr), im2col(xte)

    params, facc = train_float(ctr.astype(np.float32) / 255.0, ytr,
                               cte.astype(np.float32) / 255.0, yte)
    q = quantize(params, ctr[:10000])
    pred = int_forward(q, cte)
    qacc = float((pred == yte).mean())
    print(f"float acc {facc:.4f} -> int8 acc {qacc:.4f}  "
          f"(Mc={q['Mc']}, M1={q['M1']}, SH={q['SH']})")

    np.savez(os.path.join(BUILD, "model_cnn.npz"), **q)
    write_hex8(os.path.join(BUILD, "wc.hex"), q["wc"].T)    # oc-major: oc*9+ky*3+kx
    write_hex32(os.path.join(BUILD, "bc.hex"), q["bc"])
    write_hex8(os.path.join(BUILD, "w1c.hex"), q["w1"].T)   # neuron-major
    write_hex32(os.path.join(BUILD, "b1c.hex"), q["b1"])
    write_hex8(os.path.join(BUILD, "w2c.hex"), q["w2"].T)
    write_hex32(os.path.join(BUILD, "b2c.hex"), q["b2"])

    write_hex8(os.path.join(BUILD, "golden_cnn.hex"), xte[0].reshape(-1))
    with open(os.path.join(BUILD, "golden_cnn_class.txt"), "w") as f:
        f.write(str(int(pred[0])))
    n = 1000
    xte[:n].reshape(n, -1).tofile(os.path.join(BUILD, "test_vectors.bin"))
    with open(os.path.join(BUILD, "test_expected_cnn.txt"), "w") as f:
        for c in pred[:n]:
            f.write(f"{int(c)}\n")
    print("frozen CNN model + ROMs + golden vectors written to build/")


if __name__ == "__main__":
    main()
