#!/usr/bin/env python
"""Scan-vs-serial equality + constructed-ceiling checks for lab/monoid.py.

PLAN2 §5: "Write a serial reference implementation and assert equality first —
off-by-one and left/right composition-order bugs here are silent and will look
like 'the architecture doesn't work.'"  This file is that assertion.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from monoid import (  # noqa: E402
    FAMILIES,
    MonoidALU,
    constrain,
    digits_to_int,
    int_to_digits,
    prefix_hop,
    prefix_scan,
    prefix_serial,
    scan_depth,
)


def check_prefix(device, dtype=torch.float32) -> int:
    fails = 0
    print("== prefix-product equality (serial reference vs log-depth scan vs HOP) ==")
    for d in (3, 8, 16, 32):
        for K in (1, 2, 3, 5, 6, 7, 8, 9, 12, 16, 17):
            # (a) permutation matrices: products are exactly representable, so
            #     equality here must be BIT-EXACT, not approximate.
            perm = torch.stack(
                [torch.eye(d, device=device)[torch.randperm(d, device=device)] for _ in range(4 * K)]
            ).view(4, K, d, d)
            ser, sc = prefix_serial(perm), prefix_scan(perm)
            exact = torch.equal(ser, sc)
            # (b) random dense, fp32: identical up to float re-association
            M = torch.randn(4, K, d, d, device=device, dtype=dtype) / math.sqrt(d)
            s2, c2 = prefix_serial(M), prefix_scan(M)
            err = (s2 - c2).abs().max().item()
            ok = exact and err < 2e-4
            if not ok:
                fails += 1
            print(
                f"  d={d:>2} K={K:>2}  perm_bit_exact={exact}  dense_max_abs_err={err:.2e}"
                f"  depth serial={scan_depth('serial', K)} scan={scan_depth('scan', K)}"
                f"  {'OK' if ok else 'FAIL'}"
            )
    # (c) torch._higher_order_ops.associative_scan
    try:
        M = torch.randn(4, 8, 8, 8, device=device) / math.sqrt(8)
        h = prefix_hop(M)
        err = (prefix_serial(M) - h).abs().max().item()
        print(f"  associative_scan HOP: max_abs_err vs serial = {err:.2e} "
              f"{'OK' if err < 2e-4 else 'FAIL'}")
        if err >= 2e-4:
            fails += 1
    except Exception as exc:  # pragma: no cover
        print(f"  associative_scan HOP: UNAVAILABLE ({type(exc).__name__}: {exc})")
    return fails


def check_families(device) -> int:
    fails = 0
    print("\n== constraint families: shape/stochasticity/stability of prefix products ==")
    for fam in FAMILIES:
        logits = torch.randn(2, 12, 16, 16, device=device)
        M = constrain(logits, fam)
        P = prefix_scan(M)
        ser = prefix_serial(M)
        err = (P - ser).abs().max().item()
        colsum = M.sum(-2).mean().item()
        pnorm = P[:, -1].abs().max().item()
        print(f"  {fam:<11} col_sum={colsum:.4f} max|P_last|={pnorm:.3e} "
              f"scan_vs_serial={err:.2e} {'OK' if err < 5e-3 else 'FAIL'}")
        if not math.isfinite(pnorm) or err >= 5e-3:
            fails += 1
    return fails


def check_construct(device, moduli, d, family, impl, verbose=True) -> int:
    print(f"\n== constructed ceiling (DIAGNOSTIC ORACLE) d={d} family={family} impl={impl} ==")
    fails = 0
    for modulus in moduli:
        S = len(str(modulus))
        alu = MonoidALU(S, d=d, family=family, impl=impl).to(device)
        alu.construct_()
        units = [x for x in range(1, modulus) if math.gcd(x, modulus) == 1]
        units = units[:2048]
        x = int_to_digits(units, alu.L, device)
        n = int_to_digits([modulus] * len(units), alu.L, device)
        with torch.no_grad():
            soft = alu(x, n, hard=False)
            hard = alu(x, n, hard=True)
        want = torch.tensor([(u * u) % modulus for u in units], dtype=torch.float64, device=device)
        got_s = digits_to_int(soft)
        got_h = digits_to_int(hard)
        acc_s = (got_s == want).float().mean().item()
        acc_h = (got_h == want).float().mean().item()
        ok = acc_s == 1.0 and acc_h == 1.0
        if not ok:
            fails += 1
        if verbose:
            print(
                f"  N={modulus:<7} S={S} L={alu.L} units={len(units):<5} "
                f"soft_exact={acc_s:.4f} hard_exact={acc_h:.4f} "
                f"op_depth={alu.op_depth} graph_depth={alu.graph_depth} "
                f"params={sum(p.numel() for p in alu.parameters()):,} {'OK' if ok else 'FAIL'}"
            )
    return fails


def check_bf16(device) -> int:
    """Manifests are bf16+amp; PLAN2 §5 says run the prefix products in fp32."""
    print("\n== bf16 autocast: fp32 scan interior ==")
    modulus, S = 323, 3
    alu = MonoidALU(S, d=16, family="colsoftmax", impl="scan").to(device)
    alu.construct_()
    units = [x for x in range(1, modulus) if math.gcd(x, modulus) == 1]
    x = int_to_digits(units, alu.L, device)
    n = int_to_digits([modulus] * len(units), alu.L, device)
    want = torch.tensor([(u * u) % modulus for u in units], dtype=torch.float64, device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        got = digits_to_int(alu(x, n, hard=False))
    acc = (got == want).float().mean().item()
    print(f"  N=323 under bf16 autocast: exact={acc:.4f} {'OK' if acc == 1.0 else 'FAIL'}")
    return 0 if acc == 1.0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--moduli", type=int, nargs="+", default=[323, 899, 2021, 10403])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)

    t0 = time.time()
    fails = check_prefix(device)
    fails += check_families(device)
    for d in ([16] if args.quick else [8, 16, 32]):
        fails += check_construct(device, args.moduli, d, "colsoftmax", "scan")
    # the same tables under the serial prefix must give the same answers
    fails += check_construct(device, args.moduli[:2], 16, "colsoftmax", "serial")
    fails += check_construct(device, args.moduli[:2], 16, "sthard", "scan")
    fails += check_bf16(device)
    print(f"\n{'ALL CHECKS PASSED' if fails == 0 else f'{fails} CHECK(S) FAILED'} "
          f"({time.time() - t0:.1f}s)")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
