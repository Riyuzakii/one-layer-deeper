#!/usr/bin/env python
"""Correctness gates for `lab/o1reduce.py` — run before believing any number.

Checks, in order:
  1. the Barrett bound the architecture relies on (r in [0, 2N)) over the whole
     operand range of the moduli actually used;
  2. `ColSum` + `CarryMonoid` reproduce exact multi-row addition;
  3. `LongDivRecip` reproduces floor(10^{2S}/N);
  4. the full constructed `O1ReduceALU` is exact, soft and argmax-hard,
     including under bf16 autocast;
  5. the reported learned-op depths are what the code actually does.

All of this is DIAGNOSTIC (it uses `construct_`); nothing here is a submission.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from o1reduce import (  # noqa: E402
    O1ReduceALU,
    digits_to_int,
    int_to_digits,
    true_mu,
)

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FAILED = []


def check(name, ok, extra=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' ' + extra) if extra else ''}", flush=True)
    if not ok:
        FAILED.append(name)


# --------------------------------------------------------------------------- #
def slots_for(moduli):
    return max(len(str(m)) for m in moduli)


def test_barrett_bound(moduli, S=None, tag=""):
    """r = p - floor(p*mu/10^{2S})*N must land in [0, 2N) for every operand.

    The bound needs p = x^2 < 10^{2S}, i.e. S >= the digit length of N; the
    first run of this gate caught exactly that (a 4-digit modulus at S=3 gave
    floor(r/N) up to 4).
    """
    S = S or slots_for(moduli)
    worst = 0
    for n in moduli:
        mu = (10 ** (2 * S)) // n
        for x in random.Random(0).sample(range(1, n), min(3000, n - 1)):
            p = x * x
            q = (p * mu) // (10 ** (2 * S))
            r = p - q * n
            assert r >= 0, (n, x, r)
            worst = max(worst, r // n)
    check(f"barrett bound S={S} {tag}", worst <= 1,
          f"(max floor(r/N) = {worst}; n_corr must exceed it)")
    return worst


# --------------------------------------------------------------------------- #
def test_longdiv(moduli, S=None):
    S = S or slots_for(moduli)
    m = O1ReduceALU(S, recip="div").to(DEV)
    m.construct_()
    n = int_to_digits(moduli, m.L, DEV)
    mu = m.recip(n)
    got = digits_to_int(mu).tolist()
    want = [(10 ** (2 * S)) // v for v in moduli]
    check(f"longdiv recip S={S}", got == want, f"({sum(a == b for a, b in zip(got, want))}/{len(want)})")


# --------------------------------------------------------------------------- #
def test_construct(moduli, recip, hard, S=None, amp=False, T=1, n_ops=400):
    S = S or slots_for(moduli)
    m = O1ReduceALU(S, recip=recip).to(DEV)
    m.construct_()
    rng = random.Random(1)
    pairs = []
    for n in moduli:
        units = [x for x in rng.sample(range(2, n), min(200, n - 2)) if math.gcd(x, n) == 1]
        pairs += [(n, x) for x in units[: max(1, n_ops // len(moduli))]]
    ns = [n for n, _ in pairs]
    xs = [x for _, x in pairs]
    ys = []
    for n, x in pairs:
        v = x
        for _ in range(T):
            v = v * v % n
        ys.append(v)
    xin = int_to_digits(xs, m.L, DEV)
    nin = int_to_digits(ns, m.L, DEV)
    if recip == "oracle":
        m.set_mu(true_mu(ns, S, m.Lmu, DEV))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        y = xin
        for _ in range(T):
            y = m(y, nin, hard=hard)
    got = digits_to_int(y[:, :S]).tolist()
    ok = sum(int(a) == b for a, b in zip(got, ys))
    tag = f"construct S={S} recip={recip} {'hard' if hard else 'soft'}{' amp' if amp else ''} T={T}"
    check(tag, ok == len(ys), f"({ok}/{len(ys)})")


# --------------------------------------------------------------------------- #
def test_depths():
    for S in (3, 7):
        m = O1ReduceALU(S, recip="div")
        check(f"reduce depth O(1) S={S}", m.reduce_op_depth == 5, f"(= {m.reduce_op_depth})")
        check(f"op depth S={S}", m.op_depth == 7, f"(= {m.op_depth})")
        n_cells = m.table_correct(m._reference())["_n_cells"]
        print(f"    S={S}: params={sum(p.numel() for p in m.parameters()):,} "
              f"cells={n_cells} recip_depth={m.recip_op_depth} full={m.full_op_depth}")


# --------------------------------------------------------------------------- #
def sample_semiprimes(bits, k, seed=7):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from probe_monoid import sample_semiprime
    rng = random.Random(seed)
    out = []
    while len(out) < k:
        m = sample_semiprime(bits, rng)
        if m not in out:
            out.append(m)
    return out


def test_basin_instrument():
    """POSITIVE CONTROL: `repaired` must register a repair when one happens.

    Every basin cell in this branch reads 0, so the instrument has to be shown
    not to be blind.  Corrupt k cells, hand half of them back their true values,
    and check the count is exactly that half -- and that the untouched half
    still reads as unrepaired.
    """
    m = O1ReduceALU(3, recip="oracle").to(DEV)
    m.construct_()
    ref = m._reference()
    g = torch.Generator().manual_seed(3)
    hit = m.corrupt_(40, g, 0.5, "uniform")
    pre = m.cell_correct(ref)
    ok0, tot0, _ = m.repaired(hit, ref, pre)
    check("instrument: nothing repaired before repairing", ok0 == 0, f"({ok0}/{tot0})")

    want, n_put = 0, {}
    with torch.no_grad():
        rc = {n: p for n, p, _, _, _ in ref._cells()}
        for name, p, shape, dim, b in m._cells():
            if name not in hit:
                continue
            idx = hit[name]
            wrong = idx[~pre[name][idx.to(p.device)].cpu()]
            half = wrong[: len(wrong) // 2]
            if len(half):
                p.data.view(*shape)[half.to(p.device)] = rc[name].view(*shape)[half.to(p.device)]
                want += len(half)
                n_put[name] = len(half)
    ok, tot, by_depth = m.repaired(hit, ref, pre)
    check("instrument: registers exactly the cells handed back",
          ok == want and tot == tot0, f"({ok} repaired of {tot}, expected {want} of {tot0})")
    check("instrument: resolves them by depth",
          sum(v[0] for v in by_depth.values()) == want,
          f"(by_depth={ {k: v for k, v in by_depth.items()} })")


def main():
    torch.manual_seed(0)
    # hf1: ID bits [16,18,20], OOD-N [17,19,21]; S is set by the widest (7 digits)
    hf_id = sum((sample_semiprimes(b, 3) for b in (16, 18, 20)), [])
    hf_ood = sum((sample_semiprimes(b, 2, seed=9) for b in (17, 19, 21)), [])
    S_HF = slots_for(hf_id + hf_ood)

    print("== Barrett bound ==")
    test_barrett_bound([323, 899])
    test_barrett_bound([2021, 10403])
    test_barrett_bound(hf_id, S_HF, "hf1-ID")
    test_barrett_bound(hf_ood, S_HF, "hf1-OODN")

    print("== reciprocal by learned long division ==")
    test_longdiv([323, 899])
    test_longdiv([10403, 38021, 65537 * 3])
    test_longdiv(hf_id + hf_ood, S_HF)

    print("== constructed ceiling (DIAGNOSTIC) ==")
    for recip in ("oracle", "div"):
        test_construct([323, 899], recip, hard=False)
        test_construct([323, 899], recip, hard=True)
    test_construct([323], "div", hard=False, T=4)
    test_construct([10403, 38021], "div", hard=False)
    test_construct([10403, 38021], "div", hard=True)
    # hf1's own moduli -- the ceiling the mandate demands, soft AND argmax-hard
    for hard in (False, True):
        test_construct(hf_id, "div", hard=hard, S=S_HF, n_ops=180)
        test_construct(hf_ood, "div", hard=hard, S=S_HF, n_ops=120)
    test_construct(hf_id, "div", hard=False, S=S_HF, amp=True, n_ops=180)
    test_construct(hf_id, "div", hard=True, S=S_HF, amp=True, n_ops=180)
    test_construct(hf_id, "div", hard=False, S=S_HF, T=4, n_ops=120)

    print("== basin instrument (positive control) ==")
    test_basin_instrument()

    print("== depths ==")
    test_depths()

    print(("ALL CHECKS PASSED" if not FAILED else f"FAILURES: {FAILED}"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
