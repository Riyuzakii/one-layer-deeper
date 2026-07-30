#!/usr/bin/env python
"""Correctness + realised-eigenvalue verification for §3.2 / §3.3.

PLAN2 §5: "Write a serial reference implementation and assert equality first --
off-by-one and left/right composition-order bugs here are silent and will look
like 'the architecture doesn't work.'"

PLAN2 §3.3: "Verify the realised eigenvalue range numerically."

Run:  $VENV lab/p2_verify.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lab.p2_layers import (  # noqa: E402
    DeltaProductMixer,
    MatrixScanMixer,
    PDSSMMixer,
    set_impl,
)

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def hr(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


# --------------------------------------------------------------------------- #
def test_matrix_scan_against_serial() -> None:
    hr("1. matrix associative scan == serial left-fold  (§3.1 reference)")
    b, h, t, n = 3, 2, 13, 8
    m = torch.eye(n).expand(b, h, t, n, n) + 0.15 * torch.randn(b, h, t, n, n)
    m = m.contiguous()
    fast = MatrixScanMixer._prefix_products(m.clone())
    serial = torch.empty_like(m)
    acc = torch.eye(n).expand(b, h, n, n).clone()
    for k in range(t):
        acc = torch.matmul(m[:, :, k], acc)  # P_k = m_k @ P_{k-1}
        serial[:, :, k] = acc
    err = (fast - serial).abs().max().item()
    print(f"  T={t} N={n}  max|fast-serial| = {err:.3e}   {'OK' if err < 2e-4 else 'FAIL'}")
    assert err < 2e-4, "prefix-product composition order is wrong"


# --------------------------------------------------------------------------- #
def test_delta_householder_spectrum() -> None:
    hr("2. §3.3 realised eigenvalue range  --  the trap PLAN2 flags")
    b, t, d = 4, 13, 128
    for eig in ("pos", "neg"):
        mix = DeltaProductMixer(d, n_heads=2, head_dim=16, n_h=2, eig_range=eig,
                                  bidirectional=True).to(DEV)
        x = torch.randn(b, t, d, device=DEV)
        ev = mix.transition_eigenvalues(x)
        # explicit construction of I - beta k k^T and its numerical spectrum
        with torch.no_grad():
            k = torch.nn.functional.normalize(
                mix.k_proj(x).float().view(b, t, 2, 2, 2, 16)[:, :, 0, :, :, :], dim=-1
            )
            beta = mix._betas(x)[:, :, 0]  # (B,T,H,n_h)
            kk = k.reshape(-1, 16)
            bb = beta.permute(0, 1, 2, 3).reshape(-1)
            eye = torch.eye(16, device=DEV)
            hh = eye - bb[:, None, None] * torch.einsum("bi,bj->bij", kk, kk)
            spec = torch.linalg.eigvals(hh.cpu()).real
        print(
            f"  eig_range={eig:>3}  beta in [{(1-ev).min():.4f}, {(1-ev).max():.4f}]"
            f"   nontrivial eig 1-beta in [{ev.min():+.4f}, {ev.max():+.4f}]"
        )
        print(
            f"                 numerical spectrum of I-beta k k^T in "
            f"[{spec.min():+.4f}, {spec.max():+.4f}]   "
            f"frac(eig<0) = {(spec < -1e-6).float().mean():.4f}"
        )
        if eig == "pos":
            assert ev.min() > -1e-6, "pos range should not produce negative eigenvalues"
        else:
            assert ev.min() < -0.05, "neg range produced NO negative eigenvalue -- the trap"
        assert ev.min() >= -1.0001 and ev.max() <= 1.0001


# --------------------------------------------------------------------------- #
def test_deltaproduct_nh_equivalence() -> None:
    hr("3. §3.3  DeltaProduct(n_h) == DeltaNet on an n_h-times-expanded axis")
    b, t, d = 2, 7, 64
    mix = DeltaProductMixer(d, n_heads=1, head_dim=8, n_h=3, eig_range="neg",
                              bidirectional=True).to(DEV)
    x = torch.randn(b, t, d, device=DEV)
    with torch.no_grad():
        q = torch.nn.functional.normalize(
            mix.q_proj(x).float().view(b, t, 2, 8)[:, :, 0].view(b, t, 1, 8).transpose(1, 2),
            dim=-1,
        )
        k = mix.k_proj(x).float().view(b, t, 2, 3 * 8)[:, :, 0]
        v = mix.v_proj(x).float().view(b, t, 2, 3 * 8)[:, :, 0]
        beta = mix._betas(x)[:, :, 0]
        k = torch.nn.functional.normalize(
            k.view(b, t, 3, 1, 8).permute(0, 3, 1, 2, 4).reshape(b, 1, t * 3, 8), dim=-1
        )
        v = v.view(b, t, 3, 1, 8).permute(0, 3, 1, 2, 4).reshape(b, 1, t * 3, 8)
        beta = beta.permute(0, 2, 1, 3).reshape(b, 1, t * 3)
        got = mix._scan(q, k, v, beta)
        # explicit reference: build the product of Householders per token
        state = torch.zeros(b, 1, 8, 8, device=DEV)
        ref = []
        idx = 0
        prod_all = torch.eye(8, device=DEV).expand(b, 1, 8, 8).clone()
        for step in range(t):
            for _ in range(3):
                ki, vi, bi = k[:, :, idx], v[:, :, idx], beta[:, :, idx, None]
                hh = torch.eye(8, device=DEV) - bi[..., None] * torch.einsum(
                    "bhi,bhj->bhij", ki, ki
                )
                state = torch.matmul(state, hh) + bi[..., None] * torch.einsum(
                    "bhi,bhj->bhij", vi, ki
                )
                prod_all = torch.matmul(prod_all, hh)
                idx += 1
            ref.append(torch.einsum("bhij,bhj->bhi", state, q[:, :, step]))
        ref = torch.stack(ref, dim=2)
    err = (got - ref).abs().max().item()
    print(f"  max|fused - explicit Householder product| = {err:.3e}  {'OK' if err<1e-4 else 'FAIL'}")
    assert err < 1e-4
    # complex kernels JIT-fail on this box's arch -> do spectra on CPU
    spec = torch.linalg.eigvals(prod_all.reshape(-1, 8, 8).cpu())
    print(
        f"  |eig| of the n_h=3 PRODUCT over all {t} tokens: "
        f"[{spec.abs().min():.4f}, {spec.abs().max():.4f}]  (BIBO: max<=1)"
    )
    print(f"  fraction of product eigenvalues with negative real part: "
          f"{(spec.real < 0).float().mean():.4f}   "
          f"(a diagonal-[0,1] model can produce NONE of these)")
    assert spec.abs().max().item() <= 1.0001


# --------------------------------------------------------------------------- #
def test_pdssm_structure() -> None:
    hr("4. §3.2  PD-SSM: A_t = P_t D_t, P column one-hot, |eig(D)| < 1")
    b, t, d, n = 4, 13, 64, 8
    for tau in (0.1, 1.0, 5.0):
        mix = PDSSMMixer(d, n_heads=2, state_size=n, tau=tau, ste="hard",
                         bidirectional=True).to(DEV)
        x = torch.randn(b, t, d, device=DEV)
        st = mix.transition_stats(x)
        print(
            f"  tau={tau:<5} p_max_prob={st['p_max_prob']:.4f} "
            f"p_entropy={st['p_entropy']:.4f} perm_frac={st['p_perm_frac']:.3f} "
            f"|D| mean={st['d_abs_mean']:.4f} max={st['d_abs_max']:.4f}"
        )
    # forward is exactly column one-hot?
    mix = PDSSMMixer(d, n_heads=2, state_size=n, tau=1.0, ste="hard",
                     bidirectional=True).to(DEV)
    x = torch.randn(b, t, d, device=DEV, requires_grad=True)
    logits = mix.p_proj(x).float().view(b, t, 2, 2, n, n)[:, :, 0]
    soft = torch.softmax(logits / mix.tau, dim=-2)
    idx = soft.argmax(dim=-2, keepdim=True)
    hard = torch.zeros_like(soft).scatter_(-2, idx, 1.0)
    p = hard + soft - soft.detach()
    colsum = p.sum(dim=-2)
    print(f"  forward P column sums: min={colsum.min():.6f} max={colsum.max():.6f} (should be 1)")
    nnz = (p.detach() > 0.5).float().sum(dim=-2)
    print(f"  forward P nonzeros per column: min={nnz.min():.1f} max={nnz.max():.1f} (should be 1)")
    assert abs(colsum.min().item() - 1) < 1e-4 and abs(colsum.max().item() - 1) < 1e-4
    # gradient flows through the SOFT path
    y = mix(x)
    y.sum().backward()
    g = mix.p_proj.weight.grad
    print(f"  d(loss)/d(p_proj.weight) norm = {g.norm():.4e}  "
          f"{'OK (STE passes gradient)' if g.norm() > 0 else 'FAIL (no gradient)'}")
    assert g.norm().item() > 0


# --------------------------------------------------------------------------- #
def test_pdssm_closure() -> None:
    hr("5. §3.2  the closure that makes PD-SSM scannable at O(N) per combine")
    n = 6
    torch.manual_seed(1)

    def rand_pd():
        sigma = torch.randint(0, n, (n,))
        p = torch.zeros(n, n, dtype=torch.cfloat)
        p[sigma, torch.arange(n)] = 1.0
        d = torch.randn(n, dtype=torch.cfloat)
        d = d / d.abs() * 0.9
        return sigma, d, p @ torch.diag(d)

    s1, d1, a1 = rand_pd()
    s2, d2, a2 = rand_pd()
    dense = a1 @ a2
    # closed form:  (P1 D1)(P2 D2) = P_{s1 o s2} diag( D1[s2] * D2 )
    s = s1[s2]
    d = d1[s2] * d2
    p = torch.zeros(n, n, dtype=torch.cfloat)
    p[s, torch.arange(n)] = 1.0
    closed = p @ torch.diag(d)
    err = (dense - closed).abs().max().item()
    print(f"  max|dense product - O(N) closed form| = {err:.3e}  {'OK' if err<1e-5 else 'FAIL'}")
    print("  => the PD family is closed under composition; an associative scan")
    print("     combines two elements in O(N) (a gather + a complex multiply).")
    assert err < 1e-5


# --------------------------------------------------------------------------- #
def test_fast_paths_match_sequential() -> None:
    hr("6. fast (chunkwise / log-depth) path == sequential reference")
    b, t, d = 6, 13, 128
    x = torch.randn(b, t, d, device=DEV)
    for n_h in (1, 2, 3):
        mix = DeltaProductMixer(
            d, n_heads=4, head_dim=32, n_h=n_h, eig_range="neg", bidirectional=True
        ).to(DEV)
        set_impl("seq")
        ref = mix(x)
        set_impl("fast")
        got = mix(x)
        err = (got - ref).abs().max().item()
        rel = err / ref.abs().max().item()
        print(f"  DeltaProduct n_h={n_h}  max abs err={err:.3e} rel={rel:.3e}  "
              f"{'OK' if rel < 1e-4 else 'FAIL'}")
        assert rel < 1e-4
    for ste in ("hard", "none"):
        mix = PDSSMMixer(
            d, n_heads=4, state_size=16, tau=1.0, ste=ste, bidirectional=True
        ).to(DEV)
        set_impl("seq")
        ref = mix(x)
        set_impl("fast")
        got = mix(x)
        err = (got - ref).abs().max().item()
        rel = err / ref.abs().max().item()
        print(f"  PD-SSM ste={ste:<5}      max abs err={err:.3e} rel={rel:.3e}  "
              f"{'OK' if rel < 1e-4 else 'FAIL'}")
        assert rel < 1e-4
    set_impl("fast")


def test_wall_clock() -> None:
    hr("7. wall clock at the real task's shape (B=128, L=13, D=128, 2 layers)")
    import time

    b, t, d = 128, 13, 128
    x = torch.randn(b, t, d, device=DEV)
    specs = [
        ("delta n_h=1", lambda: DeltaProductMixer(d, n_heads=4, head_dim=32, n_h=1,
                                                  eig_range="neg", bidirectional=True)),
        ("delta n_h=2", lambda: DeltaProductMixer(d, n_heads=4, head_dim=32, n_h=2,
                                                  eig_range="neg", bidirectional=True)),
        ("delta n_h=4", lambda: DeltaProductMixer(d, n_heads=4, head_dim=32, n_h=4,
                                                  eig_range="neg", bidirectional=True)),
        ("pdssm N=16", lambda: PDSSMMixer(d, n_heads=4, state_size=16, tau=1.0,
                                          ste="hard", bidirectional=True)),
        ("matscan N=16", lambda: MatrixScanMixer(d, n_heads=4, state_size=16,
                                                 bidirectional=True)),
    ]
    for name, ctor in specs:
        for impl in ("seq", "fast"):
            if name.startswith("matscan") and impl == "seq":
                continue
            set_impl(impl)
            mix = ctor().to(DEV)
            for _ in range(3):
                mix(x).sum().backward()
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(20):
                mix.zero_grad(set_to_none=True)
                mix(x).sum().backward()
            torch.cuda.synchronize()
            ms = (time.time() - t0) / 20 * 1000
            print(f"  {name:<14} impl={impl:<5}  fwd+bwd {ms:7.2f} ms")
    set_impl("fast")


def test_full_model_throughput(batch: int = 128, length: int = 13) -> None:
    """Matched wall clock at the level that decides steps-in-budget.

    All configurations are timed back-to-back in ONE process so they share
    whatever GPU contention exists; relative numbers are the meaningful ones.
    """
    hr(f"8. full-model fwd+bwd, B={batch} L={length} vocab=17, 2 layers "
       f"(matched wall clock)")
    import importlib.util
    import os
    import time

    from benchmark import ModelSpec

    spec = ModelSpec(vocab_size=17, max_seq_len=length,
                     maximum_model_state_elements=5 * 10**8)
    ids = torch.randint(0, 17, (batch, length), device=DEV)
    mask = torch.ones(batch, length, dtype=torch.bool, device=DEV)
    cfgs = [
        ("§3.3 DeltaProduct n_h=1", dict(P2_ARCH="delta", P2_NH=1)),
        ("§3.3 DeltaProduct n_h=2", dict(P2_ARCH="delta", P2_NH=2)),
        ("§3.3 DeltaProduct n_h=3", dict(P2_ARCH="delta", P2_NH=3)),
        ("§3.3 DeltaProduct n_h=4", dict(P2_ARCH="delta", P2_NH=4)),
        ("§3.2 PD-SSM N=16", dict(P2_ARCH="pdssm", P2_STATE=16)),
        ("§3.2 PD-SSM N=8", dict(P2_ARCH="pdssm", P2_STATE=8)),
        ("§3.1 matrix scan N=16", dict(P2_ARCH="matscan", P2_STATE=16)),
    ]
    base = None
    for label, env in cfgs:
        for k in ("P2_ARCH", "P2_NH", "P2_STATE"):
            os.environ.pop(k, None)
        for k, v in env.items():
            os.environ[k] = str(v)
        s = importlib.util.spec_from_file_location("_p2_timing", _SUBPATH)
        mod = importlib.util.module_from_spec(s)
        s.loader.exec_module(mod)
        m = mod.build_model(spec).to(DEV)
        params = sum(p.numel() for p in m.parameters())
        for _ in range(5):
            m(ids, mask)[0].float().logsumexp(-1).sum().backward()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(30):
            m.zero_grad(set_to_none=True)
            m(ids, mask)[0].float().logsumexp(-1).sum().backward()
        torch.cuda.synchronize()
        ms = (time.time() - t0) / 30 * 1000
        if base is None:
            base = ms
        print(f"  {label:<26} {params:>9,} params   {ms:7.2f} ms/step   "
              f"{1000/ms:7.1f} steps/s   {base/ms:5.2f}x vs n_h=1")
    for k in ("P2_ARCH", "P2_NH", "P2_STATE"):
        os.environ.pop(k, None)


_SUBPATH = (
    Path(__file__).resolve().parent.parent
    / "submissions" / "plan2-pd-ssm-delta" / "submission.py"
)


if __name__ == "__main__":
    test_matrix_scan_against_serial()
    test_delta_householder_spectrum()
    test_deltaproduct_nh_equivalence()
    test_pdssm_structure()
    test_pdssm_closure()
    test_fast_paths_match_sequential()
    test_wall_clock()
    test_full_model_throughput()
    print("\nALL VERIFICATION CHECKS PASSED\n")
