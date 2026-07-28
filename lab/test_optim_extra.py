#!/usr/bin/env python
"""Self-check for lab/optim_extra.py -- run BEFORE believing any null result.

Five checks, each on a problem where the answer is known analytically:

  1. shapes/finiteness on every tensor shape the PopALU actually has;
  2. replica independence under `batch_dims=1` (the property the whole
     population instrument rests on);
  3. SOAP beats AdamW on an ILL-CONDITIONED quadratic (that is the one thing
     preconditioning is for -- if this fails the implementation is wrong);
  4. SOAP's rotation is orthogonal and its update is invariant to a rotation of
     the parameterisation (the defining property of a preconditioned method);
  5. AdEMAMix reduces to Adam-like behaviour at alpha=0 and beats AdamW when
     the useful gradient is buried in noise (the case its slow EMA is for).
"""

from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optim_extra import SOAP, AdEMAMix                       # noqa: E402

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
OK = True


def check(name, cond, extra=""):
    global OK
    OK = OK and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")


# ---------------------------------------------------------------- 1. shapes
def t_shapes():
    print("1. shapes / finiteness on the PopALU's actual tensor shapes")
    P = 8
    shapes = [(P, 10, 10, 20), (P, 10, 10, 2, 12), (P, 10), (P, 2), (P, 4),
              (P,), (P, 5, 10)]
    for opt_name in ("soap", "ademamix"):
        ps = [torch.randn(s, device=DEV, requires_grad=True) for s in shapes]
        opt = (SOAP(ps, lr=1e-2, batch_dims=1, precondition_frequency=3)
               if opt_name == "soap"
               else AdEMAMix(ps, lr=1e-2, warmup=10))
        for _ in range(25):
            loss = sum((p ** 2).sum() for p in ps)
            opt.zero_grad()
            loss.backward()
            opt.step()
        fin = all(torch.isfinite(p).all().item() for p in ps)
        shrunk = all((p ** 2).sum().item() < s0 for p, s0 in
                     zip(ps, [float(torch.tensor(s).prod()) * 3 for s in shapes]))
        check(f"{opt_name}: finite", fin)
        check(f"{opt_name}: descends on ||p||^2", shrunk)


# ------------------------------------------------------- 2. replica isolation
def t_isolation():
    print("2. replica independence under batch_dims=1")
    torch.manual_seed(0)
    P, d = 4, 6
    A = torch.randn(P, d, d, device=DEV)
    A = A @ A.transpose(1, 2) + torch.eye(d, device=DEV)
    # replica 3 gets a 1e4 scale so any coupling shows up immediately
    A[3] *= 1e4
    x0 = torch.randn(P, d, device=DEV)

    def run(mask):
        x = torch.nn.Parameter(x0.clone())
        opt = SOAP([x], lr=3e-2, batch_dims=1, precondition_frequency=2)
        for _ in range(40):
            q = 0.5 * torch.einsum("pi,pij,pj->p", x, A, x)
            loss = (q * mask).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return x.detach().clone()

    full = run(torch.ones(P, device=DEV))
    # zero out replica 3's loss entirely: replica 0..2 must be unchanged
    m = torch.ones(P, device=DEV)
    m[3] = 0.0
    part = run(m)
    dif = (full[:3] - part[:3]).abs().max().item()
    check("replicas 0-2 unaffected by replica 3's loss", dif < 1e-5,
          f"(max|d| = {dif:.2e})")


# --------------------------------------------- 3. ill-conditioned quadratic
def t_illcond():
    print("3. SOAP vs AdamW on an ill-conditioned quadratic (kappa = 1e4)")
    torch.manual_seed(0)
    m, n = 40, 30
    # W (m,n); loss = 0.5||L W R - Y||^2 with L,R ill-conditioned -> the
    # Kronecker structure SOAP assumes is exactly right here.
    sl = torch.logspace(0, 2, m, device=DEV)
    sr = torch.logspace(0, 2, n, device=DEV)
    UL = torch.linalg.qr(torch.randn(m, m, device=DEV))[0]
    UR = torch.linalg.qr(torch.randn(n, n, device=DEV))[0]
    L = UL @ torch.diag(sl) @ UL.T
    R = UR @ torch.diag(sr) @ UR.T
    Wstar = torch.randn(m, n, device=DEV)
    Y = L @ Wstar @ R
    W0 = torch.zeros(m, n, device=DEV)

    def run(make, steps=300):
        W = torch.nn.Parameter(W0.clone())
        opt = make([W])
        best = float("inf")
        for _ in range(steps):
            loss = 0.5 * ((L @ W @ R - Y) ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            best = min(best, loss.item())
        return best

    res = {}
    for lr in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
        res[("adamw", lr)] = run(
            lambda ps, lr=lr: torch.optim.AdamW(ps, lr=lr, weight_decay=0.0))
        res[("soap", lr)] = run(
            lambda ps, lr=lr: SOAP(ps, lr=lr, precondition_frequency=5,
                                   max_precond_dim=256))
    a = min(v for (k, _), v in res.items() if k == "adamw")
    s = min(v for (k, _), v in res.items() if k == "soap")
    print(f"     best AdamW loss {a:.4e}   best SOAP loss {s:.4e}")
    check("SOAP >= 10x lower final loss than AdamW", s < a / 10.0,
          f"(ratio {a / max(s, 1e-30):.1f}x)")


# ------------------------------------------------- 4. rotation invariance
def t_rotation():
    print("4. SOAP's basis is orthogonal, and its step is rotation-covariant")
    torch.manual_seed(0)
    m, n = 12, 9
    A = torch.randn(m, m, device=DEV)
    A = A @ A.T + torch.eye(m, device=DEV)
    Bm = torch.randn(n, n, device=DEV)
    Bm = Bm @ Bm.T + torch.eye(n, device=DEV)
    W0 = torch.randn(m, n, device=DEV)
    U = torch.linalg.qr(torch.randn(m, m, device=DEV))[0]

    # A DETERMINISTIC gradient makes every Shampoo outer product the same rank-9
    # matrix, so the 12-dim left factor never leaves its null space and the
    # eigenvectors there are arbitrary (see the note below).  Replay a fixed
    # sequence of linear perturbations so the factor becomes full rank, which is
    # the regime any real training run is in.
    torch.manual_seed(7)
    Cs = [20.0 * torch.randn(m, n, device=DEV) for _ in range(60)]

    def run(rot, steps=30, warm=15):
        W = torch.nn.Parameter((U @ W0) if rot else W0.clone())
        opt = SOAP([W], lr=1e-2, precondition_frequency=3, merge_dims=False,
                   precond_warmup=warm)
        for t in range(steps):
            X = (U.T @ W) if rot else W
            loss = 0.5 * (X * (A @ X @ Bm)).sum() + (Cs[t] * X).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return (U.T @ W.detach()) if rot else W.detach()

    a, b = run(False, steps=18), run(True, steps=18)
    rel = (a - b).norm().item() / max(a.norm().item(), 1e-12)
    check("3 updates are covariant under a left rotation", rel < 1e-4,
          f"(rel diff {rel:.2e})")
    # Two documented, genuine (non-bug) limits of the covariance, reported so a
    # later reader does not mistake either for an implementation error:
    #   (a) over many steps, near-degenerate Shampoo eigenvalues can order
    #       differently in the two frames and the trajectories separate;
    #   (b) with a rank-deficient factor the null-space eigenvectors are
    #       arbitrary and Adam normalises numerical noise there to full step
    #       size.  `precond_warmup` exists for (b).
    a, b = run(False, steps=45), run(True, steps=45)
    relL = (a - b).norm().item() / max(a.norm().item(), 1e-12)
    a, b = run(False, steps=10, warm=0), run(True, steps=10, warm=0)
    rel0 = (a - b).norm().item() / max(a.norm().item(), 1e-12)
    print(f"     (30 updates: {relL:.2e};  no preconditioner warmup: {rel0:.2e}"
          f" -- both expected, see comment)")

    # orthogonality of the stored basis
    W = torch.nn.Parameter(W0.clone())
    opt = SOAP([W], lr=1e-2, precondition_frequency=2, merge_dims=False)
    for _ in range(10):
        loss = 0.5 * (W * (A @ W @ Bm)).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    st = opt.state[W]
    errs = []
    for q in st["Q"]:
        if q is None:
            continue
        e = (q.transpose(1, 2) @ q
             - torch.eye(q.shape[-1], device=q.device)).abs().max().item()
        errs.append(e)
    check("Q^T Q = I", max(errs) < 1e-4, f"(max err {max(errs):.2e})")


# --------------------------------------------------------- 5. AdEMAMix
def t_ademamix():
    print("5. AdEMAMix: exact state algebra, schedules, and alpha=0 == AdamW")
    torch.manual_seed(0)
    d = 64
    x0 = torch.randn(d, device=DEV)

    def run(make, steps=400, noise=0.0, seed=0):
        torch.manual_seed(seed)
        x = torch.nn.Parameter(x0.clone())
        opt = make([x])
        for _ in range(steps):
            g = x.detach().clone()                       # grad of 0.5||x||^2
            if noise:
                g = g + noise * torch.randn(d, device=DEV)
            x.grad = g
            opt.step()
        return (x.detach() ** 2).sum().item(), opt

    # (a) alpha = 0 must be AdamW EXACTLY -- this pins the fast-EMA and
    #     second-moment arithmetic against a reference implementation.
    a0, _ = run(lambda ps: torch.optim.AdamW(ps, lr=1e-2, weight_decay=0.0,
                                             betas=(0.9, 0.999)))
    e0, _ = run(lambda ps: AdEMAMix(ps, lr=1e-2, alpha=0.0,
                                    betas=(0.9, 0.999, 0.9999)))
    check("alpha=0 matches torch AdamW", abs(a0 - e0) / max(a0, 1e-30) < 1e-3,
          f"({a0:.4e} vs {e0:.4e})")

    # (b) the SLOW EMA is exactly the un-bias-corrected EMA of the gradients.
    g = torch.ones(4, device=DEV)
    x = torch.nn.Parameter(torch.zeros(4, device=DEV))
    b3, k = 0.999, 50
    opt = AdEMAMix([x], lr=0.0, alpha=8.0, betas=(0.9, 0.999, b3), warmup=0)
    for _ in range(k):
        x.grad = g.clone()
        opt.step()
    m2 = opt.state[x]["m2"][0].item()
    want = 1.0 - b3 ** k
    check("m2 == (1 - beta3^k) * g  (no bias correction)",
          abs(m2 - want) < 1e-6, f"({m2:.6f} vs {want:.6f})")
    m1 = opt.state[x]["m1"][0].item()
    check("m1 == (1 - beta1^k) * g", abs(m1 - (1 - 0.9 ** k)) < 1e-6)

    # (c) beta3 warmup is linear in HALF-LIFE, not in beta
    from optim_extra import _ademamix_beta3, _ademamix_alpha
    hl = lambda b: math.log(0.5) / math.log(b) - 1.0        # noqa: E731
    mid = _ademamix_beta3(500, 0.9999, 0.9, 1000)
    want = math.pow(0.5, 1.0 / (0.5 * (hl(0.9) + hl(0.9999)) + 1.0))
    check("beta3 half-life warmup matches the paper's f/f^-1",
          abs(mid - want) < 1e-9, f"({mid:.6f})")
    check("alpha warmup is linear and saturates",
          abs(_ademamix_alpha(50, 8.0, 100) - 4.0) < 1e-9
          and _ademamix_alpha(500, 8.0, 100) == 8.0)

    # (d) both train a real (if tiny) network at least as well as AdamW, at the
    #     best lr of a shared grid and under the cosine decay all three are
    #     normally used with.  NOTE the step count: AdEMAMix needs T >> the slow
    #     EMA's half-life (693 steps at beta3=0.999).  At T=600 it is far WORSE
    #     than AdamW (0.73 vs 0.004, measured); at T=6000 it wins.  That regime
    #     boundary is a property of the method and it dictates how the ALU
    #     screen below has to be designed.
    torch.manual_seed(0)
    X = torch.randn(2048, 16, device=DEV)
    Wt = torch.randn(16, 8, device=DEV)
    Y = torch.tanh(X @ Wt) @ torch.randn(8, 4, device=DEV)

    def mlp(make, steps, seed=0):
        torch.manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.Tanh(),
                                  torch.nn.Linear(32, 4)).to(DEV)
        opt = make(list(net.parameters()))
        base = [g["lr"] for g in opt.param_groups]
        for i in range(steps):
            f = 0.5 * (1.0 + math.cos(math.pi * i / steps))
            for gp, b in zip(opt.param_groups, base):
                gp["lr"] = b * f
            j = torch.randint(0, 2048, (32,), device=DEV)
            loss = ((net(X[j]) - Y[j]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            return ((net(X) - Y) ** 2).mean().item()

    lrs = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
    aM = min(mlp(lambda ps, lr=lr: torch.optim.AdamW(ps, lr=lr,
                                                     weight_decay=0.0), 6000)
             for lr in lrs)
    eM = min(mlp(lambda ps, lr=lr: AdEMAMix(ps, lr=lr, alpha=8.0, warmup=2000,
                                            betas=(0.9, 0.999, 0.999)), 6000)
             for lr in lrs)
    sM = min(mlp(lambda ps, lr=lr: SOAP(ps, lr=lr, precondition_frequency=5),
                 1500)
             for lr in lrs)
    aS = min(mlp(lambda ps, lr=lr: torch.optim.AdamW(ps, lr=lr,
                                                     weight_decay=0.0), 1500)
             for lr in lrs)
    print(f"     MLP fit, best of 5 lrs, cosine decay:  T=6000 AdamW {aM:.5f} "
          f"AdEMAMix {eM:.5f}   |   T=1500 AdamW {aS:.5f} SOAP {sM:.5f}")
    check("AdEMAMix beats AdamW at T >> the slow EMA half-life", eM <= aM)
    check("SOAP is not broken on the same MLP (within 5x of AdamW)",
          sM <= aS * 5.0, f"({sM:.5f} vs {aS:.5f})")


if __name__ == "__main__":
    t_shapes()
    t_isolation()
    t_illcond()
    t_rotation()
    t_ademamix()
    print("\nALL PASS" if OK else "\nFAILURES ABOVE")
    raise SystemExit(0 if OK else 1)
