#!/usr/bin/env python
"""SOAP and AdEMAMix as plain `torch.optim.Optimizer` subclasses.

WHY THIS FILE EXISTS.  Across this project the optimizer sweep covered AdamW,
Muon, Schedule-Free AdamW, Grokfast, perp-Grad and StableMax -- all on the
*dense transformer* (`explore/grok-optimization`).  On the `DigitALU` tables
only AdamW hyperparameters were ever varied.  Nothing second-order and nothing
with a slow-EMA component has ever been tried on the ALU.  The ALU is 6,820
parameters, so full-matrix preconditioning is affordable here in a way it is
not at scale.

Both are written from the published algorithms; nothing is pip-installed.

  SOAP      Vyas, Morwani, Zhao, Shapira, Brandfonbrener, Janson, Kakade,
            "SOAP: Improving and Stabilizing Shampoo using Adam" (2024).
            Shampoo's Kronecker-factored preconditioner is used only to supply
            a rotation; Adam is then run in that eigenbasis.
  AdEMAMix  Pagliardini, Ablin, Grangier, "The AdEMAMix Optimizer: Better,
            Faster, Older" (2024).  A mixture of a fast (beta1) and a very slow
            (beta3 ~ 0.9999) gradient EMA, divided by Adam's second moment.

COMPLIANCE.  `benchmark/runner.py` calls `build_optimizer` and then owns the
loop, the backward, the grad clip and the one-`step()`-per-batch cadence.
A custom `torch.optim.Optimizer` is explicitly permitted (BRIEF §4 rule 4 bans
a custom training *loop*, not a custom optimizer).  Neither optimizer here
touches data, labels, or the autograd graph: `step()` sees only `p` and
`p.grad`.

THE BATCH-DIM ARGUMENT (`batch_dims`, load-bearing for the population probes).
`lab/probe_pop.py` gives every parameter a leading replica axis of size P, and
the whole population instrument rests on replicas being *independent draws*
(alu-population §4.2).  A preconditioner computed across the replica axis would
couple them and silently destroy that property.  With `batch_dims=1` SOAP keeps
a separate preconditioner per replica, so P replicas remain P independent SOAP
runs.  AdEMAMix is elementwise and independent by construction.
"""

from __future__ import annotations

import math

import torch


# --------------------------------------------------------------------------
# AdEMAMix
# --------------------------------------------------------------------------
def _ademamix_alpha(step: int, alpha: float, warmup: int) -> float:
    """Linear warmup of the slow-EMA mixing coefficient (paper eq. 6)."""
    if warmup <= 0 or step >= warmup:
        return alpha
    return alpha * step / float(warmup)


def _ademamix_beta3(step: int, beta_end: float, beta_start: float,
                    warmup: int) -> float:
    """Warm up beta3 linearly in HALF-LIFE, as the paper's scheduler does.

    Interpolating beta directly spends almost all of the warmup near beta_end;
    the paper interpolates f(beta) = log(0.5)/log(beta) - 1, the EMA half-life.
    """
    if warmup <= 0 or step >= warmup:
        return beta_end

    def hl(b: float) -> float:
        return math.log(0.5) / math.log(b + 1e-12) - 1.0

    a = step / float(warmup)
    t = (1.0 - a) * hl(beta_start) + a * hl(beta_end)
    return math.pow(0.5, 1.0 / (t + 1.0))


class AdEMAMix(torch.optim.Optimizer):
    """AdEMAMix (Pagliardini et al., 2024).

        m1 <- b1 m1 + (1-b1) g            (fast EMA, bias-corrected)
        m2 <- b3 m2 + (1-b3) g            (slow EMA, NO bias correction)
        nu <- b2 nu + (1-b2) g^2
        p  <- p - lr * ( (m1_hat + alpha*m2) / (sqrt(nu_hat) + eps) + wd*p )

    The slow EMA is deliberately left un-bias-corrected: it is supposed to warm
    up slowly, and correcting it would make it behave like the fast one early.
    `alpha` and `beta3` are warmed up over `warmup` steps (paper §3.2) because
    a large alpha applied to a nearly-empty m2 is unstable.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.9999), alpha=8.0,
                 eps=1e-8, weight_decay=0.0, warmup=0, beta3_start=None):
        b1, b2, b3 = betas
        if beta3_start is None:
            beta3_start = b1
        defaults = dict(lr=lr, betas=(b1, b2, b3), alpha=alpha, eps=eps,
                        weight_decay=weight_decay, warmup=warmup,
                        beta3_start=beta3_start)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2, b3 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            warm = group["warmup"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = 0
                    st["m1"] = torch.zeros_like(p)
                    st["m2"] = torch.zeros_like(p)
                    st["nu"] = torch.zeros_like(p)
                st["step"] += 1
                k = st["step"]
                a_t = _ademamix_alpha(k, group["alpha"], warm)
                b3_t = _ademamix_beta3(k, b3, group["beta3_start"], warm)

                m1, m2, nu = st["m1"], st["m2"], st["nu"]
                m1.mul_(b1).add_(g, alpha=1.0 - b1)
                m2.mul_(b3_t).add_(g, alpha=1.0 - b3_t)
                nu.mul_(b2).addcmul_(g, g, value=1.0 - b2)

                bc1 = 1.0 - b1 ** k
                bc2 = 1.0 - b2 ** k
                denom = (nu / bc2).sqrt_().add_(eps)
                upd = (m1 / bc1 + a_t * m2) / denom
                if wd:
                    p.mul_(1.0 - lr * wd)
                p.add_(upd, alpha=-lr)
        return loss


# --------------------------------------------------------------------------
# SOAP
# --------------------------------------------------------------------------
def _merge_dims(shape, max_dim):
    """Official SOAP's dimension merging: fold adjacent axes together while the
    product stays <= max_dim, so a (10,10,20) table becomes (100,20) at 512 and
    a single (2000,) axis -- i.e. a genuine FULL-matrix preconditioner -- at
    4096."""
    new, cur = [], 1
    for sh in shape:
        if cur * sh > max_dim:
            if cur > 1:
                new.append(cur)
                cur = sh
            else:
                new.append(sh)
                cur = 1
        else:
            cur = cur * sh
    if cur > 1 or not new:
        new.append(cur)
    return tuple(new)


def _eigvecs(gg):
    """Descending-eigenvalue eigenvectors of a batch of SPD matrices."""
    d = gg.shape[-1]
    eye = torch.eye(d, device=gg.device, dtype=torch.float32)
    a = gg.float() + 1e-30 * eye
    try:
        _, q = torch.linalg.eigh(a)
    except Exception:                                    # pragma: no cover
        _, q = torch.linalg.eigh(a.double())
        q = q.float()
    return torch.flip(q, dims=[-1]).to(gg.dtype)


def _mm_axis(x, q, axis, back=False):
    """Contract `x`'s `axis` with the eigenbasis `q` (B,d,d), batched over the
    leading axis 0.  Forward is Q^T x (project into the eigenbasis); `back=True`
    is Q x (project out)."""
    x = x.movedim(axis, -1)
    shp = x.shape
    x2 = x.reshape(shp[0], -1, shp[-1])
    out = torch.bmm(x2, q.transpose(1, 2) if back else q)
    return out.reshape(shp).movedim(-1, axis)


def _reorder_axis(x, axis, idx):
    """Permute `x` along `axis` by the per-batch index `idx` (B,d)."""
    x = x.movedim(axis, -1)
    shp = x.shape
    x2 = x.reshape(shp[0], -1, shp[-1])
    out = torch.gather(x2, 2, idx[:, None, :].expand(-1, x2.shape[1], -1))
    return out.reshape(shp).movedim(-1, axis)


class SOAP(torch.optim.Optimizer):
    """SOAP (Vyas et al., 2024) with an optional leading batch axis.

    Per parameter, per non-batch axis i, keep the Shampoo factor
    `GG[i] = EMA(G_(i) G_(i)^T)` and its eigenvectors `Q[i]`.  Rotate the
    gradient into that basis, run Adam there, rotate the update back.  As in the
    reference implementation the FIRST moment is kept in the original space and
    rotated on use, while the SECOND moment lives in the rotated space; when the
    basis is refreshed the second moment is permuted to follow it.

    `batch_dims=k` treats the first k axes as independent problems: every
    preconditioner carries those axes, so replica p's update depends only on
    replica p's gradient.
    """

    MOMENTUM_SPACES = ("rot", "orig")

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), shampoo_beta=0.95,
                 eps=1e-8, weight_decay=0.0, precondition_frequency=10,
                 max_precond_dim=512, merge_dims=True, batch_dims=0,
                 precondition_1d=True, correct_bias=True,
                 momentum_space="rot", precond_warmup=0):
        assert momentum_space in self.MOMENTUM_SPACES
        defaults = dict(lr=lr, betas=betas, shampoo_beta=shampoo_beta, eps=eps,
                        weight_decay=weight_decay,
                        precondition_frequency=precondition_frequency,
                        max_precond_dim=max_precond_dim, merge_dims=merge_dims,
                        batch_dims=batch_dims, precondition_1d=precondition_1d,
                        correct_bias=correct_bias,
                        momentum_space=momentum_space,
                        precond_warmup=precond_warmup)
        super().__init__(params, defaults)

    # -- shapes -------------------------------------------------------------
    def _work_shape(self, p, group):
        """(B, d1, ..., dk): batch axes flattened to one, the rest optionally
        merged.  Returns None when there is nothing to precondition."""
        bd = group["batch_dims"]
        bshape = p.shape[:bd]
        rest = tuple(p.shape[bd:])
        if group["merge_dims"] and len(rest) > 1:
            rest = _merge_dims(rest, group["max_precond_dim"])
        B = 1
        for s in bshape:
            B *= s
        return B, rest

    def _view(self, t, group, B, rest):
        return t.reshape(B, *rest)

    # -- preconditioner -----------------------------------------------------
    def _accum(self, st, gw, group):
        """gw: (B, d1, ..., dk).  Update every Shampoo factor in place."""
        beta = group["shampoo_beta"]
        nd = gw.dim() - 1
        if nd == 1 and not group["precondition_1d"]:
            return
        for i in range(nd):
            if st["GG"][i] is None:
                continue
            x = gw.movedim(1 + i, 1).reshape(gw.shape[0], gw.shape[1 + i], -1)
            st["GG"][i].lerp_(torch.bmm(x, x.transpose(1, 2)), 1.0 - beta)

    def _refresh(self, st, group, force=False):
        for i, gg in enumerate(st["GG"]):
            if gg is None:
                continue
            if st["Q"][i] is None or force:
                st["Q"][i] = _eigvecs(gg)
                continue
            q = st["Q"][i]
            est = torch.einsum("bij,bik,bkj->bj", q, gg, q)
            idx = est.argsort(dim=-1, descending=True)
            q = torch.gather(q, 2, idx[:, None, :].expand(-1, q.shape[1], -1))
            st["exp_avg_sq"] = _reorder_axis(st["exp_avg_sq"], 1 + i, idx)
            if st.get("rot_avg") is not None:
                st["rot_avg"] = _reorder_axis(st["rot_avg"], 1 + i, idx)
            try:
                qn, _ = torch.linalg.qr(torch.bmm(gg, q))
            except Exception:                            # pragma: no cover
                qn = _eigvecs(gg)
            st["Q"][i] = qn

    def _project(self, x, st, back=False):
        for i, q in enumerate(st["Q"]):
            if q is None:
                continue
            x = _mm_axis(x, q, 1 + i, back=back)
        return x

    # -- step ---------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                B, rest = self._work_shape(p, group)
                nd = len(rest)
                if len(st) == 0:
                    st["step"] = 0
                    rot = group["momentum_space"] == "rot"
                    st["exp_avg"] = None if rot else torch.zeros_like(p)
                    st["rot_avg"] = torch.zeros(B, *rest, device=p.device,
                                                dtype=p.dtype) if rot else None
                    st["exp_avg_sq"] = torch.zeros(B, *rest, device=p.device,
                                                   dtype=p.dtype)
                    mx = group["max_precond_dim"]
                    ok = nd > 1 or group["precondition_1d"]
                    st["GG"] = [
                        torch.zeros(B, d, d, device=p.device, dtype=p.dtype)
                        if (ok and d <= mx and d > 1) else None
                        for d in rest
                    ]
                    st["Q"] = [None] * nd
                    st["warm"] = 0
                    self._accum(st, self._view(g, group, B, rest), group)
                    self._refresh(st, group)
                    continue                       # reference impl skips step 1

                gw = self._view(g, group, B, rest)
                if st["warm"] < group["precond_warmup"]:
                    # An outer-product update has rank <= prod(other dims), so a
                    # factor of dimension d is rank-deficient for the first
                    # ~d/rank steps.  Adam then normalises the (numerical-noise)
                    # components along the null directions up to FULL step size,
                    # which is the one way this algorithm can genuinely misbehave
                    # (measured in lab/test_optim_extra.py check 4).  Warming the
                    # preconditioner up before the first update removes it.
                    st["warm"] += 1
                    self._accum(st, gw, group)
                    self._refresh(st, group, force=True)
                    continue
                gp = self._project(gw, st)
                st["step"] += 1
                k = st["step"]

                st["exp_avg_sq"].mul_(b2).addcmul_(gp, gp, value=1.0 - b2)
                if st["rot_avg"] is not None:
                    # paper Algorithm 1: BOTH moments live in the eigenbasis.
                    # Keeping the first moment in the original space (as the
                    # reference code does) makes the numerator and denominator
                    # two independently-rounded projections of the same tensor;
                    # in near-null directions their float32 cancellation errors
                    # do not agree and m/(sqrt(v)+eps) blows up.  Here the ratio
                    # is bounded exactly as Adam's is.
                    st["rot_avg"].mul_(b1).add_(gp, alpha=1.0 - b1)
                    mp = st["rot_avg"]
                else:
                    st["exp_avg"].mul_(b1).add_(g, alpha=1.0 - b1)
                    mp = self._project(
                        self._view(st["exp_avg"], group, B, rest), st)

                bc1, bc2 = 1.0 - b1 ** k, 1.0 - b2 ** k
                if not group["correct_bias"]:
                    bc1 = bc2 = 1.0
                denom = (st["exp_avg_sq"] / bc2).sqrt_().add_(eps)
                upd = self._project(mp / bc1 / denom, st,
                                    back=True).reshape(p.shape)

                if wd:
                    p.mul_(1.0 - lr * wd)
                p.add_(upd, alpha=-lr)

                self._accum(st, gw, group)
                if k % group["precondition_frequency"] == 0:
                    self._refresh(st, group)
        return loss


# --------------------------------------------------------------------------
# factory used by the probes
# --------------------------------------------------------------------------
def build(name, groups, *, lr, wd=0.0, betas=(0.9, 0.95), batch_dims=0,
          soap_freq=10, soap_max_dim=512, soap_merge=True, soap_beta=0.95,
          soap_mspace="rot", soap_warmup=0,
          ade_alpha=8.0, ade_beta3=0.9999, ade_beta2=0.999, ade_warmup=0):
    """`groups` is anything torch.optim accepts (list of params or of dicts)."""
    name = name.lower()
    if name in ("adamw", "adam"):
        return torch.optim.AdamW(groups, lr=lr, weight_decay=wd, betas=betas)
    if name == "soap":
        return SOAP(groups, lr=lr, weight_decay=wd, betas=betas,
                    shampoo_beta=soap_beta, precondition_frequency=soap_freq,
                    max_precond_dim=soap_max_dim, merge_dims=bool(soap_merge),
                    batch_dims=batch_dims, momentum_space=soap_mspace,
                    precond_warmup=soap_warmup)
    if name == "ademamix":
        return AdEMAMix(groups, lr=lr, weight_decay=wd,
                        betas=(betas[0], ade_beta2, ade_beta3),
                        alpha=ade_alpha, warmup=ade_warmup)
    raise ValueError(f"unknown optimizer {name!r}")
