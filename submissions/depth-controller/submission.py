"""depth-controller: a counted-halting depth controller that reaches the top of
the certification ladder from the T values a tier actually provides.

    prompt  --MarkerPointer-->  place-valued digit slots
            --DigitALU x k  -->  candidate answers  out_0 .. out_{k-1}
            --counted halting-->  which iteration is the answer

The controller is the part this file is about.  `lab/reports/depth-controller.md`
measures it in isolation (parser and squaring step held at their construction, a
LAB DIAGNOSTIC that appears nowhere here): with the tier's own T values and
nothing else, it routes **1.000 at every rung of T = 1,2,4,8,16,32,64** at a
modulus whose ladder does not collapse.  Four design decisions carry that, and
each is an ablation in the report:

1.  **Counting, not decoding.**  Every head that maps T's digits to a scalar
    loop count has parameters indexed by (place of T, digit).  Training presents
    T in {1,2,3} or {4,8,16}; the ladder needs {1,2,4,8,16,32,64}, so the
    (place, digit) pairs the ladder needs but training never shows are
    unconstrained.  This controller instead subtracts a learned unit digit from
    T's digit register once per iteration, using the same `Tsub` table and
    borrow state the modular reduction uses, and halts when the register matches
    the ALU's own learned zero digit.  Twelve learned scalars, none of them
    indexed by a place or a digit of T.

2.  **No mass dump.**  The halting weights are `w_k = p_k * prod_{j<k}(1-p_j)`
    and any unspent mass is simply *lost* from the readout.  Piling it on the
    last candidate (the usual PonderNet convenience) makes "never halt" exactly
    correct for the deepest training T, so the deepest training example carries
    no gradient at all -- and it also makes the self-consistency law in (4)
    unsatisfiable.

3.  **A discrete count.**  The digit register is snapped to one-hot with a
    straight-through estimator after every decrement, so a count is a count and
    not a drifting point on the simplex.  Without it the halting statistic
    becomes a continuous quantity that happens to correlate with T -- the same
    value-encoding side channel that a continuous carry vector opens in the
    arithmetic.

4.  **Self-consistency in the loop counter.**  `w(inc r) = shift_right(w(r))`
    holds for the true controller at every register `r`, needs no labels and no
    extra data, and its first component says `p(r) = 0` at every register that
    is a successor.  That is the leak-suppression constraint the cross-entropy
    cannot see: CE through saturated log-probabilities is flat once the argmax
    is right, so a detector that fires 16% of the time on a partly-zero register
    costs nothing at T = 3 and destroys T = 16.  The orbit is built with the
    model's own learned digit increment, so the term is fully differentiable and
    introduces no new tensor.  It is an auxiliary self-supervised regulariser on
    the *loop counter*, not on the arithmetic.

Training and evaluation deliberately differ (README: use `self.training`).
Training runs the soft, differentiable mixture at a small loop count.
Evaluation commits to the mode of the learned halting distribution and stops as
soon as the remaining halting mass can no longer beat the running maximum --
an exact early-exit rule for that readout, and the reason the model fits the
Easy evaluation budget, which `lab/reports/alu-compose.md` measured as a hard
`TimeoutError` for a fixed 64-iteration readout.

Every tensor here is learned from random init.  There is no modular-
exponentiation routine, no digit-multiplication rule, no carry rule and no
lookup of answers.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import (
    ModelSpec,
    OptimizerBundle,
    OptimizerSpec,
    Submission,
    assert_model_state,
)

D_MODEL = 32
REDUCE = 11               # weight-tied conditional subtractions per Horner place
N_CARRY = 2
N_BORROW = 2
TRAIN_LOOPS_EASY = 3      # Easy datasets train on T in {1,2,3}
TRAIN_LOOPS_DEEP = 16     # Medium/Hard train on T in {4,8,16}
EVAL_LOOPS = 64           # the top rung of the certification ladder
DIGIT_TAU = 0.5
HARD_EVAL_STATES = False  # measured: +42% wall clock, zero accuracy gain
IDENTITY_INIT = 5.0
CONS_W = 0.1              # self-consistency weight (report SS6: 0.1 reaches the
#                           top of the ladder, 1.0 over-constrains the anchor)
CONS_J = 8                # how far above the training T the orbit runs
CONS_LOOPS = 4            # halting-chain depth used inside the consistency term
# Halting threshold at init.  0.0 is the SHALLOW init: an untrained detector
# then fires with probability ~0.5-0.6, so an untrained model spends ~2
# iterations per eval batch instead of ~37.  Measured: at the "semantically
# right" init (n_t - 0.5) an untrained model needs 31.7s of the Easy tier's 30s
# eval budget and silently truncates the OOD-N ladder to 2 of 7 rungs.  The
# parameter is trained; training has to earn depth.
SEL_THRESH_INIT = 0.0
LR = 3e-3
SEL_LR = 1e-1
WD = 0.0
O_LO = -2
NEG = -1e4                # finite: torch.finfo(x.dtype).min overflows bf16/amp


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class MarkerPointer(nn.Module):
    """Learned differentiable field-relative pointer.  Four anchors: the [N],
    [X] and [T] marker tokens (located by a learned linear probe) and the last
    valid position of the attention mask.  `R` chooses (anchor, offset) per
    slot; `G` scores cumulative anchor mass, which is the differentiable form of
    a field's *opening* boundary."""

    def __init__(self, d_model: int, n_slot: int, o_hi: int) -> None:
        super().__init__()
        self.o_hi = o_hi
        self.n_off = o_hi - O_LO + 1
        self.probe = nn.Linear(d_model, 3, bias=False)
        self.R = nn.Parameter(torch.randn(n_slot, 4, self.n_off) * 0.5)
        self.G = nn.Parameter(torch.randn(n_slot, 4) * 0.5)

    def forward(self, emb: Tensor, mask: Tensor) -> Tensor:
        b, L, _ = emb.shape
        sc = self.probe(emb).transpose(1, 2)
        sc = sc.masked_fill(~mask[:, None, :], NEG)
        a = F.softmax(sc, dim=-1)
        idx = mask.long().sum(-1) - 1
        a = torch.cat([a, F.one_hot(idx, L).to(a.dtype)[:, None]], dim=1)
        rels = []
        for o in range(O_LO, self.o_hi + 1):
            if o >= 0:
                rels.append(F.pad(a[:, :, o:], (0, o)))
            else:
                rels.append(F.pad(a[:, :, : L + o], (-o, 0)))
        rel = torch.stack(rels, dim=-1)
        logit = (torch.einsum("balo,sao->bsl", rel, self.R)
                 + torch.einsum("bal,sa->bsl", a.cumsum(-1), self.G))
        logit = logit.masked_fill(~mask[:, None, :], NEG)
        return torch.einsum("bsl,bld->bsd", F.softmax(logit, dim=-1), emb)


class DigitALU(nn.Module):
    """One squaring step over digit slots: a Horner recurrence over the places
    of the product, with a shift, a scan order and a gate as the only fixed
    structure.  Every learned tensor is indexed by a digit or by a small
    carry/borrow state; nothing is indexed by the residue."""

    def __init__(self, slots: int) -> None:
        super().__init__()
        self.S = slots
        self.K = 2 * slots - 1
        self.W = slots + 1
        self.Ca, self.Cb = N_CARRY, N_BORROW
        self.Tmul = nn.Parameter(torch.randn(10, 10, 20) * 0.5)
        self.Tadd = nn.Parameter(torch.randn(10, 10, N_CARRY, 10 + N_CARRY) * 0.5)
        self.Tsub = nn.Parameter(torch.randn(10, 10, N_BORROW, 10 + N_BORROW) * 0.5)
        self.zero = nn.Parameter(torch.randn(10) * 0.5)
        self.carry0 = nn.Parameter(torch.randn(N_CARRY) * 0.5)
        self.borrow0 = nn.Parameter(torch.randn(N_BORROW) * 0.5)
        self.gate = nn.Linear(N_BORROW, 1)
        self.copy_scale = nn.Parameter(torch.tensor(IDENTITY_INIT))
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -4.0)
        self.hard = False

    def _sm(self, logits: Tensor) -> Tensor:
        p = F.softmax(logits, -1)
        if self.hard:
            h = F.one_hot(p.argmax(-1), p.shape[-1]).to(p.dtype)
            p = h + p - p.detach()
        return p

    def add_scan(self, r: Tensor, addend: Tensor) -> Tensor:
        c = self._sm(self.carry0).to(r.dtype).expand(r.shape[0], self.Ca)
        outs = []
        for m in range(self.W):
            o = torch.einsum("bu,bv,bc,uvco->bo", r[:, m], addend[:, m], c,
                             self.Tadd.to(r.dtype))
            outs.append(self._sm(o[:, :10] + self.copy_scale * r[:, m]))
            c = self._sm(o[:, 10:] + self.copy_scale * c)
        return torch.stack(outs, 1)

    def cond_sub(self, r: Tensor, ndig: Tensor) -> Tensor:
        c = self._sm(self.borrow0).to(r.dtype).expand(r.shape[0], self.Cb)
        outs = []
        for m in range(self.W):
            o = torch.einsum("bu,bv,bc,uvco->bo", r[:, m], ndig[:, m], c,
                             self.Tsub.to(r.dtype))
            outs.append(self._sm(o[:, :10] + self.copy_scale * r[:, m]))
            c = self._sm(o[:, 10:] + self.copy_scale * c)
        t = torch.stack(outs, 1)
        g = torch.sigmoid(self.gate(c))[:, :, None]
        return g * t + (1 - g) * r

    def forward(self, s: Tensor, ndig: Tensor) -> Tensor:
        b = s.shape[0]
        z = self._sm(self.zero).to(s.dtype).expand(b, 10)
        prod = {}
        for i in range(self.S):
            for j in range(self.S):
                o = torch.einsum("bu,bv,uvo->bo", s[:, i], s[:, j],
                                 self.Tmul.to(s.dtype))
                prod[(i, j)] = (self._sm(o[:, :10]), self._sm(o[:, 10:]))
        r = z[:, None].expand(b, self.W, 10)
        for k in range(self.K - 1, -1, -1):
            r = torch.cat([z[:, None], r[:, : self.W - 1]], dim=1)
            for i in range(self.S):
                j = k - i
                if 0 <= j < self.S:
                    lo, hi = prod[(i, j)]
                    slots = [lo[:, None], hi[:, None]] + [z[:, None]] * (self.W - 2)
                    r = self.add_scan(r, torch.cat(slots, dim=1))
            for _ in range(REDUCE):
                r = self.cond_sub(r, ndig)
        return torch.log(r[:, : self.S] + 1e-9)


def _one_hot_st(r: Tensor) -> Tensor:
    """Straight-through one-hot: the count stays exactly discrete forward, the
    gradient still reaches the learned unit digit."""
    h = F.one_hot(r.argmax(-1), r.shape[-1]).to(r.dtype)
    return h + r - r.detach()


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.vocab = spec.vocab_size
        self.S = max(3, (spec.max_seq_len - 4) // 2)
        self.W = self.S + 1
        self.n_t = 2
        self.train_loops = TRAIN_LOOPS_DEEP
        self.emb = nn.Embedding(spec.vocab_size, D_MODEL)
        self.ptr = MarkerPointer(D_MODEL, self.S + self.W + self.n_t,
                                 o_hi=max(9, self.W + 1))
        self.digit = nn.Linear(D_MODEL, 10)
        self.alu = DigitALU(self.S)
        # --- the depth controller: 12 learned scalars ---
        self.sel_one = nn.Parameter(torch.randn(10) * 0.5)
        self.sel_gain = nn.Parameter(torch.tensor(2.0))
        self.sel_thresh = nn.Parameter(torch.tensor(SEL_THRESH_INIT))
        self.head = nn.Linear(10, spec.vocab_size)
        self.register_buffer("nstep", torch.zeros((), dtype=torch.long),
                             persistent=False)

    # ---------------- the digit register ----------------
    def _shift_reg(self, c: Tensor, table: Tensor, state0: Tensor,
                   n_state: int) -> Tensor:
        """One digit-serial pass over the register with a shared table."""
        alu = self.alu
        one = alu._sm(self.sel_one).to(c.dtype)
        zer = alu._sm(alu.zero).to(c.dtype)
        opnd = torch.stack([one] + [zer] * (self.n_t - 1), 0)[None].expand(
            c.shape[0], self.n_t, 10)
        st = alu._sm(state0).to(c.dtype).expand(c.shape[0], n_state)
        outs = []
        for m in range(self.n_t):
            o = torch.einsum("bu,bv,bc,uvco->bo", c[:, m], opnd[:, m], st,
                             table.to(c.dtype))
            outs.append(alu._sm(o[:, :10]))
            st = alu._sm(o[:, 10:])
        return _one_hot_st(torch.stack(outs, 1))

    def decrement(self, c: Tensor) -> Tensor:
        return self._shift_reg(c, self.alu.Tsub, self.alu.borrow0, self.alu.Cb)

    def increment(self, c: Tensor) -> Tensor:
        return self._shift_reg(c, self.alu.Tadd, self.alu.carry0, self.alu.Ca)

    def is_zero(self, c: Tensor) -> Tensor:
        z = self.alu._sm(self.alu.zero).to(c.dtype)
        m = torch.einsum("btd,d->b", c, z)[:, None]
        return torch.sigmoid(self.sel_gain.to(c.dtype)
                             * (m - self.sel_thresh.to(c.dtype)))

    def halting(self, c: Tensor, loops: int) -> Tensor:
        """The PonderNet marginal, WITHOUT dumping unspent mass."""
        rest = self.is_zero(c).new_ones(c.shape[0], 1)
        ws = []
        for _ in range(loops):
            c = self.decrement(c)
            p = self.is_zero(c)
            ws.append(rest * p)
            rest = rest * (1 - p)
        return torch.cat(ws, dim=-1)

    def consistency(self, c: Tensor) -> Tensor:
        """`w(inc r) == shift_right(w(r))` along the increment orbit of the T
        register.  No labels, no data: it is a law the true controller obeys at
        every register, and it supervises the counter at loop counts the tier
        never presents."""
        r = c[:1]                      # the register depends only on T
        w = self.halting(r, CONS_LOOPS)
        tot = w.new_zeros(())
        for _ in range(CONS_J):
            r = self.increment(r)
            w_next = self.halting(r, CONS_LOOPS)
            tgt = torch.cat([torch.zeros_like(w[:, :1]), w[:, :-1]], dim=1)
            tot = tot + ((w_next - tgt.detach()) ** 2).sum(-1).mean()
            w = w_next
        return tot / CONS_J

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None):
        b, L = input_ids.shape
        if attention_mask is None:
            mask = torch.ones(b, L, dtype=torch.bool, device=input_ids.device)
        else:
            mask = attention_mask.to(torch.bool)
            if mask.dim() == 3:
                mask = mask[:, 0]
        e = self.emb(input_ids)
        slots = self.ptr(e, mask)
        dg = F.softmax(self.digit(slots) / DIGIT_TAU, dim=-1)
        sx = dg[:, : self.S]
        sn = dg[:, self.S: self.S + self.W]
        st = dg[:, self.S + self.W:]

        if self.training:
            self.nstep += 1
            self.alu.hard = False
            loops = self.train_loops
        else:
            self.alu.hard = HARD_EVAL_STATES
            loops = EVAL_LOOPS

        state, c = sx, st[:, : self.n_t]
        rest = self.is_zero(c).new_ones(b, 1)
        acc = None
        best_w = None
        best_out = None
        for k in range(loops):
            out = self.alu(state, sn)
            state = F.softmax(out, dim=-1)
            c = self.decrement(c)
            p = self.is_zero(c)
            w = (rest * p).to(out.dtype)
            rest = rest * (1 - p)
            if self.training:
                acc = w[:, :, None] * out if acc is None else acc + w[:, :, None] * out
            else:
                # commit to the MODE of the learned halting distribution
                if best_w is None:
                    best_w, best_out = w, out
                else:
                    take = (w > best_w).to(out.dtype)
                    best_w = torch.maximum(best_w, w)
                    best_out = take[:, :, None] * out + (1 - take[:, :, None]) * best_out
                # exact early exit: once the remaining halting mass cannot beat
                # the running maximum, no later iteration can win the argmax.
                # Checked every 4th iteration so the host/device sync is not
                # itself the cost on a kernel-launch-bound model.
                if (k & 3) == 3 and float(
                        (rest.to(best_w.dtype) - best_w).max()) <= 0.0:
                    break
        mixed = acc if self.training else best_out

        vals = self.head(mixed)
        logits = input_ids.new_zeros(b, L, self.vocab, dtype=vals.dtype)
        last = mask.long().sum(-1) - 1
        for p_i in range(self.S):
            pos = (last - p_i).clamp(min=0)
            logits.scatter_(1, pos[:, None, None].expand(-1, 1, self.vocab),
                            vals[:, p_i][:, None, :])
        aux = self.consistency(st[:, : self.n_t]) if self.training else None
        return logits, aux


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def training_loss(logits: Tensor, labels: Tensor, aux) -> Tensor:
    loss = F.cross_entropy(logits.float(), labels)
    if aux is not None:
        loss = loss + CONS_W * aux.float()
    return loss


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    # `training_time_seconds` is public, data-independent manifest information.
    if hasattr(model, "train_loops"):
        model.train_loops = (TRAIN_LOOPS_EASY if spec.training_time_seconds <= 120
                             else TRAIN_LOOPS_DEEP)
    sel = {id(p) for n, p in model.named_parameters() if n.startswith("sel_")}
    decay = [p for p in model.parameters() if p.ndim >= 2 and id(p) not in sel]
    no_decay = [p for p in model.parameters() if p.ndim < 2 and id(p) not in sel]
    sel_params = [p for n, p in model.named_parameters() if n.startswith("sel_")]
    return OptimizerBundle(
        torch.optim.AdamW(
            [{"params": decay, "weight_decay": WD},
             {"params": no_decay, "weight_decay": 0.0},
             # the halting gain has to travel O(10) to make the detector sharp;
             # at the body's learning rate that alone is thousands of steps
             {"params": sel_params, "weight_decay": 0.0, "lr": SEL_LR}],
            lr=LR, betas=(0.9, 0.95),
            capturable=spec.device_type == "cuda",
        )
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    training_loss=training_loss,
    batch_size=128,
    # kernel-launch bound, not compute bound: 8x the eval batch costs 5% more
    # wall clock, and a large eval batch makes every scoring split one batch.
    eval_batch_size=4096,
    max_steps=None,
)
