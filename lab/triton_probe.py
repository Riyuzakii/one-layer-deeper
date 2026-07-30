#!/usr/bin/env python
"""Does a hand-written Triton kernel actually *run* on this box?

BRIEF2 §5 says `triton` 3.7.1 is importable and therefore "custom chunkwise/fused
kernels are on the table".  Importable is not the same as usable; this probe compiles
and launches the simplest possible kernel and reports the outcome.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(o_ptr + off, tl.load(x_ptr + off, mask=m) + tl.load(y_ptr + off, mask=m), mask=m)


def main() -> int:
    print("triton", triton.__version__, "torch", torch.__version__)
    print("device capability", torch.cuda.get_device_capability())
    x = torch.randn(4096, device="cuda")
    y = torch.randn(4096, device="cuda")
    o = torch.empty_like(x)
    try:
        _add_kernel[(4,)](x, y, o, 4096, BLOCK=1024)
        torch.cuda.synchronize()
        print("TRITON KERNEL: OK, correct =", bool(torch.allclose(o, x + y)))
        return 0
    except Exception as exc:  # noqa: BLE001
        print("TRITON KERNEL: FAILED", type(exc).__name__)
        print(str(exc)[:800])
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
