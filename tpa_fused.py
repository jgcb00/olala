"""Fused Triton kernels for the training path of the Differential-TPA layer.

Ported from the vLLM Olala inference path (jgcb00/vllm, ``layers/mamba/olala``)
and given a backward pass. Around FlashAttention the layer runs about forty
small memory-bound kernels per forward: the rank-``R`` K/V products, the token
shift, the K RMS norm, the Q RMS norm, the scalable-softmax scale and the
differential combine. At 4k tokens they cost about as much as the attention.
Each group becomes one kernel here, with a fused backward that saves only the
kernel's inputs and recomputes the rest:

* ``tpa_kv``          k_t = norm(a_t k_{t-1} + (1 - a_t) A_t B_t / R)   (V: no norm)
* ``tpa_q``           q   = norm(q) * s_h * log(min(pos + 1, W))
* ``tpa_diff_combine`` o  = o_sig - sigmoid(lambda) * o_noise

Everything is computed in fp32 and rounded once on output, so results match the
eager path to bf16 rounding; the eager path rounds after every op.

The decode-time *factor* attention kernel (attention over the rank-``R``
factors without building K/V) is NOT ported: it does about ``R`` times more
QK^T / PV FLOPs than dense attention. That is a good trade at decode, which is
limited by memory bandwidth, and a bad one in training, where FlashAttention is
limited by compute.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------
# K / V: rank-R product + token shift (+ RMS norm for K)
# --------------------------------------------------------------------------


@triton.jit
def _load_b(B, sB, tok, N, d, R: tl.constexpr, D: tl.constexpr):
    """The R rows of B for tokens tok (BT, 1), as R (BT, D) fp32 tiles, zero outside [0, N)."""
    ok = (tok >= 0) & (tok < N)
    base = B + tok.to(tl.int64) * sB + d
    b0 = tl.load(base, mask=ok, other=0.0).to(tl.float32)
    b1 = tl.load(base + D, mask=ok & (R > 1), other=0.0).to(tl.float32)
    b2 = tl.load(base + 2 * D, mask=ok & (R > 2), other=0.0).to(tl.float32)
    b3 = tl.load(base + 3 * D, mask=ok & (R > 3), other=0.0).to(tl.float32)
    return b0, b1, b2, b3


@triton.jit
def _load_a(A, sA, tok, N, h, R: tl.constexpr):
    """A[tok, h, 0..3] as four (BT, 1) fp32 columns (zero past R)."""
    ok = (tok >= 0) & (tok < N)
    base = A + tok.to(tl.int64) * sA + h * R
    a0 = tl.load(base, mask=ok, other=0.0).to(tl.float32)
    a1 = tl.load(base + 1, mask=ok & (R > 1), other=0.0).to(tl.float32)
    a2 = tl.load(base + 2, mask=ok & (R > 2), other=0.0).to(tl.float32)
    a3 = tl.load(base + 3, mask=ok & (R > 3), other=0.0).to(tl.float32)
    return a0, a1, a2, a3


@triton.jit
def _flags(POS, tok, N, L, HAS_POS: tl.constexpr):
    """(is_doc_start, prev_is_zero) for tokens tok of a (rows of length L) batch."""
    row_start = (tok % L) == 0
    if HAS_POS:
        is_doc = tl.load(POS + tok.to(tl.int64), mask=(tok >= 0) & (tok < N), other=1) == 0
    else:
        is_doc = row_start
    return is_doc, is_doc | row_start


@triton.jit
def _gate(LG, sL, tok, N, h, is_doc):
    lg = tl.load(LG + tok.to(tl.int64) * sL + h, mask=(tok >= 0) & (tok < N), other=0.0).to(tl.float32)
    return tl.where(is_doc, 0.0, tl.sigmoid(lg))


# The K/V kernels work on (BT tokens, D) tiles and loop over the H heads: per-head
# products are register FMAs, sums over heads are register adds, and the only
# cross-thread reductions are over D (warp shuffles). R <= 4 (the B rows of
# three consecutive tokens stay in registers).


@triton.jit
def _kv_fwd_kernel(A, sA, B, sB, LG, sL, POS, Y, W, N, L, eps, inv_r,
                   H: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
                   BT: tl.constexpr, HAS_POS: tl.constexpr, NORM: tl.constexpr, ZC: tl.constexpr, NS: tl.constexpr = 1):
    tok = tl.program_id(0) * BT + tl.arange(0, BT)[:, None]
    d = tl.arange(0, D)[None, :]
    valid = tok < N
    t64 = tok.to(tl.int64)
    if NORM:
        g = tl.load(W + d).to(tl.float32)
        if ZC:
            g = 1.0 + g
    b0, b1, b2, b3 = _load_b(B, sB, tok, N, d, R, D)
    p0, p1, p2, p3 = _load_b(B, sB, tok - 1, N, d, R, D)
    is_doc, pz = _flags(POS, tok, N, L, HAS_POS)
    for h in tl.range(0, H, num_stages=NS):
        a0, a1, a2, a3 = _load_a(A, sA, tok, N, h, R)
        q0, q1, q2, q3 = _load_a(A, sA, tok - 1, N, h, R)
        cur = (a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3) * inv_r
        prev = tl.where(pz, 0.0, (q0 * p0 + q1 * p1 + q2 * p2 + q3 * p3) * inv_r)
        a = _gate(LG, sL, tok, N, h, is_doc)
        s = a * prev + (1.0 - a) * cur
        if NORM:
            s = s * tl.rsqrt(tl.sum(s * s, axis=1, keep_dims=True) / D + eps) * g
        tl.store(Y + t64 * (H * D) + h * D + d, s.to(Y.dtype.element_ty), mask=valid)


@triton.jit
def _kv_s_grad(dy, s, g, eps, D: tl.constexpr, NORM: tl.constexpr):
    """(grad wrt the pre-norm s, dy * xhat for the gain)."""
    if NORM:
        inv = tl.rsqrt(tl.sum(s * s, axis=1, keep_dims=True) / D + eps)
        xh = s * inv
        dxh = dy * g
        return inv * (dxh - xh * (tl.sum(dxh * xh, axis=1, keep_dims=True) / D)), dy * xh
    else:
        return dy, dy


@triton.jit
def _kv_bwd_kernel(A, sA, B, sB, LG, sL, POS, W, DY, DA, DB, DLG, DG, N, L, eps, inv_r,
                   H: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
                   BT: tl.constexpr, HAS_POS: tl.constexpr, NORM: tl.constexpr, ZC: tl.constexpr, NS: tl.constexpr = 1):
    pid = tl.program_id(0)
    tok = pid * BT + tl.arange(0, BT)[:, None]
    d = tl.arange(0, D)[None, :]
    valid = tok < N
    nvalid = (tok + 1) < N
    t64 = tok.to(tl.int64)
    if NORM:
        g = tl.load(W + d).to(tl.float32)
        if ZC:
            g = 1.0 + g
    else:
        g = tl.zeros((1, D), dtype=tl.float32)
    m0, m1, m2, m3 = _load_b(B, sB, tok - 1, N, d, R, D)
    b0, b1, b2, b3 = _load_b(B, sB, tok, N, d, R, D)
    n0, n1, n2, n3 = _load_b(B, sB, tok + 1, N, d, R, D)
    doc0, pz0 = _flags(POS, tok, N, L, HAS_POS)
    doc1, pz1 = _flags(POS, tok + 1, N, L, HAS_POS)
    db0 = tl.zeros((BT, D), dtype=tl.float32)
    db1 = tl.zeros((BT, D), dtype=tl.float32)
    db2 = tl.zeros((BT, D), dtype=tl.float32)
    db3 = tl.zeros((BT, D), dtype=tl.float32)
    dg = tl.zeros((BT, D), dtype=tl.float32)
    for h in tl.range(0, H, num_stages=NS):
        a0, a1, a2, a3 = _load_a(A, sA, tok, N, h, R)
        q0, q1, q2, q3 = _load_a(A, sA, tok - 1, N, h, R)
        r0, r1, r2, r3 = _load_a(A, sA, tok + 1, N, h, R)
        cur_m = (q0 * m0 + q1 * m1 + q2 * m2 + q3 * m3) * inv_r
        cur = (a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3) * inv_r
        cur_p = (r0 * n0 + r1 * n1 + r2 * n2 + r3 * n3) * inv_r
        # this token's own s, and the next token's (whose prev is this token)
        prev0 = tl.where(pz0, 0.0, cur_m)
        prev1 = tl.where(pz1, 0.0, cur)
        g0 = _gate(LG, sL, tok, N, h, doc0)
        g1 = _gate(LG, sL, tok + 1, N, h, doc1)
        s0 = g0 * prev0 + (1.0 - g0) * cur
        s1 = g1 * prev1 + (1.0 - g1) * cur_p
        dy0 = tl.load(DY + t64 * (H * D) + h * D + d, mask=valid, other=0.0).to(tl.float32)
        dy1 = tl.load(DY + (t64 + 1) * (H * D) + h * D + d, mask=nvalid, other=0.0).to(tl.float32)
        ds0, dgc = _kv_s_grad(dy0, s0, g, eps, D, NORM)
        ds1, _ = _kv_s_grad(dy1, s1, g, eps, D, NORM)
        if NORM:
            dg += dgc
        da = tl.sum(ds0 * (prev0 - cur), axis=1, keep_dims=True)
        tl.store(DLG + t64 * H + h, tl.where(doc0, 0.0, da * g0 * (1.0 - g0)), mask=valid)
        dcur = ((1.0 - g0) * ds0 + tl.where(pz1, 0.0, g1) * ds1) * inv_r
        da_base = DA + t64 * (H * R) + h * R
        tl.store(da_base, tl.sum(dcur * b0, axis=1, keep_dims=True).to(DA.dtype.element_ty), mask=valid)
        db0 += dcur * a0
        if R > 1:
            tl.store(da_base + 1, tl.sum(dcur * b1, axis=1, keep_dims=True).to(DA.dtype.element_ty), mask=valid)
            db1 += dcur * a1
        if R > 2:
            tl.store(da_base + 2, tl.sum(dcur * b2, axis=1, keep_dims=True).to(DA.dtype.element_ty), mask=valid)
            db2 += dcur * a2
        if R > 3:
            tl.store(da_base + 3, tl.sum(dcur * b3, axis=1, keep_dims=True).to(DA.dtype.element_ty), mask=valid)
            db3 += dcur * a3
    db_base = DB + t64 * (R * D) + d
    tl.store(db_base, db0.to(DB.dtype.element_ty), mask=valid)
    if R > 1:
        tl.store(db_base + D, db1.to(DB.dtype.element_ty), mask=valid)
    if R > 2:
        tl.store(db_base + 2 * D, db2.to(DB.dtype.element_ty), mask=valid)
    if R > 3:
        tl.store(db_base + 3 * D, db3.to(DB.dtype.element_ty), mask=valid)
    if NORM:
        tl.store(DG + pid.to(tl.int64) * D + tl.arange(0, D), tl.sum(dg, axis=0))


def _rows(x: torch.Tensor, n: int) -> tuple[torch.Tensor, int]:
    """View x as n rows with a unit inner stride; returns (tensor, row stride)."""
    x2 = x.reshape(n, -1)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    return x2, x2.stride(0)


_KV_BT, _KV_W, _KV_BTB, _KV_WB, _KV_NS = 8, 4, 4, 2, 3  # fwd tokens/warps, bwd tokens/warps, head-loop stages


class _TPAKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, logit, pos, weight, L, eps, zero_centered):
        # A (..., H, R), B (..., R, D), logit (..., H); pos (N,) or None; weight (D,) or None
        H, R = A.shape[-2:]
        D = B.shape[-1]
        N = A.numel() // (H * R)
        if R > 4:
            raise ValueError(f"tpa_kv supports rank <= 4, got {R}")
        A2, sA = _rows(A, N)
        B2, sB = _rows(B, N)
        L2, sL = _rows(logit, N)
        y = torch.empty(*A.shape[:-2], H, D, device=A.device, dtype=B.dtype)
        norm = weight is not None
        grid = (triton.cdiv(N, _KV_BT),)
        _kv_fwd_kernel[grid](
            A2, sA, B2, sB, L2, sL, pos if pos is not None else A2, y, weight if norm else A2, N, L, eps, 1.0 / R,
            H=H, R=R, D=D, BT=_KV_BT, HAS_POS=pos is not None,
            NORM=norm, ZC=bool(zero_centered), NS=_KV_NS, num_warps=_KV_W,
        )
        ctx.save_for_backward(A2, B2, L2, pos, weight)
        ctx.meta = (A.shape, B.shape, logit.shape, L, eps, bool(zero_centered), A.dtype, logit.dtype)
        return y

    @staticmethod
    def backward(ctx, dy):
        A2, B2, L2, pos, weight = ctx.saved_tensors
        a_shape, b_shape, l_shape, L, eps, zc, a_dtype, l_dtype = ctx.meta
        H, R = a_shape[-2:]
        D = b_shape[-1]
        N = A2.shape[0]
        dy = dy.contiguous()
        norm = weight is not None
        nprog = triton.cdiv(N, _KV_BTB)
        dA = torch.empty(N, H * R, device=dy.device, dtype=a_dtype)
        dB = torch.empty(N, R * D, device=dy.device, dtype=B2.dtype)
        dlg = torch.empty(N, H, device=dy.device, dtype=torch.float32)
        dg = torch.empty(nprog, D, device=dy.device, dtype=torch.float32) if norm else dlg
        _kv_bwd_kernel[(nprog,)](
            A2, A2.stride(0), B2, B2.stride(0), L2, L2.stride(0), pos if pos is not None else A2,
            weight if norm else A2, dy, dA, dB, dlg, dg, N, L, eps, 1.0 / R,
            H=H, R=R, D=D, BT=_KV_BTB, HAS_POS=pos is not None,
            NORM=norm, ZC=zc, NS=_KV_NS, num_warps=_KV_WB,
        )
        dw = dg.sum(0).to(weight.dtype) if norm else None
        return (dA.view(a_shape), dB.view(b_shape), dlg.to(l_dtype).view(l_shape), None, dw, None, None, None)


def tpa_kv(A, B, logit, pos=None, weight=None, *, row_len, eps=1e-6, zero_centered=True):
    """Token-shifted rank-R K or V of a (batch, row_len) layout, flattened.

    A (..., H, R), B (..., R, D), logit (..., H) = pre-sigmoid shift gate.
    pos (N,) int: position ids; a token with pos 0 starts a document (no
    shift). Without pos, only row starts do. weight (D,): K's RMS-norm gain
    (``1 + weight`` if zero_centered); None for V (no norm). Returns
    (..., H, D) in B's dtype.
    """
    return _TPAKV.apply(A, B, logit, pos, weight, row_len, eps, zero_centered)


# --------------------------------------------------------------------------
# Q: RMS norm + scalable-softmax scale
# --------------------------------------------------------------------------


# Q and the differential combine are row-wise: (token, head) rows of D, BR rows
# per program, no padding of the head count.


@triton.jit
def _q_row(Q, sQ, POS, S, rows, N, wsize, eps, H: tl.constexpr, D: tl.constexpr, SCALABLE: tl.constexpr):
    """(q (BR, D) fp32, 1/rms (BR, 1), scale c (BR, 1), log-pos (BR, 1), row mask)."""
    tok = rows // H
    h = rows % H
    m = tok < N
    d = tl.arange(0, D)[None, :]
    q = tl.load(Q + tok.to(tl.int64) * sQ + h * D + d, mask=m, other=0.0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(q * q, axis=1, keep_dims=True) / D + eps)
    if SCALABLE:
        p = tl.load(POS + tok.to(tl.int64), mask=m, other=0).to(tl.float32) + 1.0
        if wsize > 0:
            p = tl.minimum(p, wsize * 1.0)
        lp = tl.log(p)
        c = tl.load(S + h, mask=m, other=0.0).to(tl.float32) * lp
    else:
        lp = tl.full(inv.shape, 1.0, tl.float32)
        c = lp
    return q, inv, c, lp, m


@triton.jit
def _q_fwd_kernel(Q, sQ, POS, S, W, Y, N, wsize, eps,
                  H: tl.constexpr, D: tl.constexpr, BR: tl.constexpr, SCALABLE: tl.constexpr, ZC: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)[:, None]
    d = tl.arange(0, D)[None, :]
    g = tl.load(W + d).to(tl.float32)
    if ZC:
        g = 1.0 + g
    q, inv, c, _, m = _q_row(Q, sQ, POS, S, rows, N, wsize, eps, H, D, SCALABLE)
    tl.store(Y + rows.to(tl.int64) * D + d, (q * (inv * c) * g).to(Y.dtype.element_ty), mask=m)


@triton.jit
def _q_bwd_kernel(Q, sQ, POS, S, W, DY, DQ, DG, DS, N, wsize, eps,
                  H: tl.constexpr, D: tl.constexpr, BR: tl.constexpr, SCALABLE: tl.constexpr, ZC: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)[:, None]
    d = tl.arange(0, D)[None, :]
    g = tl.load(W + d).to(tl.float32)
    if ZC:
        g = 1.0 + g
    q, inv, c, lp, m = _q_row(Q, sQ, POS, S, rows, N, wsize, eps, H, D, SCALABLE)
    dy = tl.load(DY + rows.to(tl.int64) * D + d, mask=m, other=0.0).to(tl.float32)
    xh = q * inv
    dyc = dy * c
    tl.store(DG + pid.to(tl.int64) * D + tl.arange(0, D), tl.sum(dyc * xh, axis=0))
    if SCALABLE:
        tl.store(DS + rows.to(tl.int64), tl.sum(dy * xh * g, axis=1, keep_dims=True) * lp, mask=m)
    dxh = dyc * g
    dq = inv * (dxh - xh * (tl.sum(dxh * xh, axis=1, keep_dims=True) / D))
    tl.store(DQ + rows.to(tl.int64) * D + d, dq.to(DQ.dtype.element_ty), mask=m)


_Q_BR, _Q_W = 16, 2


class _TPAQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, pos, scaler, weight, wsize, eps, zero_centered, out_dtype):
        H, D = q.shape[-2:]
        N = q.numel() // (H * D)
        q2, sQ = _rows(q, N)
        y = torch.empty(*q.shape, device=q.device, dtype=out_dtype)
        scalable = scaler is not None
        _q_fwd_kernel[(triton.cdiv(N * H, _Q_BR),)](
            q2, sQ, pos if scalable else q2, scaler if scalable else q2, weight, y, N, wsize, eps,
            H=H, D=D, BR=_Q_BR, SCALABLE=scalable, ZC=bool(zero_centered), num_warps=_Q_W,
        )
        ctx.save_for_backward(q2, pos, scaler, weight)
        ctx.meta = (q.shape, q.dtype, wsize, eps, bool(zero_centered))
        return y

    @staticmethod
    def backward(ctx, dy):
        q2, pos, scaler, weight = ctx.saved_tensors
        q_shape, q_dtype, wsize, eps, zc = ctx.meta
        H, D = q_shape[-2:]
        N = q2.shape[0]
        dy = dy.contiguous()
        scalable = scaler is not None
        nprog = triton.cdiv(N * H, _Q_BR)
        dq = torch.empty(N, H * D, device=dy.device, dtype=q_dtype)
        dg = torch.empty(nprog, D, device=dy.device, dtype=torch.float32)
        dsc = torch.empty(N, H, device=dy.device, dtype=torch.float32) if scalable else dg
        _q_bwd_kernel[(nprog,)](
            q2, q2.stride(0), pos if scalable else q2, scaler if scalable else q2, weight, dy, dq, dg, dsc,
            N, wsize, eps, H=H, D=D, BR=_Q_BR, SCALABLE=scalable, ZC=zc, num_warps=_Q_W,
        )
        dscaler = dsc.sum(0).to(scaler.dtype) if scalable else None
        return dq.view(q_shape), None, dscaler, dg.sum(0).to(weight.dtype), None, None, None, None


def tpa_q(q, weight, *, pos=None, scaler=None, wsize=0, eps=1e-6, zero_centered=True, out_dtype=torch.bfloat16):
    """RMS-normalized q (..., H, D), times ``scaler[h] * log(min(pos + 1, wsize))``
    when ``scaler`` is given (scalable softmax; wsize <= 0 means no clamp)."""
    return _TPAQ.apply(q, pos, scaler, weight, wsize, eps, zero_centered, out_dtype)


# --------------------------------------------------------------------------
# Differential combine
# --------------------------------------------------------------------------


@triton.jit
def _diff_fwd_kernel(O, LAM, sLam, Y, N, G: tl.constexpr, SNR: tl.constexpr, D: tl.constexpr, BR: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)[:, None]  # (token, group) rows
    d = tl.arange(0, D)[None, :]
    tok = rows // G
    m = tok < N
    r64 = rows.to(tl.int64)
    lam = tl.sigmoid(tl.load(LAM + tok.to(tl.int64) * sLam + rows % G, mask=m, other=0.0).to(tl.float32))
    obase = O + r64 * ((SNR + 1) * D) + d
    noise = tl.load(obase + SNR * D, mask=m, other=0.0).to(tl.float32) * lam
    for j in tl.static_range(SNR):
        sig = tl.load(obase + j * D, mask=m, other=0.0).to(tl.float32)
        tl.store(Y + r64 * (SNR * D) + j * D + d, (sig - noise).to(Y.dtype.element_ty), mask=m)


@triton.jit
def _diff_bwd_kernel(O, LAM, sLam, DY, DO, DLAM, N, G: tl.constexpr, SNR: tl.constexpr, D: tl.constexpr,
                     BR: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)[:, None]
    d = tl.arange(0, D)[None, :]
    tok = rows // G
    m = tok < N
    r64 = rows.to(tl.int64)
    lam = tl.sigmoid(tl.load(LAM + tok.to(tl.int64) * sLam + rows % G, mask=m, other=0.0).to(tl.float32))
    dsum = tl.zeros((BR, D), dtype=tl.float32)
    for j in tl.static_range(SNR):
        dy = tl.load(DY + r64 * (SNR * D) + j * D + d, mask=m, other=0.0)
        tl.store(DO + r64 * ((SNR + 1) * D) + j * D + d, dy.to(DO.dtype.element_ty), mask=m)
        dsum += dy.to(tl.float32)
    noise = tl.load(O + r64 * ((SNR + 1) * D) + SNR * D + d, mask=m, other=0.0).to(tl.float32)
    tl.store(DO + r64 * ((SNR + 1) * D) + SNR * D + d, (-lam * dsum).to(DO.dtype.element_ty), mask=m)
    tl.store(DLAM + r64, -tl.sum(dsum * noise, axis=1, keep_dims=True) * lam * (1.0 - lam), mask=m)


_DIFF_BR, _DIFF_W = 16, 4


class _TPADiff(torch.autograd.Function):
    @staticmethod
    def forward(ctx, o, lam, snr):
        # o (..., G*(snr+1), D) attention output, lam (..., G) pre-sigmoid
        D = o.shape[-1]
        G = o.shape[-2] // (snr + 1)
        N = o.numel() // (o.shape[-2] * D)
        o = o.contiguous()
        lam2, sLam = _rows(lam, N)
        y = torch.empty(*o.shape[:-2], G * snr, D, device=o.device, dtype=o.dtype)
        _diff_fwd_kernel[(triton.cdiv(N * G, _DIFF_BR),)](
            o, lam2, sLam, y, N, G=G, SNR=snr, D=D, BR=_DIFF_BR, num_warps=_DIFF_W,
        )
        ctx.save_for_backward(o, lam2)
        ctx.meta = (snr, lam.shape, lam.dtype)
        return y

    @staticmethod
    def backward(ctx, dy):
        o, lam2 = ctx.saved_tensors
        snr, lam_shape, lam_dtype = ctx.meta
        D = o.shape[-1]
        G = o.shape[-2] // (snr + 1)
        N = lam2.shape[0]
        dy = dy.contiguous()
        do = torch.empty_like(o)
        dlam = torch.empty(N, G, device=o.device, dtype=torch.float32)
        _diff_bwd_kernel[(triton.cdiv(N * G, _DIFF_BR),)](
            o, lam2, lam2.stride(0), dy, do, dlam, N, G=G, SNR=snr, D=D, BR=_DIFF_BR, num_warps=_DIFF_W,
        )
        return do, dlam.to(lam_dtype).view(lam_shape), None


def tpa_diff_combine(o, lam, snr):
    """o (..., G*(snr+1), D) with heads ordered [sig_0..sig_{snr-1}, noise] per
    group, lam (..., G) pre-sigmoid -> (..., G*snr, D) = sig - sigmoid(lam) * noise."""
    return _TPADiff.apply(o, lam, snr)
