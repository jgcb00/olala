"""Parity tests for tpa_fused.py (CUDA only).

    python -m pytest test_tpa_fused.py      # or: python test_tpa_fused.py

1. Each kernel's forward and gradients against an fp32 reference of the eager
   formulas in modeling_olala.py, in fp32 (tight) and bf16 (rounding-level).
2. The Differential-TPA layer, fused against eager, both against the same layer
   with fp32 parameters: the fused path must be at least as close as the eager
   one for the output and every gradient.
"""

import copy
import os
import sys

import torch

try:
    import pytest
except ImportError:  # plain `python test_tpa_fused.py` works without it
    pytest = None
import torch.nn.functional as F

if __package__ in (None, ""):  # run as a script from the checkout
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    import importlib

    importlib.import_module(__package__)

if pytest is not None:
    pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _parametrize(names, cases):
    return pytest.mark.parametrize(names, cases) if pytest is not None else (lambda f: f)

from .tpa_fused import tpa_diff_combine, tpa_kv, tpa_q  # noqa: E402


def _ref_kv(A, B, lg, pos, w, eps):
    b, L, H, R = A.shape
    kv = torch.einsum("blhr,blrd->blhd", A, B) / R
    a = torch.sigmoid(lg).unsqueeze(-1)
    prev = F.pad(kv, (0, 0, 0, 0, 1, 0))[:, :-1]
    if pos is not None:
        m = pos == 0
    else:
        m = torch.zeros(b, L, dtype=torch.bool, device=A.device)
        m[:, 0] = True
    m = m[..., None, None]
    s = a.masked_fill(m, 0) * prev.masked_fill(m, 0) + (1 - a.masked_fill(m, 0)) * kv
    if w is not None:
        s = s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps) * (1 + w)
    return s


def _ref_q(q, w, pos, sc, wsize, eps):
    y = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + eps) * (1 + w)
    if sc is not None:
        p = pos.float()[..., None, None] + 1
        y = sc.view(1, 1, -1, 1) * (torch.clamp_max(p, wsize) if wsize > 0 else p).log() * y
    return y


def _ref_diff(o, lam, snr):
    b, L, hq, d = o.shape
    o = o.view(b, L, hq // (snr + 1), snr + 1, d)
    return (o[:, :, :, :snr] - torch.sigmoid(lam)[..., None, None] * o[:, :, :, snr:]).reshape(b, L, -1, d)


def _check(got, want, tol, what):
    err = (got.float() - want.float()).abs().max().item()
    scale = want.float().abs().max().item() + 1e-12
    assert err / scale < tol, f"{what}: max err {err:.3e} (rel {err / scale:.2e})"


CASES = [  # (batch, row length, docs per row, use position ids, dtype, tolerance)
    (1, 1000, 3, True, torch.float32, 1e-5),
    (2, 517, 1, False, torch.float32, 1e-5),
    (2, 4096, 5, True, torch.bfloat16, 2e-2),
]


@_parametrize("b,L,ndoc,use_pos,dtype,tol", CASES)
def test_kernels(b, L, ndoc, use_pos, dtype, tol):
    torch.manual_seed(0)
    dev = "cuda"
    H, R, D = 12, 4, 128

    def mk(*shape):
        return torch.randn(*shape, device=dev).to(dtype).requires_grad_()

    def ref_leaf(t):
        return t.detach().float().requires_grad_()

    pos = None
    if use_pos:
        pos = (torch.arange(L, device=dev) % (L // ndoc + 1)).repeat(b, 1)
        pos[1:, 0] = 7  # a row start that is not a document start

    A, B, lg = mk(b, L, H, R), mk(b, L, R, D), mk(b, L, H)
    w = (0.1 * torch.randn(D, device=dev)).requires_grad_()
    for norm in (True, False):
        y = tpa_kv(A, B, lg, None if pos is None else pos.flatten(), w if norm else None, row_len=L)
        Ar, Br, lgr = ref_leaf(A), ref_leaf(B), ref_leaf(lg)
        wr = w.detach().clone().requires_grad_()
        yr = _ref_kv(Ar, Br, lgr, pos, wr if norm else None, 1e-6)
        _check(y, yr, tol, f"kv(norm={norm})")
        g = torch.randn_like(yr)
        y.backward(g.to(y.dtype))
        yr.backward(g)
        for name, t, tr in (("A", A, Ar), ("B", B, Br), ("logit", lg, lgr)):
            _check(t.grad, tr.grad, tol, f"kv(norm={norm}) d{name}")
            t.grad = None
        if norm:
            _check(w.grad, wr.grad, tol, "kv dweight")
            w.grad = None

    q = mk(b, L, 48, D)
    wq = (0.1 * torch.randn(D, device=dev)).requires_grad_()
    sc = torch.rand(48, device=dev).requires_grad_()
    p = pos if pos is not None else torch.arange(L, device=dev).repeat(b, 1)
    for wsize in (300, 0):
        y = tpa_q(q, wq, pos=p.flatten(), scaler=sc, wsize=wsize, out_dtype=torch.float32)
        qr, wr, sr = ref_leaf(q), wq.detach().clone().requires_grad_(), sc.detach().clone().requires_grad_()
        yr = _ref_q(qr, wr, p, sr, wsize, 1e-6)
        _check(y, yr, tol, f"q(wsize={wsize})")
        g = torch.randn_like(yr)
        y.backward(g)
        yr.backward(g)
        _check(q.grad, qr.grad, tol, "q dq")
        _check(wq.grad, wr.grad, tol, "q dweight")
        _check(sc.grad, sr.grad, tol, "q dscaler")
        q.grad = wq.grad = sc.grad = None
    _check(tpa_q(q, wq, out_dtype=torch.float32), _ref_q(q.detach().float(), wq.detach(), None, None, 0, 1e-6),
           tol, "q (no scalable softmax)")

    o, lam = mk(b, L, 48, D), mk(b, L, 12)
    y = tpa_diff_combine(o, lam, 3)
    orr, lr = ref_leaf(o), ref_leaf(lam)
    yr = _ref_diff(orr, lr, 3)
    _check(y, yr, tol, "diff")
    g = torch.randn_like(yr)
    y.backward(g.to(y.dtype))
    yr.backward(g)
    _check(o.grad, orr.grad, tol, "diff do")
    _check(lam.grad, lr.grad, tol, "diff dlambda")


def test_layer_fused_vs_eager():
    from . import modeling_olala as mo
    from .configuration_olala import OlalaConfig

    if mo.ATTN_IMPL not in ("fa2", "fa3"):
        if pytest is not None:
            pytest.skip("the fused path needs flash-attn")
        print("skipped: the fused path needs flash-attn")
        return
    cfg = OlalaConfig(
        hidden_size=1536, num_attention_heads=48, num_signal_heads_diff=36, head_dim=128, tpa_rank=4,
        qk_norm=True, token_shift_attn=True, token_conv1d_attn=False, scalable_softmax=True,
        scalar_proj_as_hidden_matrix=True, zero_centered_gamma=True, rope_theta=0.0, rope_type="",
        slw_wsize=1024, softcap_attn=150.0, use_completed_p=True, norm_epsilon=1e-6,
    )
    torch.manual_seed(0)
    S = 4096
    l32 = mo.OlalaDifferentialTensorProductAttentionV2(cfg, 4).cuda()
    for prm in l32.parameters():
        prm.data.normal_(0, 0.02)
    for prm in (l32.q_norm.norm.weight, l32.k_norm.norm.weight):
        prm.data.normal_(0, 0.1)
    l32.softmax_scaler.data.uniform_(0.3, 1.0)
    l16 = copy.deepcopy(l32).bfloat16()
    x32 = torch.randn(1, S, cfg.hidden_size, device="cuda").requires_grad_()
    x16 = x32.detach().bfloat16().requires_grad_()
    pos = (torch.arange(S, device="cuda") % 1400).unsqueeze(0)
    g = torch.randn(1, S, 36, 128, device="cuda")

    def run(layer, x, fused):
        os.environ["OLALA_TPA_FUSED"] = "1" if fused else "0"
        try:
            assert layer._use_fused(x, pos, None, None) == fused
            out = layer(x, position_ids=pos)[0]
            out.float().backward(g)
        finally:
            os.environ.pop("OLALA_TPA_FUSED", None)
        res = {n: prm.grad.float().clone() for n, prm in layer.named_parameters()}
        res["x"], res["out"] = x.grad.float().clone(), out.float().detach()
        layer.zero_grad(set_to_none=True)
        x.grad = None
        return res

    ref = run(l32, x32, False)
    eager, fused = run(l16, x16, False), run(l16, x16, True)
    for k in ref:
        e = ((eager[k] - ref[k]).norm() / ref[k].norm()).item()
        f = ((fused[k] - ref[k]).norm() / ref[k].norm()).item()
        assert f < max(1.1 * e, 1e-3), f"{k}: fused rel err {f:.2e} vs eager {e:.2e}"


if __name__ == "__main__":
    for case in CASES:
        test_kernels(*case)
        print("kernels ok", case[:5])
    test_layer_fused_vs_eager()
    print("layer ok")
