# ademamixh.py
"""
AdEMAMixH: AdEMAMix with AdamH-style norm-preserving / projection step (2D/3D params).

We first compute the AdEMAMix update direction u_t (without applying weight decay into u_t),
then apply the "H" update:

    R  = ||W||          (Frobenius norm for 2D; per first-dim slice for 3D)
    û  = u / ||u||
    W' = W - lr * R * û
    W_{t+1} = R * Normalize(W')

So norms are preserved.
"""

import math
import torch
from torch.optim import Optimizer


def linear_warmup_scheduler(step, alpha_end, alpha_start=0.0, warmup=1):
    if step < warmup:
        a = step / float(warmup)
        return (1.0 - a) * alpha_start + a * alpha_end
    return alpha_end


def linear_hl_warmup_scheduler(step, beta_end, beta_start=0.0, warmup=1):
    # warm up in "half-life space"
    def f(beta, eps=1e-8):
        return math.log(0.5) / math.log(beta + eps) - 1

    def f_inv(t):
        return math.pow(0.5, 1.0 / (t + 1.0))

    if step < warmup:
        a = step / float(warmup)
        return f_inv((1.0 - a) * f(beta_start) + a * f(beta_end))
    return beta_end


class AdEMAMixH(Optimizer):
    r"""Implements the AdEMAMixH algorithm (AdEMAMix + H projection).

    Arguments:
        params (iterable): parameters to optimize or dicts defining parameter groups
        lr (float): learning rate (default: 1e-3)
        betas (Tuple[float, float, float]): (beta1, beta2, beta3) (default: (0.9, 0.95, 0.999))
        alpha (float): mixing coefficient for slow EMA (default: 8.0)
        beta3_warmup (int, optional): warmup steps to increase beta3 (default: None)
        alpha_warmup (int, optional): warmup steps to increase alpha (default: None)
        eps (float): numerical stability term (default: 1e-8)
        normalize_alpha (bool): if True, scale denom by (1+alpha) (default: False)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.95, 0.999),
        alpha=8.0,
        beta3_warmup=None,
        alpha_warmup=None,
        eps=1e-8,
        normalize_alpha=False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not (0.0 <= betas[0] < 1.0):
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not (0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not (0.0 <= betas[2] < 1.0):
            raise ValueError(f"Invalid beta parameter at index 2: {betas[2]}")
        if alpha < 0.0:
            raise ValueError(f"Invalid alpha value: {alpha}")

        self.normalize_alpha = normalize_alpha
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            alpha=alpha,
            beta3_warmup=beta3_warmup,
            alpha_warmup=alpha_warmup,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            beta1, beta2, beta3_final = group["betas"]
            beta3_warmup = group["beta3_warmup"]
            alpha_final = group["alpha"]
            alpha_warmup = group["alpha_warmup"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if not torch.is_floating_point(p):
                    continue

                # H variant as requested: 2D or 3D only
                if p.ndim not in (2, 3):
                    raise AssertionError(
                        f"AdEMAMixH expects 2D or 3D params, got shape={tuple(p.shape)}"
                    )

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdEMAMixH does not support sparse gradients.")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    if beta1 != 0.0:
                        state["exp_avg_fast"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                    else:
                        state["exp_avg_fast"] = None
                    state["exp_avg_slow"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg_fast = state["exp_avg_fast"]
                exp_avg_slow = state["exp_avg_slow"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                step = state["step"]

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step

                # warmups
                if alpha_warmup is not None:
                    alpha = linear_warmup_scheduler(
                        step, alpha_end=alpha_final, alpha_start=0.0, warmup=alpha_warmup
                    )
                else:
                    alpha = alpha_final

                if beta3_warmup is not None:
                    beta3 = linear_hl_warmup_scheduler(
                        step, beta_end=beta3_final, beta_start=beta1, warmup=beta3_warmup
                    )
                else:
                    beta3 = beta3_final

                # moments
                if beta1 != 0.0:
                    exp_avg_fast.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    fast = exp_avg_fast
                else:
                    fast = grad  # no buffer

                exp_avg_slow.mul_(beta3).add_(grad, alpha=1.0 - beta3)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                if self.normalize_alpha:
                    denom = denom * (1.0 + alpha)

                # AdEMAMix direction (same as your reference implementation)
                u = (fast.div(bias_correction1) + alpha * exp_avg_slow) / denom

                # H step: norm-scaled step then project back to target norm
                if p.ndim == 2:
                    p_norm = p.norm()

                    u_norm = u.norm().clamp_min(1e-10)
                    new_p = p - lr * u * (p_norm / u_norm)

                    new_p_norm = new_p.norm().clamp_min(1e-10)
                    p.copy_(new_p / new_p_norm * p_norm)
                else:
                    axes = tuple(range(1, p.ndim))  # preserve norm per first dim
                    p_norm = torch.sqrt(torch.sum(p * p, dim=axes, keepdim=True))

                    u_norm = torch.sqrt(torch.sum(u * u, dim=axes, keepdim=True)).clamp_min(
                        1e-10
                    )
                    new_p = p - lr * u * (p_norm / u_norm)

                    new_p_norm = torch.sqrt(
                        torch.sum(new_p * new_p, dim=axes, keepdim=True)
                    ).clamp_min(1e-10)
                    p.copy_(new_p / new_p_norm * p_norm)

        return loss

if __name__ == "__main__":
    # small dummy test
    torch.manual_seed(0)
    x = torch.randn((10, 7))
    model = torch.nn.Linear(7, 1, bias=False)

    opt = AdEMAMixH(
        params=model.parameters(),
        lr=1e-2,
        betas=(0.9, 0.999, 0.9999),
        alpha=2.0,
        beta3_warmup=45,
        alpha_warmup=45,
    )

    print("init ||W||:", model.weight.data.norm().item())
    for itr in range(50):
        y = model(x).mean()
        opt.zero_grad()
        y.backward()
        opt.step()
    print("final ||W||:", model.weight.data.norm().item())
