import torch
from torch.optim import Optimizer


class AdamH(Optimizer):
    """
    AdamH for 2D params only (e.g. Linear weights).

    u = Adam direction
    new_p = p - lr * u * ||p|| / ||u||
    p <- new_p / ||new_p|| * ||p||
    """

    def __init__(self, params, lr: float, betas=(0.9, 0.95), eps: float = 1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if not torch.is_floating_point(p):
                    continue
                assert p.ndim == 2 or p.ndim == 3, f"AdamH expects 2D or 3D params, got shape={tuple(p.shape)}"

                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # moments
                exp_avg.mul_(b1).add_(g, alpha=1 - b1)
                exp_avg_sq.mul_(b2).addcmul_(g, g, value=1 - b2)

                # bias correction
                bc1 = 1.0 - (b1 ** t)
                bc2 = 1.0 - (b2 ** t)

                m_hat = exp_avg / bc1
                v_hat = exp_avg_sq / bc2
                u = m_hat / (v_hat.sqrt().add_(eps))  # Adam direction

                # projected / norm-preserving step (Frobenius norm)
                if p.ndim == 2:
                    p_norm = p.norm()
                    u_norm = u.norm().clamp_min(1e-10)
                    new_p = p - lr * u * (p_norm / u_norm)
                    p.copy_(new_p / new_p.norm().clamp_min(1e-10) * p_norm)
                else:
                    axes = tuple(range(1, p.ndim))  # preserve norm per first dim (e.g. per expert)
                    p_norm = torch.sqrt(torch.sum(p * p, dim=axes, keepdim=True))
                    u_norm = torch.sqrt(torch.sum(u * u, dim=axes, keepdim=True)).clamp_min(1e-10)
                    new_p = p - lr * u * (p_norm / u_norm)
                    new_p_norm = torch.sqrt(torch.sum(new_p * new_p, dim=axes, keepdim=True)).clamp_min(1e-10)
                    p.copy_(new_p / new_p_norm * p_norm)

        return loss