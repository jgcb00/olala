import torch
import torch.nn as nn



class Snoo:
    """
    @DominikKallusky, @vishal9-team, @vinaysrao
    Sparse Nesterov Outer Optimizer (Snoo) is a momentum-based wrapper to any optimizer that can
    improve the stability and smoothness of the optimization process and thus the quality
    of large language models (LLM) and other models. Snoo implicitly adds temporal regularization
    to the parameters, thus smoothing the training trajectory and instilling a bias towards flatter
    minima and lower parameter norms. Snoo is computationally efficient, incurring minimal overhead
    in compute and moderate memory usage.
    """

    @torch.no_grad()
    def __init__(self, model: nn.Module, lr: float, momentum: float, k: int) -> None:
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.k = k
        self.current_step = 0
        self.outer_buf = [p.clone() for p in model.parameters()]
        self.model_params = list(self.model.parameters())
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
            nesterov=True,
            fused=True,
        )

    @torch.no_grad()
    def step(
        self,
    ) -> None:
        if self.current_step % self.k == 0:
            for p_new, p_old in zip(self.model_params, self.outer_buf):
                p_new.grad = p_old.data - p_new.data
                p_new.copy_(p_old, non_blocking=True)

            self.optimizer.step()

            for p_new, p_old in zip(self.model_params, self.outer_buf):
                p_old.copy_(p_new, non_blocking=True)
        self.current_step += 1

    def state_dict(self):
        state_dict = {
            "current_step": self.current_step,
            "lr": self.lr,
            "momentum": self.momentum,
            "k": self.k,
            "outer_buf": [p.clone() for p in self.outer_buf],
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.current_step = state_dict["current_step"]
        self.lr = state_dict["lr"]
        self.momentum = state_dict["momentum"]
        self.k = state_dict["k"]
        for p_src, p_dst in zip(state_dict["outer_buf"], self.outer_buf):
            p_dst.copy_(p_src)
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
