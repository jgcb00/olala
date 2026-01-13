import re
from collections import defaultdict, OrderedDict
from typing import Callable, Iterable, Optional, Any, Dict, Tuple, List, Union

import torch
import matplotlib.pyplot as plt

LayerKey = Union[int, str]

def default_layer_key_from_name(name: str) -> LayerKey:
    """
    Best-effort layer index extractor from parameter names.
    Matches common patterns: layers.{i}.
    """
    m = re.search(r"(?:^|\.)(?:layers)\.(\d+)\.", name)
    if m:
        return int(m.group(1))
    return "other"

def default_param_filter(name: str, p: torch.nn.Parameter) -> bool:
    """
    Default: track matrix-like weights, skip biases & norm parameters.
    """
    if not p.requires_grad:
        return False
    if p.ndim < 2:
        return False
    lname = name.lower()
    if "norm" in lname or "layernorm" in lname or "rmsnorm" in lname or ".ln" in lname:
        return False
    return True

@torch.no_grad()
def _snapshot_params(
    named_params: List[Tuple[str, torch.nn.Parameter]],
    param_filter: Callable[[str, torch.nn.Parameter], bool],
) -> Dict[str, torch.Tensor]:
    snap = {}
    for n, p in named_params:
        if param_filter(n, p):
            snap[n] = p.detach().clone()
    return snap


def width_update_norm_check(
    d_models: List[int],
    model_factory: Callable[[int], torch.nn.Module],
    optimizer_factory: Callable[[Iterable[torch.nn.Parameter], int], torch.optim.Optimizer],
    *,
    vocab_size: int = 50304,
    batch_size: int = 4,
    seq_len: int = 8192,
    steps: int = 4,
    seed: int = 0,
    layer_key_fn: Callable[[str], LayerKey] = default_layer_key_from_name,
    param_filter: Callable[[str, torch.nn.Parameter], bool] = default_param_filter,
    max_layers_to_plot: Optional[int] = None,
    x_log2: bool = False,
) -> Tuple[plt.Figure, Dict[int, Dict[LayerKey, List[float]]]]:
    """
    Runs `steps` optimizer updates for each width in `d_models`.
    Records per-layer ||ΔW_l||_2 for each step, then plots:
      - subplots: iteration 0..steps-1
      - x: d_model
      - y: per-layer update norm (log scale)

    Returns (fig, results) where:
      results[iter][layer_key] = list aligned with d_models
    """

    def train_step(model: torch.nn.Module) -> torch.Tensor:
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")
        targets   = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")
        loss = model(input_ids=input_ids, labels=targets).loss
        return loss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: Dict[int, Dict[LayerKey, List[float]]] = {i: defaultdict(list) for i in range(steps)}
    for d in d_models:
        torch.manual_seed(seed)  # reset so comparisons across widths are less noisy
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        model = model_factory(d).to(device)
        model.train()
        opt = optimizer_factory(model.parameters(), d)

        named_params = list(model.named_parameters())

        for it in range(steps):
            # snapshot BEFORE update
            before = _snapshot_params(named_params, param_filter)

            opt.zero_grad(set_to_none=True)
            loss = train_step(model)
            loss.backward()
            opt.step()

            # compute per-layer ||ΔW||_2
            per_layer_sq = defaultdict(float)
            for n, p in named_params:
                if n not in before:
                    continue
                dp = (p.detach() - before[n]).float()
                k = layer_key_fn(n)
                per_layer_sq[k] += float(dp.pow(2).sum().item())

            # stable ordering: numeric layers first
            layer_items = sorted(
                per_layer_sq.items(),
                key=lambda kv: (0, kv[0]) if isinstance(kv[0], int) else (1, str(kv[0])),
            )
            for k, s in layer_items:
                results[it][k].append((s ** 0.5) + 1e-30)  # avoid log(0)

        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # optionally limit number of plotted layers (when many)
    all_keys = set()
    for it in range(steps):
        all_keys |= set(results[it].keys())
    ordered_keys = sorted(all_keys, key=lambda k: (0, k) if isinstance(k, int) else (1, str(k)))
    if max_layers_to_plot is not None:
        ordered_keys = ordered_keys[:max_layers_to_plot]

    fig, axes = plt.subplots(1, steps, figsize=(4.2 * steps, 3.6), sharey=True)
    if steps == 1:
        axes = [axes]

    for it, ax in enumerate(axes):
        for k in ordered_keys:
            if k in results[it]:
                ax.plot(d_models, results[it][k], alpha=0.7)
        ax.set_title(f"Iteration {it}")
        ax.set_xlabel("d_model")
        ax.set_yscale("log")
        if x_log2:
            ax.set_xscale("log", base=2)
        ax.grid(True, which="both", alpha=0.3)

    axes[0].set_ylabel(r"$\|\Delta W_\ell\|_2$")
    fig.tight_layout()
    return fig, results
