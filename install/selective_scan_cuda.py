"""Import-only stub for ``selective_scan_cuda`` (olala inference, torch 2.11).

The real extension targets the torch 2.9 ABI and fails to load under 2.11
(undefined c10::cuda symbols). Dragon's vLLM path never calls selective scan.
Any actual use raises rather than silently computing something wrong.
"""

_MESSAGE = (
    "selective_scan_cuda is a stub in this environment: the real extension "
    "targets the torch 2.9 ABI. Dragon inference does not use selective scan; "
    "rebuild mamba_ssm's CUDA extension against the current torch to use it."
)


def __getattr__(name: str):
    raise NotImplementedError(f"{_MESSAGE} (attempted access: {name!r})")
