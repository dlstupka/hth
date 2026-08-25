from __future__ import annotations

import importlib


def use_modern_torch_autocast() -> bool:
    """Route pinned Doc-UFCN's deprecated autocast alias to the current PyTorch API.

    Doc-UFCN 0.2.0rc4 imports ``autocast`` from ``torch.cuda.amp`` into
    ``doc_ufcn.model``.  PyTorch now deprecates that entry point in favor of
    ``torch.amp.autocast("cuda", ...)``.  Patch the imported module-level alias
    at HTH's integration boundary so upstream inference executes through the
    current API without modifying the installed third-party package.

    Returns True when the alias was replaced.
    """
    import torch

    amp = getattr(torch, "amp", None)
    modern = getattr(amp, "autocast", None)
    if modern is None:
        raise RuntimeError(
            "Doc-UFCN requires a PyTorch runtime exposing torch.amp.autocast"
        )

    model_module = importlib.import_module("doc_ufcn.model")

    def autocast(*args, **kwargs):
        # Doc-UFCN's historical call site does not supply a device type because
        # torch.cuda.amp.autocast implied CUDA. Preserve that meaning explicitly.
        return modern("cuda", *args, **kwargs)

    autocast.__name__ = "autocast"
    autocast.__doc__ = (
        "HTH compatibility alias for torch.amp.autocast('cuda', ...)."
    )
    model_module.autocast = autocast
    return True


__all__ = ["use_modern_torch_autocast"]
