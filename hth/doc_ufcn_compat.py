from __future__ import annotations

import importlib


def use_modern_torch_autocast() -> None:
    import torch

    model_module = importlib.import_module("doc_ufcn.model")

    def autocast(enabled=True, dtype=None, cache_enabled=True):
        kwargs = {"enabled": enabled, "cache_enabled": cache_enabled}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return torch.amp.autocast("cuda", **kwargs)

    model_module.autocast = autocast


__all__ = ["use_modern_torch_autocast"]
