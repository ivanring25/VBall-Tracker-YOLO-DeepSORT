"""Shared inference-runtime helpers for the YOLO detectors.

Two cheap wins the pipeline was not taking:

  * ``torch.inference_mode()`` — skips autograd bookkeeping entirely. The
    models are only ever used for inference, so the graph machinery was pure
    overhead.
  * fp16 on CUDA — roughly halves memory traffic on the GPU. Enabled only on
    CUDA; on CPU fp16 is usually *slower*, so it stays off there.

Also a warmup pass: the first inference pays lazy CUDA kernel/cuDNN
autotuning, which otherwise shows up as a multi-second stall on frame one.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import numpy as np

from vtracker.core.logging import get_logger

_log = get_logger("vtracker.runtime")

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch optional for the light install
    _HAS_TORCH = False


@contextlib.contextmanager
def inference_context() -> Iterator[None]:
    """No-grad inference, falling back cleanly when torch is absent."""
    if not _HAS_TORCH:
        yield
        return
    with torch.inference_mode():
        yield


def use_half(device: str) -> bool:
    """fp16 is a win on CUDA and usually a loss on CPU."""
    return device.startswith("cuda")


def warmup(model, device: str, size: tuple[int, int], *, batch: int = 1) -> None:
    """Run one throwaway inference so the first real frame isn't the one that
    pays for kernel autotuning."""
    try:
        w, h = size
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        source = [blank] * batch if batch > 1 else blank
        with inference_context():
            model.predict(source=source, device=device, verbose=False,
                          half=use_half(device))
        _log.info("model warmed up on %s", device)
    except Exception:
        # Warmup is an optimisation; never let it stop a run.
        _log.debug("warmup skipped", exc_info=True)
