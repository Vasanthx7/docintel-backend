"""Device / precision detection shared by the embedding and reranker models.

FlagEmbedding auto-moves models to CUDA when torch sees a GPU. The only knob
we need is fp16: it's ~2x faster and halves VRAM on GPU, but on CPU it must
stay off (fp16 on CPU is slower and can be numerically wrong).
"""
from functools import lru_cache

from .config import settings


@lru_cache(maxsize=1)
def has_gpu() -> bool:
    """True if a CUDA-capable GPU is visible to torch. A CPU-only torch build
    (see requirements.txt) always returns False even if the machine has a GPU."""
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def use_fp16() -> bool:
    """Resolve the fp16 setting: honour an explicit config override, else
    default to fp16 on GPU and fp32 on CPU."""
    if settings.use_fp16 is not None:
        return settings.use_fp16
    return has_gpu()


def device_label() -> str:
    return "GPU (cuda, fp16)" if use_fp16() else ("GPU (cuda, fp32)" if has_gpu() else "CPU")
