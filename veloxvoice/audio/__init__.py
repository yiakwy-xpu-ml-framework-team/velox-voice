from .frontend_torch import FrontendConfig
from .text_tokenizer import TextTokenizer

__all__ = ["TextTokenizer", "FrontendConfig"]


def make_frontend(cfg: FrontendConfig, device: str | None = None):
    from veloxvoice.runtime.device import get_backend

    if get_backend() == "mlx":
        from .frontend_mlx import MlxGpuFrontend

        return MlxGpuFrontend(cfg)
    from .frontend_torch import TorchGpuFrontend

    return TorchGpuFrontend(cfg, device=device or "cuda:0")
