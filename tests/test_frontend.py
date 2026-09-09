import numpy as np
import pytest

from veloxvoice.audio import FrontendConfig


def test_torch_frontend_stream_equivalence():
    torch = pytest.importorskip("torch")

    from veloxvoice.audio.frontend_torch import TorchGpuFrontend

    cfg = FrontendConfig()
    rng = np.random.default_rng(0)
    pcm = rng.standard_normal(16000).astype(np.float32)

    gpu_fe = TorchGpuFrontend(cfg, device="cuda")
    parts = [gpu_fe.accept(pcm[i : i + 1600]) for i in range(0, len(pcm), 1600)]
    streamed = torch.cat([p for p in parts if p.shape[0] > 0]).cpu().numpy()

    cpu_fe = TorchGpuFrontend(cfg, device="cpu")
    one_shot_ref = cpu_fe.accept(pcm).cpu().numpy()

    n = min(len(streamed), len(one_shot_ref))
    assert np.allclose(streamed[:n], one_shot_ref[:n], atol=5e-3)


def test_torch_frontend_gpu_smoke():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from veloxvoice.audio.frontend_torch import TorchGpuFrontend

    f = TorchGpuFrontend(FrontendConfig(), device="cuda:0")
    out = f.accept(np.random.randn(1600).astype(np.float32))
    assert out.is_cuda and out.shape[-1] == 80
