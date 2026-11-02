"""Apple Neural Engine bridge for the MLX lane.

MLX has no public ANE backend, so we use Core ML (``coremltools``) as the
execution engine and make the Metal <-> ANE hand-off explicit:

    mx.eval(inputs)          # drain Metal GPU work
    coreml.predict(...)      # synchronous ANE/CPU execution
    mx.array(outputs)        # copy back into unified memory

The copy is the synchronization point.  For an autoregressive decoder the
dependency is serial, so there is no benefit in pipelining two stages; for
encoder-side stages (mel frontend, FFN blocks, LM head) the caller can keep
the GPU busy with the next window while the ANE drains the previous one.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any

import mlx.core as mx
import numpy as np


def is_available() -> bool:
    try:
        import coremltools  # noqa: F401
        import platform
        return platform.machine() == "arm64"
    except Exception:
        return False


def _compute_units(name: str = "ane"):
    import coremltools as ct
    name = (name or "ane").lower()
    if name in ("ane", "neural", "cpu_and_ne"):
        return ct.ComputeUnit.CPU_AND_NE
    if name in ("all", "gpu", "cpu_and_gpu"):
        return ct.ComputeUnit.ALL
    if name == "cpu":
        return ct.ComputeUnit.CPU_ONLY
    return ct.ComputeUnit.CPU_AND_NE


class ANEExecutor:
    """Synchronous Core ML model wrapper with explicit MLX hand-off."""

    def __init__(self, model_path: str | os.PathLike,
                 compute_units: str = "ane"):
        import coremltools as ct
        self.path = pathlib.Path(model_path)
        self.model = ct.models.MLModel(str(self.path),
                                       compute_units=_compute_units(compute_units))
        spec = self.model.get_spec()
        self.input_names = [i.name for i in spec.description.input]
        self.output_names = [o.name for o in spec.description.output]

    def __call__(self, inputs: dict[str, mx.array]) -> dict[str, mx.array]:
        np_inputs = {}
        for name, value in inputs.items():
            mx.eval(value)  # Metal -> unified memory barrier
            np_inputs[name] = np.array(value.astype(mx.float32), copy=True)
        out = self.model.predict(np_inputs)  # blocks until ANE completes
        return {name: mx.array(value) for name, value in out.items()}

    def predict_one(self, value: mx.array) -> mx.array:
        name = self.input_names[0]
        return self({name: value})[self.output_names[0]]


def convert_module(module: Any,
                   example_inputs: tuple,
                   output_path: str | os.PathLike,
                   *,
                   input_names: list[str] | None = None,
                   output_name: str = "out",
                   compute_units: str = "ane",
                   minimum_deployment_target: str = "macOS15",
                   precision: str = "float16") -> pathlib.Path:
    """Trace a torch module and compile it to a Core ML ``mlpackage``.

    This is intentionally generic: the caller decides which stage is worth
    offloading (mel frontend, a conformer FFN, the LM head, ...).  The model is
    traced once with the exact production shape, so Core ML can specialize the
    ANE program.
    """
    import coremltools as ct
    import torch

    module = module.eval()
    traced = torch.jit.trace(module, example_inputs)
    if input_names is None:
        input_names = [f"in_{i}" for i in range(len(example_inputs))]
    ct_inputs = [
        ct.TensorType(name=name, shape=tuple(t.shape))
        for name, t in zip(input_names, example_inputs)
    ]
    target = getattr(ct.target, minimum_deployment_target)
    model = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=[ct.TensorType(name=output_name)],
        compute_units=_compute_units(compute_units),
        minimum_deployment_target=target,
        convert_to="mlprogram",
        compute_precision=(ct.precision.FLOAT16 if precision == "float16"
                           else ct.precision.FLOAT32),
    )
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    return output_path


def convert_mlx_linear(weight: mx.array,
                       bias: mx.array | None,
                       example_input: mx.array,
                       output_path: str | os.PathLike,
                       *,
                       input_name: str = "x",
                       output_name: str = "y",
                       **kwargs) -> pathlib.Path:
    """Compile an MLX linear layer to Core ML behind a torch-free interface.

    The caller only passes ``mx.array`` objects.  Torch is an implementation
    detail of the one-time compiler, not part of the MLX runtime lane.
    """
    import torch

    w = np.array(weight.astype(mx.float32))
    b = None if bias is None else np.array(bias.astype(mx.float32))
    lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=b is not None)
    with torch.no_grad():
        lin.weight.copy_(torch.from_numpy(w))
        if b is not None:
            lin.bias.copy_(torch.from_numpy(b))
    ex = torch.from_numpy(np.array(example_input.astype(mx.float32)))
    return convert_module(lin, (ex,), output_path,
                          input_names=[input_name], output_name=output_name,
                          **kwargs)


def benchmark(executor: ANEExecutor, inputs: dict[str, mx.array],
              warmup: int = 10, rep: int = 50) -> dict[str, float]:
    import time
    for _ in range(warmup):
        executor(inputs)
    times = []
    for _ in range(rep):
        t0 = time.perf_counter()
        executor(inputs)
        times.append((time.perf_counter() - t0) * 1e6)
    times.sort()
    return {"median_us": times[len(times) // 2], "min_us": times[0],
            "max_us": times[-1]}
