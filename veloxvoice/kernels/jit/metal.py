"""Metal JIT kernels via mlx.fast.metal_kernel.

Metal kernels are compiled by Apple's shader compiler at first use; we keep an
in-memory cache keyed exactly like the CUDA side (name+source+template) so the
program-contract is identical across backends.
"""

from __future__ import annotations

from .base import FlashFloatJitKernel


class MetalJitKernel(FlashFloatJitKernel):
    """A named Metal kernel. `spec` supplies grid/threadgroup/template builders."""

    def __init__(
        self,
        name: str,
        source: str,
        input_names: list[str],
        output_names: list[str],
        header: str = "",
    ):
        self.name = name
        self._source = source
        self._header = header
        self.input_names = input_names
        self.output_names = output_names
        self._fn = None

    def source(self) -> str:
        return self._header + "\n" + self._source

    def build(self, **_):
        if self._fn is not None:
            return self._fn
        import mlx.core as mx

        self._fn = mx.fast.metal_kernel(
            name=f"velox_{self.name}",
            input_names=self.input_names,
            output_names=self.output_names,
            source=self._source,
            header=self._header,
            ensure_row_contiguous=True,
        )
        return self._fn

    def __call__(
        self,
        *,
        inputs,
        output_shapes,
        output_dtypes,
        grid,
        threadgroup,
        template=(),
        **kw,
    ):
        fn = self.build()
        return fn(
            inputs=inputs,
            template=list(template),
            grid=grid,
            threadgroup=threadgroup,
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
            **kw,
        )
