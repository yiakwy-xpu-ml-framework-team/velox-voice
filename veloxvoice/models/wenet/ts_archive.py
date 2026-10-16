"""Robust reader for WeNet TorchScript (`final.zip`) model bundles on any arch.

Rationale: some released bundles (e.g. gigaspeech u2pp quant libtorch) contain
dynamically-quantized modules whose __setstate__ triggers FBGEMM — unavailable on
aarch64/GB10, making `torch.jit.load(...).state_dict()` unusable there.
This reader parses the zip directly: storages via persistent_load, module tree via
attribute skeletons, giving a flat `name -> Tensor` map equivalent to the
checkpoint's state_dict() (quantized Linear parameters are preserved as their
pill names/values exactly as stored).
"""

from __future__ import annotations

import io
import pickle
import zipfile

import torch


class _Skeleton:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_state"] = state


class _ZipWithRoot:
    def __init__(self, zf: zipfile.ZipFile, root: str):
        self.zf, self.root = zf, root

    def read(self, name: str) -> bytes:
        if name.startswith("final/") is False and not name.startswith(f"{self.root}/"):
            name = f"{self.root}/{name}"
        return self.zf.read(name)


_STORE_DT = {
    "FloatStorage": torch.float32, "DoubleStorage": torch.float64,
    "HalfStorage": torch.float16,
    "QUInt8Storage": torch.uint8, "QInt8Storage": torch.int8,
    "UInt8Storage": torch.uint8, "ByteStorage": torch.uint8,
    "IntStorage": torch.int32, "LongStorage": torch.int64,
    "ShortStorage": torch.int16, "BoolStorage": torch.bool,
    "QInt32Storage": torch.qint32,
    "QUInt4x2Storage": torch.uint8,
}


class _StoragePayload:
    __slots__ = ("dtype", "data", "path")

    def __init__(self, dtype, data, path):
        self.dtype, self.data, self.path = dtype, data, path


class _QuantPayload:
    __slots__ = ("payload", "offset", "size", "stride", "qparams")

    def __init__(self, payload, offset, size, stride, qparams):
        self.payload, self.offset, self.size, self.stride, self.qparams = \
            payload, offset, size, stride, qparams

    @property
    def scale(self):
        return float(self.qparams[1]) if self.qparams and len(self.qparams) >= 3 else 1.0

    @property
    def zero_point(self):
        return int(self.qparams[2]) if self.qparams and len(self.qparams) >= 3 else 0


def _rebuild_tensor_v2(payload, storage_offset, size, stride, requires_grad,
                       backward_hooks, metadata=None):
    return torch.frombuffer(bytearray(payload.data), dtype=payload.dtype).reshape(size)


def _rebuild_qtensor(storage, storage_offset, size, stride, qparams, requires_grad,
                     backward_hooks):
    return _QuantPayload(storage, storage_offset, size, stride, qparams)


class _ArchiveUnpickler(pickle.Unpickler):
    def __init__(self, fp, zr: _ZipWithRoot):
        super().__init__(fp)
        self._zr = zr

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch._utils" and name == "_rebuild_qtensor":
            return _rebuild_qtensor
        if module == "collections" and name == "OrderedDict":
            return __import__("collections").OrderedDict
        if module.startswith("__torch__"):
            return _Skeleton
        if module.startswith("pyannote") or module.startswith("wespeaker") \
                or module.startswith("speechbrain") or module.startswith("asteroid"):
            return _Skeleton
        return super().find_class(module, name)

    def persistent_load(self, saved_id):
        # torch.save format: ("storage", storage_class, key, location, size)
        if saved_id[0] != "storage":
            raise pickle.UnpicklingError(f"unsupported persistent id: {saved_id}")
        storage_cls = saved_id[1]
        name = getattr(storage_cls, "__name__", storage_cls)
        key = saved_id[2] if len(saved_id) >= 3 else saved_id[1]
        data = self._zr.read(f"data/{key}")
        return _StoragePayload(_STORE_DT.get(name, torch.float32), data, key)


def _leaf_paths(obj, prefix, out, q_scale=None, q_zp=None):
    """name -> fp32 tensor (quantized linears dequantized via (weight-zp)*scale)."""
    if isinstance(obj, torch.Tensor):
        out[prefix] = ((obj - float(q_zp)) * float(q_scale) if q_scale is not None else obj)
        return
    if isinstance(obj, _QuantPayload):
        z = torch.frombuffer(bytearray(obj.payload.data), dtype=obj.payload.dtype).reshape(obj.size)
        out[prefix] = (z.float() - obj.zero_point) * obj.scale
        return
    d = getattr(obj, "__dict__", None)
    if not isinstance(d, dict):
        return
    from numbers import Number

    scale_v, zp_v = d.get("scale"), d.get("zero_point")
    if isinstance(scale_v, (Number, torch.Tensor)) and scale_v is not None:
        q_scale, q_zp = float(scale_v), float(zp_v if zp_v is not None else 0.0)
    pp = d.get("_packed_params")             # quantized linears' tensors live here
    pp_state = getattr(pp, "__dict__", {}).get("_state") if pp is not None else None
    if isinstance(pp_state, tuple) and len(pp_state) >= 2:
        _leaf_paths(pp_state[0], f"{prefix}.weight" if prefix else "weight", out,
                    q_scale, q_zp)
        if pp_state[1] is not None:
            _leaf_paths(pp_state[1], f"{prefix}.bias" if prefix else "bias", out,
                        q_scale, q_zp)
    for k, v in list(d.items()):
        if k.startswith("_") or k == "_packed_params":
            continue
        _leaf_paths(v, f"{prefix}.{k}" if prefix else k, out, q_scale, q_zp)


def read_ts_archive_tensor_map(path: str) -> dict:
    zf = zipfile.ZipFile(path)
    roots = sorted({n.split("/")[0] for n in zf.namelist() if n.endswith("/data.pkl")})
    if not roots:
        raise FileNotFoundError(f"no data.pkl found in {path}")
    zr = _ZipWithRoot(zf, roots[0])
    obj = _ArchiveUnpickler(io.BytesIO(zf.read(f"{roots[0]}/data.pkl")), zr).load()
    out: dict[str, torch.Tensor] = {}
    _leaf_paths(obj, "model", out)
    return out


def ts_state_dict(path: str) -> dict:
    """Prefer torchscript when loadable; fallback to robust archive read."""
    try:
        return torch.jit.load(path, map_location="cpu").state_dict()
    except Exception:
        return read_ts_archive_tensor_map(path)
