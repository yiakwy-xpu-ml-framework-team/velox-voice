from .config import WeNetConfig, load_cmvn, load_config

__all__ = ["WeNetConfig", "load_cmvn", "load_config", "TorchWenEtModel",
           "WeNetASRModelMLX"]


def __getattr__(name):
    if name == "TorchWenEtModel":
        from .torch_model import TorchWenEtModel

        return TorchWenEtModel
    if name == "WeNetASRModelMLX":
        # TODO (yiakwy) : adding MLX modules
        # from .mlx_model import WeNetASRModelMLX

        # return WeNetASRModelMLX
        pass
    raise AttributeError(name)
