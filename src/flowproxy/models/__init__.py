from .unetex import UNetEx
from .unet import UNet
from .fno import FNO2d


def build_model(name: str, in_channels: int, out_channels: int, cfg: dict):
    name = name.lower()
    if name == "unetex":
        u = cfg["unetex"]
        return UNetEx(
            in_channels,
            out_channels,
            kernel_size=int(u["kernel_size"]),
            filters=list(u["filters"]),
            layers=int(u["layers"]),
            weight_norm=bool(u["weight_norm"]),
            batch_norm=bool(u["batch_norm"]),
        )
    if name == "unet":
        return UNet(in_channels, out_channels)
    if name == "fno":
        f = cfg["fno"]
        return FNO2d(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes=tuple(f["n_modes"]),
            hidden_channels=int(f["hidden_channels"]),
            n_layers=int(f["n_layers"]),
            lifting_channels=int(f["lifting_channels"]),
            projection_channels=int(f["projection_channels"]),
        )
    raise ValueError(f"unknown model {name}")
