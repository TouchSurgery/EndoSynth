from matplotlib import colormaps as cm
import numpy as np
from PIL import Image


def depth2rgb(x: np.ndarray, dmin: float, dmax: float) -> np.ndarray:
    cmap = cm.get_cmap("Spectral")
    x = (np.clip(x, dmin, dmax) - dmin) / (dmax - dmin)
    x = (x * 255).astype(np.uint8)
    return (cmap(x) * 255)[..., :3]


def seg2rgb(x: np.ndarray) -> np.ndarray:
    cmap = cm.get_cmap("Set2")
    rgba = np.where(x[..., None] > 0, cmap(x - 1), 0)
    return (rgba * 255)[..., :3]


def sample2png(x: dict[str, np.ndarray], path: str):
    rgb = x["rgb"]
    seg = seg2rgb(x["seg"])
    depth = depth2rgb(x["depth"], x["depth"].min(), x["depth"].max())
    seg = np.where(x["seg"][..., None] > 0, 0.5 * seg + 0.5 * rgb, rgb)
    img = np.concatenate([rgb, seg, depth], axis=1)
    Image.fromarray(img.astype(np.uint8)).save(path)
