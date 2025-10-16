from __future__ import annotations
from pathlib import Path
import base64, io
from typing import List
import numpy as np
from PIL import Image

def save_image_grid(tensors: np.ndarray, grid: int = 4, out: str | Path = "grid.png") -> str:
    # tensors: (N, H, W) or (N, 1, H, W) in [0,1]
    import math
    N = tensors.shape[0]
    g = grid or int(math.sqrt(N))
    imgs = (tensors * 255).astype(np.uint8)
    if imgs.ndim == 4:
        imgs = imgs[:,0]
    H, W = imgs.shape[1:3]
    canvas = Image.new("L", (g*W, g*H), color=0)
    k = 0
    for r in range(g):
        for c in range(g):
            if k >= N: break
            tile = Image.fromarray(imgs[k])
            canvas.paste(tile, (c*W, r*H))
            k += 1
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return str(out)

def image_to_base64(path: str | Path) -> str:
    with open(path, "rb") as f:
        b = f.read()
    return base64.b64encode(b).decode("utf-8")
