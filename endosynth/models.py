import torch
import numpy as np
from pathlib import Path
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
import torch.nn.functional as F
import cv2

# DAv1 from https://github.com/LiheYoung/Depth-Anything in .
from depth_anything.dpt import DPT_DINOv2
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

# DAv2 from https://github.com/DepthAnything/Depth-Anything-V2
from depth_anything_v2.dpt import DepthAnythingV2

# EndoDAC from https://github.com/BeileiCui/EndoDAC
import models.endodac.endodac as endodac

# MiDaS from https://github.com/isl-org/MiDaS
from midas.dpt_depth import DPTDepthModel

MAX_DEPTH = 0.3


class DepthAnythingAct(torch.nn.Module):
    """The output of the DPTHead is treated differently depending the version"""

    def __init__(self, version: str = "v1"):
        super(DepthAnythingAct, self).__init__()
        self.version = version

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.version == "v1":
            # in DAv1, the output is activated with a relu --> inverse depth
            x = F.relu(x)
            # to mimick the distribution of metric depth, we take the negative logarithm, followed by a sigmoid
            x = torch.sigmoid(-torch.log(x + 1e-5))
        elif self.version == "v2":
            # in DAv2 metric, the output is activated with a sigmoid
            x = torch.sigmoid(x)
        return x


class Wrapper(object):
    def __init__(self, device: torch.device | str):
        self.device = device
        self._model: torch.nn.Module = None
        self.act: torch.nn.Module = None

    def to_tensor(
        self, x: np.ndarray, input_size: int = None
    ) -> tuple[torch.Tensor, tuple[int, int]]:

        transform = Compose(
            [
                Resize(
                    width=input_size,
                    height=input_size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

        h, w = x.shape[:2]
        image = cv2.cvtColor(x, cv2.COLOR_BGR2RGB) / 255.0
        image = transform({"image": image})["image"]
        image = torch.from_numpy(image).unsqueeze(0)
        image = image.to(self.device)

        return image, (h, w)

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self._model.load_state_dict(ckpt)
        self._model = self._model.to(self.device).eval()

    @torch.no_grad()
    def infer(self, x: np.ndarray) -> np.ndarray:
        image, (h, w) = self.to_tensor(x, 518)
        logits = self._model(image)
        depth = self.act(logits) * MAX_DEPTH
        depth = F.interpolate(
            depth[:, None], (h, w), mode="bilinear", align_corners=True
        )[0, 0]
        return depth.cpu().numpy().squeeze()


class DAv1(Wrapper):
    def __init__(self, device: torch.device | str):
        super().__init__(device)
        config = dict(
            encoder="vitb",
            features=128,
            out_channels=[96, 192, 384, 768],
            localhub=False,
        )
        self._model = DPT_DINOv2(**config)
        self.act = DepthAnythingAct("v1")


class DAv2(Wrapper):
    def __init__(self, device: torch.device | str):
        super().__init__(device)
        config = dict(encoder="vitb", features=128, out_channels=[96, 192, 384, 768])
        self._model = DepthAnythingV2(**config)
        self.act = DepthAnythingAct("v2")


class EndoDAC(Wrapper):
    def __init__(self, device: torch.device | str):
        super().__init__(device)
        self._model = endodac(
            backbone_size="base",
            r=4,
            lora_type="dvlora",
            image_shape=(224, 280),
            pretrained_path=None,
            residual_block_indexes=[2, 5, 8, 11],
            include_cls_token=True,
        )

    def load(self, path: str):
        ckpt = torch.load(path)
        state_dict = self._model.state_dict()
        self._model.load_state_dict({k: v for k, v in ckpt.items() if k in state_dict})

    @torch.no_grad()
    def infer(self, x: np.ndarray) -> np.ndarray:
        h, w = x.shape[-2:]
        disp = self._model(x)[("disp", 0)]
        disp = F.interpolate(disp, size=(h, w), mode="bilinear", align_corners=True)
        min_disp = 1 / MAX_DEPTH
        max_disp = 1 / 0.001
        depth = 1 / (min_disp + (max_disp - min_disp) * disp)
        return depth.cpu().numpy().squeeze()


class Midas(Wrapper):
    def __init__(self, device: torch.device | str):
        super().__init__(device)
        self._model = DPTDepthModel(
            path=None, backbone="beitl16_512", non_negative=True
        )
        self.normaliser = Normalize(0.5, 0.5)
        self.ALPHA = 70000

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        keys = [k for k in ckpt if "attn.relative_position_index" in k]
        for k in keys:
            del ckpt[k]
        self._model.load_state_dict(ckpt)

    @torch.no_grad()
    def infer(self, x: np.ndarray) -> np.ndarray:
        h, w = x.shape[-2:]
        x = self.normaliser(x)
        o = self._model(x)
        o = F.interpolate(
            o.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=True
        )
        min_disp = 1 / MAX_DEPTH
        max_disp = 1 / 0.001
        depth = self.ALPHA / (min_disp + (max_disp - min_disp) * x)
        return depth.cpu().numpy().squeeze()


def load(
    arch: str, device: torch.device | str = "cpu", finetuned: bool = True
) -> Wrapper:

    if arch == "dav1":
        model = DAv1(device)
    elif arch == "dav2":
        model = DAv2(device)
    elif arch == "endodac":
        model = EndoDAC(device)
    elif arch == "midas":
        model = Midas(device)
    if finetuned:
        ckpts_path = Path(__file__).parent.parent / "checkpoints"
        model.load(ckpts_path / f"{arch}-f.pth")

    return model
