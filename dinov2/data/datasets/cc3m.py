import os
from typing import Callable, Optional, Tuple, List

from PIL import Image
from torchvision.datasets import VisionDataset


class CC3MDataset(VisionDataset):
    """
    Minimal dataset for CC3M-style data.

    Expects *images only* under `root` (can be flat or nested in subfolders).
    Labels are dummy (always 0) since DINOv2 is self-supervised and ignores them.
    """

    def __init__(
        self,
        root: str,
        split: str = "TRAIN",
        extra: Optional[str] = None,   # accepted but ignored
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root=root, transform=transform, target_transform=target_transform)

        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        samples: List[str] = []

        # Walk recursively under root and collect image files
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if fname.lower().endswith(exts):
                    samples.append(os.path.join(dirpath, fname))

        samples.sort()
        if len(samples) == 0:
            raise RuntimeError(f"No image files found under {root}")

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        path = self.samples[index]
        img = Image.open(path).convert("RGB")
        target = 0  # dummy label

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # DINOv2’s collate_fn works fine with (image, label) tuples
        return img, target
