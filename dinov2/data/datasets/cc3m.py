# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import os
import warnings
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from .extended import ExtendedVisionDataset


class CC3MDataset(ExtendedVisionDataset):
    """
    Minimal CC3M dataset for DINOv2.

    Assumes:
      - `root` is a directory containing image files (possibly in subfolders).
      - Labels are not used (self-supervised), so we return dummy targets.
    """

    Labels = int

    def __init__(
        self,
        *,
        root: str,
        split: str = "TRAIN",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        # ExtendedVisionDataset takes (root, transforms, transform, target_transform)
        super().__init__(root, transforms, transform, target_transform)

        self.split = split  # kept for API compatibility

        # Collect all image files under root (recursively)
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        paths: List[str] = []

        for dirpath, _, filenames in os.walk(self.root):
            for fname in filenames:
                if fname.lower().endswith(exts):
                    paths.append(os.path.join(dirpath, fname))

        paths.sort()
        if len(paths) == 0:
            raise RuntimeError(f"CC3MDataset: no image files found under root='{self.root}'")

        self._paths: List[str] = paths

    # --------- Required ExtendedVisionDataset interface ----------

    def __len__(self) -> int:
        return len(self._paths)

    def get_image_data(self, index: int) -> bytes:
        """
        Return raw image bytes for sample `index`.
        ExtendedVisionDataset will convert this to a PIL image internally.
        """
        path = self._paths[index]
        with open(path, "rb") as f:
            data = f.read()
        return data

    def get_target(self, index: int) -> Any:
        """
        DINOv2 training is self-supervised, so we don't actually use labels.
        We just return a dummy target.
        """
        return 0

    def get_targets(self) -> np.ndarray:
        """
        Optional helper to return all targets as a numpy array.
        Here it's just zeros.
        """
        return np.zeros(len(self._paths), dtype=np.int64)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Wrap parent __getitem__, but if a particular image is unreadable
        (PIL.UnidentifiedImageError → RuntimeError in ExtendedVisionDataset),
        skip it and try a nearby sample instead.

        This prevents a single corrupt file from crashing training.
        """
        num_trials = 0
        max_trials = 5
        cur_index = index

        while num_trials < max_trials:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return super().__getitem__(cur_index)
            except RuntimeError as e:
                msg = str(e)
                # ExtendedVisionDataset raises this when PIL can't read the image
                if "can not read image for sample" not in msg:
                    # some other unexpected error → propagate
                    raise

                # corrupt/unreadable image: move to next index
                num_trials += 1
                cur_index = (cur_index + 1) % len(self._paths)

        # too many consecutive bad samples → give a clear error
        raise RuntimeError(
            f"CC3MDataset: too many unreadable images around index {index}. "
            "Please check your CC3M folder for corrupt files."
        )
