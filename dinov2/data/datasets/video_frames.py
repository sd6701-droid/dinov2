# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import hashlib
import logging
import os
import warnings
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from .extended import ExtendedVisionDataset


logger = logging.getLogger("dinov2")

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class VideoFramesDataset(ExtendedVisionDataset):
    """
    Self-supervised dataset over a directory of decoded video frames (or any images).

    Expected layout (produced by scripts/extract_frames.py, but any nesting works):
        root/<video_id>/000001.jpg
        root/<video_id>/000002.jpg
        ...

    Every image found recursively under `root` is one sample; targets are dummy zeros.

    Walking millions of frames is slow, so the sorted file list is cached in
    `root/.frame_index_<hash>.txt` (or under `extra` if `root` is read-only).
    Delete that file if the frames on disk change.
    """

    Labels = int

    def __init__(
        self,
        *,
        root: str,
        split: str = "TRAIN",
        extra: Optional[str] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.split = split  # kept for API compatibility; frames are not split-aware
        self._extra_root = extra
        self._paths: List[str] = self._load_or_build_index()
        if len(self._paths) == 0:
            raise RuntimeError(f"VideoFramesDataset: no image files found under root='{self.root}'")

    # ---------- file index ----------

    def _index_path(self) -> str:
        tag = hashlib.md5(os.path.abspath(self.root).encode()).hexdigest()[:8]
        base = self._extra_root if self._extra_root else self.root
        return os.path.join(base, f".frame_index_{tag}.txt")

    def _load_or_build_index(self) -> List[str]:
        index_path = self._index_path()
        if os.path.isfile(index_path):
            with open(index_path) as f:
                rel_paths = [line.rstrip("\n") for line in f if line.strip()]
            logger.info(f"VideoFramesDataset: loaded {len(rel_paths):,d} frames from index {index_path}")
            return [os.path.join(self.root, p) for p in rel_paths]

        logger.info(f"VideoFramesDataset: indexing frames under {self.root} (first run, may take a while)")
        rel_paths = []
        for dirpath, _, filenames in os.walk(self.root):
            for fname in filenames:
                if fname.lower().endswith(_IMAGE_EXTS):
                    rel_paths.append(os.path.relpath(os.path.join(dirpath, fname), self.root))
        rel_paths.sort()

        try:
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            tmp_path = f"{index_path}.tmp.{os.getpid()}"
            with open(tmp_path, "w") as f:
                f.write("\n".join(rel_paths))
            os.replace(tmp_path, index_path)  # atomic, safe with several ranks racing
        except OSError as e:
            logger.warning(f"VideoFramesDataset: could not write index {index_path}: {e}")

        return [os.path.join(self.root, p) for p in rel_paths]

    # ---------- ExtendedVisionDataset interface ----------

    def __len__(self) -> int:
        return len(self._paths)

    def get_image_data(self, index: int) -> bytes:
        with open(self._paths[index], "rb") as f:
            return f.read()

    def get_target(self, index: int) -> Any:
        return 0

    def get_targets(self) -> np.ndarray:
        return np.zeros(len(self._paths), dtype=np.int64)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # Skip a few unreadable/corrupt frames instead of crashing the run.
        cur_index = index
        for _ in range(5):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return super().__getitem__(cur_index)
            except RuntimeError as e:
                if "can not read image for sample" not in str(e):
                    raise
                cur_index = (cur_index + 1) % len(self._paths)
        raise RuntimeError(f"VideoFramesDataset: too many unreadable frames around index {index} in {self.root}")
