# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Image dataset built from video annotation files (V-JEPA / SALT style).

Each sample is ONE random frame decoded on the fly from one video, so the videos
never need to be extracted to JPEGs. With frames_per_video=K, every video appears
K times per epoch (a different random frame each time).

Supported annotation formats (one video per line / entry):
  * .csv / .txt : "path label" (space separated, V-JEPA style), "path,label,...",
                  or just "path". A header row (path/video/filename/...) is skipped.
  * .json       : a list of paths, or a list of dicts with one of the keys
                  path / video_path / video / filename / file, or with an "id" key
                  (official SSv2 labels) which becomes "<id>.webm".
                  A dict {video_id: ...} is also accepted.
Relative paths are resolved against `video_root` (the `extra=` field of the dataset
string) or, if that is not given, against the folder of the annotation file.
"""

import json
import logging
import os
import random
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset


logger = logging.getLogger("dinov2")

_PATH_KEYS = ("path", "video_path", "video", "filename", "file", "fname")
_HEADER_WORDS = {"path", "video", "video_path", "filename", "file", "fname", "youtube_id", "id"}
_VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".avi", ".mov", ".m4v")


def _parse_csv_line(line: str) -> Optional[str]:
    line = line.strip()
    if not line:
        return None
    if "," in line:
        first = line.split(",")[0].strip()
    else:
        parts = line.rsplit(maxsplit=1)
        # "path label" -> drop the trailing integer label; otherwise the whole line is the path
        first = parts[0] if len(parts) == 2 and parts[1].lstrip("-").isdigit() else line
    first = first.strip().strip('"')
    if first.lower() in _HEADER_WORDS:
        return None
    return first


def _entry_to_path(entry: Any, default_ext: str) -> Optional[str]:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        for key in _PATH_KEYS:
            if key in entry and entry[key]:
                return str(entry[key])
        if "id" in entry:
            vid = str(entry["id"])
            return vid if vid.lower().endswith(_VIDEO_EXTS) else vid + default_ext
    if isinstance(entry, (list, tuple)) and entry:
        return str(entry[0])
    return None


def read_video_list(annotation_file: str, video_root: Optional[str] = None) -> List[str]:
    base = video_root if video_root else os.path.dirname(os.path.abspath(annotation_file))
    raw: List[str] = []

    if annotation_file.lower().endswith(".json"):
        with open(annotation_file) as f:
            data = json.load(f)
        if isinstance(data, dict):
            # {video_id: info} or {"videos": [...]} / {"data": [...]}
            for key in ("videos", "data", "annotations", "items"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
            else:
                data = [
                    v if isinstance(v, dict) and any(k in v for k in _PATH_KEYS) else {"id": k}
                    for k, v in data.items()
                ]
        for entry in data:
            p = _entry_to_path(entry, default_ext=".webm")
            if p:
                raw.append(p)
    else:
        with open(annotation_file) as f:
            for line in f:
                p = _parse_csv_line(line)
                if p:
                    raw.append(p)

    return [p if os.path.isabs(p) else os.path.join(base, p) for p in raw]


class _FrameReader:
    """Decode a single frame from a video with decord if available, else PyAV."""

    def __init__(self):
        try:
            import decord  # noqa: F401

            self.backend = "decord"
        except ImportError:
            try:
                import av  # noqa: F401

                self.backend = "pyav"
            except ImportError as e:
                raise ImportError(
                    "VideoFilesDataset needs a video decoder: `pip install decord` (recommended) or `pip install av`"
                ) from e

    def random_frame(self, path: str) -> Image.Image:
        if self.backend == "decord":
            import decord

            vr = decord.VideoReader(path, num_threads=1, ctx=decord.cpu(0))
            n = len(vr)
            if n == 0:
                raise RuntimeError("empty video")
            frame = vr[random.randrange(n)].asnumpy()
            return Image.fromarray(frame)

        import av

        with av.open(path) as container:
            frames = [fr for fr in container.decode(video=0)]
        if not frames:
            raise RuntimeError("empty video")
        return random.choice(frames).to_image()


class VideoFilesDataset(VisionDataset):
    Labels = int

    def __init__(
        self,
        *,
        root: str,  # path to the annotation file (.csv/.txt/.json)
        split: str = "TRAIN",
        extra: Optional[str] = None,  # optional folder that relative video paths are relative to
        frames: str = "1",  # frames sampled per video per epoch
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.split = split
        self._videos = read_video_list(root, video_root=extra)
        if not self._videos:
            raise RuntimeError(f"VideoFilesDataset: no videos parsed from annotation file {root}")
        self._frames_per_video = max(1, int(frames))
        self._reader: Optional[_FrameReader] = None  # created lazily inside each worker
        self._n_failures = 0
        logger.info(
            f"VideoFilesDataset: {len(self._videos):,d} videos from {root} "
            f"(x{self._frames_per_video} frames/video), e.g. {self._videos[0]}"
        )

    def __len__(self) -> int:
        return len(self._videos) * self._frames_per_video

    def video_path(self, index: int) -> str:
        return self._videos[index % len(self._videos)]

    def get_target(self, index: int) -> Any:
        return 0

    def get_targets(self) -> np.ndarray:
        return np.zeros(len(self), dtype=np.int64)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self._reader is None:
            self._reader = _FrameReader()

        cur = index
        for _ in range(10):
            path = self.video_path(cur)
            try:
                image = self._reader.random_frame(path).convert("RGB")
                break
            except Exception as e:  # missing / corrupt / undecodable video
                self._n_failures += 1
                if self._n_failures <= 20:
                    logger.warning(f"VideoFilesDataset: skipping unreadable video {path}: {e}")
                cur = random.randrange(len(self))
        else:
            raise RuntimeError(f"VideoFilesDataset: 10 unreadable videos in a row starting at {self.video_path(index)}")

        target = self.get_target(index)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target
