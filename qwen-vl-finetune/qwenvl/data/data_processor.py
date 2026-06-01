import json
import random
import logging
import re
import time
import itertools
import csv
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple, Any
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import transformers

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2, get_rope_index_3

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def _make_abs_paths(base: Path, files: str) -> str:
    return f"{(base / files).resolve()}"


def _is_remote_or_data_path(path: str) -> bool:
    return (
        path.startswith("http://")
        or path.startswith("https://")
        or path.startswith("file://")
        or path.startswith("data:")
    )


def _normalize_media_entry(base: Path, media: Any, media_key: str) -> Dict[str, Any]:
    if isinstance(media, str):
        path = media
        if not _is_remote_or_data_path(path):
            path = _make_abs_paths(base, path)
        return {"type": media_key, media_key: path}

    if not isinstance(media, dict):
        raise TypeError(f"{media_key} entries must be paths or dicts, got {type(media)}")

    entry = media.copy()
    path = entry.get(media_key) or entry.get("path")
    if path is None:
        raise ValueError(f"{media_key} entry is missing '{media_key}' or 'path': {media}")
    if isinstance(path, str) and not _is_remote_or_data_path(path) and not Path(path).is_absolute():
        path = _make_abs_paths(base, path)
    entry[media_key] = path
    entry.pop("path", None)
    entry.setdefault("type", media_key)
    return entry


def update_processor_pixels(processor, data_args):
    logger = logging.getLogger(__name__)

    # --- Image Processor ---
    ip = processor.image_processor
    rank0_print("=== BEFORE IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"ip.size: {ip.size}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = data_args.min_pixels
        ip.max_pixels = data_args.max_pixels
        rank0_print(f"✅ Updated image_processor min_pixels to {data_args.min_pixels}")
        rank0_print(f"✅ Updated image_processor max_pixels to {data_args.max_pixels}")

    if hasattr(ip, "size") and isinstance(ip.size, dict):
        ip.size["shortest_edge"] = data_args.min_pixels
        ip.size["longest_edge"] = data_args.max_pixels
        rank0_print(
            f"✅ Updated image_processor size['shortest_edge'] to {data_args.min_pixels}"
        )
        rank0_print(
            f"✅ Updated image_processor size['longest_edge'] to {data_args.max_pixels}"
        )

    rank0_print("=== AFTER IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    # --- Video Processor ---
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        rank0_print("\n=== BEFORE VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

        if hasattr(vp, "min_pixels") and hasattr(vp, "max_pixels"):
            vp.min_pixels = data_args.video_min_pixels
            vp.max_pixels = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor min_pixels to {data_args.video_min_pixels}"
            )
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor max_pixels to {data_args.video_max_pixels}"
            )

        if hasattr(vp, "min_frames") and hasattr(vp, "max_frames"):
            vp.min_frames = data_args.video_min_frames
            vp.max_frames = data_args.video_max_frames
            rank0_print(
                f"✅ Updated video_processor min_frames to {data_args.video_min_frames}"
            )
            rank0_print(
                f"✅ Updated video_processor max_frames to {data_args.video_max_frames}"
            )

        if hasattr(vp, "fps"):
            vp.fps = data_args.video_fps
            rank0_print(f"✅ Updated video_processor fps to {data_args.video_fps}")

        if hasattr(vp, "size") and isinstance(vp.size, dict):
            vp.size["shortest_edge"] = data_args.video_min_pixels
            vp.size["longest_edge"] = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
            )
            rank0_print(
                f"✅ Updated Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}"
            )

        rank0_print("=== AFTER VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

    return processor


def _build_messages(item: Dict[str, Any], base_path: Path) -> List[Dict[str, Any]]:
    # Extract and normalize images and videos
    images = item.get("image") or []
    if isinstance(images, (str, dict)):
        images = [images]

    videos = item.get("video") or []
    if isinstance(videos, (str, dict)):
        videos = [videos]

    # Build media pools with absolute paths
    image_pool = [_normalize_media_entry(base_path, img, "image") for img in images]
    video_pool = [_normalize_media_entry(base_path, vid, "video") for vid in videos]

    messages = []
    for turn in item["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        text: str = turn["value"]

        if role == "user":
            content = []
            # Split text by <image> or <video> placeholders while keeping delimiters
            text_parts = re.split(r"(<image>|<video>)", text)

            for seg in text_parts:
                if seg == "<image>":
                    if not image_pool:
                        raise ValueError(
                            "Number of <image> placeholders exceeds the number of provided images"
                        )
                    content.append(image_pool.pop(0))
                elif seg == "<video>":
                    if not video_pool:
                        raise ValueError(
                            "Number of <video> placeholders exceeds the number of provided videos"
                        )
                    content.append(video_pool.pop(0))
                elif seg.strip():
                    content.append({"type": "text", "text": seg.strip()})

            messages.append({"role": role, "content": content})
        else:
            # Assistant messages contain only text
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    # Check for unused media files
    if image_pool:
        raise ValueError(
            f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)"
        )
    if video_pool:
        raise ValueError(
            f"{len(video_pool)} video(s) remain unused (not consumed by placeholders)"
        )

    return messages


def preprocess_qwen_visual(
    sources,
    processor,
) -> Dict:
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")

    source = sources[0]
    base_path = Path(source.get("data_path", ""))
    messages = _build_messages(source, base_path)

    full_result = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt"
    )

    input_ids = full_result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    labels = torch.full_like(input_ids, IGNORE_INDEX)

    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    pos = 0
    while pos < L:
        if input_ids_flat[pos] == 77091:
            ans_start = pos + 2
            ans_end = ans_start
            while ans_end < L and input_ids_flat[ans_end] != 151645:
                ans_end += 1
            if ans_end < L:
                labels[0, ans_start : ans_end + 2] = input_ids[
                    0, ans_start : ans_end + 2
                ]
                pos = ans_end
        pos += 1

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids
    return full_result


def get_rope_index_fn(model_type: str):
    if model_type == "qwen3vl":
        return get_rope_index_3
    if model_type == "qwen2.5vl":
        return get_rope_index_25
    if model_type == "qwen2vl":
        return get_rope_index_2
    raise ValueError(f"model_type: {model_type} not supported")


def build_supervised_item(sources, processor, get_rope_index, merge_size) -> Dict[str, torch.Tensor]:
    data_dict = preprocess_qwen_visual(
        sources,
        processor,
    )

    seq_len = data_dict["input_ids"][0].size(0)

    if "image_grid_thw" in data_dict:
        grid_thw = data_dict.get("image_grid_thw")
        if not isinstance(grid_thw, Sequence):
            grid_thw = [grid_thw]
    else:
        grid_thw = None

    if "video_grid_thw" in data_dict:
        video_grid_thw = data_dict.get("video_grid_thw")
        if not isinstance(video_grid_thw, Sequence):
            video_grid_thw = [video_grid_thw]
        second_per_grid_ts = [
            processor.video_processor.temporal_patch_size
            / processor.video_processor.fps
        ] * len(video_grid_thw)
    else:
        video_grid_thw = None
        second_per_grid_ts = None

    position_ids, _ = get_rope_index(
        merge_size,
        data_dict["input_ids"],
        image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
        video_grid_thw=(
            torch.cat(video_grid_thw, dim=0) if video_grid_thw else None
        ),
        second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
    )

    data_dict["position_ids"] = position_ids
    data_dict["attention_mask"] = [seq_len]

    return data_dict


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, processor, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        self.get_rope_index = get_rope_index_fn(data_args.model_type)

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                rank0_print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                if isinstance(ann, list):
                    for sub_ann in ann:
                        sub_ann["data_path"] = data["data_path"]
                else:
                    ann["data_path"] = data["data_path"]
            list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}")


        rank0_print("Formatting inputs...Skip in lazy mode")
        processor = update_processor_pixels(processor, data_args)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.data_args = data_args
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.list_data_dict = list_data_dict

        if data_args.data_packing:
            self.item_fn = self._get_packed_item
        else:
            self.item_fn = self._get_item

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        num_final_retries = 30

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sources = self.list_data_dict[i]
                if isinstance(sources, dict):
                    sources = [sources]
                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                sources = self.list_data_dict[next_index]
                if isinstance(sources, dict):
                    sources = [sources]

                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sources = self.list_data_dict[i]
            if isinstance(sources, dict):
                sources = [sources]
            sample = self.item_fn(sources)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        data_dict = build_supervised_item(
            sources,
            self.processor,
            self.get_rope_index,
            self.merge_size,
        )

        text = self.processor.tokenizer.decode(
            data_dict["input_ids"][0], skip_special_tokens=False
        )

        labels = data_dict["labels"][0]
        labels = [
            tid if tid != -100 else self.processor.tokenizer.pad_token_id
            for tid in labels
        ]
        label = self.processor.tokenizer.decode(labels, skip_special_tokens=False)

        return data_dict

    def _get_packed_item(self, sources) -> Dict[str, torch.Tensor]:

        if isinstance(sources, dict):
            if isinstance(source, dict):
                sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
            return self._get_item(sources)

        if isinstance(sources, list):
            data_list = []
            new_data_dict = {}
            for source in sources:
                if isinstance(source, dict):
                    source = [source]
                assert (
                    len(source) == 1
                ), f"Don't know why it is wrapped to a list.\n {source}"  # FIXME
                data_list.append(self._get_item(source))

            input_ids = torch.cat([d["input_ids"] for d in data_list], dim=1)
            labels = torch.cat([d["labels"] for d in data_list], dim=1)
            position_ids = torch.cat([d["position_ids"] for d in data_list], dim=2)
            attention_mask = [
                d["attention_mask"][0] for d in data_list if "attention_mask" in d
            ]
            new_data_dict = {
                "input_ids": input_ids,
                "labels": labels,
                "position_ids": position_ids,
                "attention_mask": attention_mask if attention_mask else None,
            }

            if any("pixel_values" in d for d in data_list):
                new_data_dict.update(
                    {
                        "pixel_values": torch.cat(
                            [
                                d["pixel_values"]
                                for d in data_list
                                if "pixel_values" in d
                            ],
                            dim=0,
                        ),
                        "image_grid_thw": torch.cat(
                            [
                                d["image_grid_thw"]
                                for d in data_list
                                if "image_grid_thw" in d
                            ],
                            dim=0,
                        ),
                    }
                )

            if any("pixel_values_videos" in d for d in data_list):
                new_data_dict.update(
                    {
                        "pixel_values_videos": torch.cat(
                            [
                                d["pixel_values_videos"]
                                for d in data_list
                                if "pixel_values_videos" in d
                            ],
                            dim=0,
                        ),
                        "video_grid_thw": torch.cat(
                            [
                                d["video_grid_thw"]
                                for d in data_list
                                if "video_grid_thw" in d
                            ],
                            dim=0,
                        ),
                    }
                )
            return new_data_dict


@dataclass(frozen=True)
class RobotSubtaskSegment:
    dataset: str
    episode_id: str
    subtask: str
    start_frame: int
    end_frame: int
    video_path: str
    fps: float
    episode_length: int


@dataclass(frozen=True)
class RobotSubtaskWindow:
    segment: RobotSubtaskSegment
    timestep: int
    clip_start_frame: int
    clip_end_frame: int
    nframes: int
    label: str  # gripper subtask at the lookahead frame (timestep + lookahead_frames)


def _parse_episode_range(spec: str) -> set[int]:
    spec = spec.strip()
    if not spec:
        return set()
    episode_ids: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            start_text, end_text = part.split(":", 1)
            start = int(start_text) if start_text else 0
            end = int(end_text)
            episode_ids.update(range(start, end))
        else:
            episode_ids.add(int(part))
    return episode_ids


def _load_episode_lengths(dataset_root: Path) -> dict[str, int]:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    lengths = {}
    for row in read_jsonl(episodes_path):
        episode_id = f"{int(row['episode_index']):06d}"
        lengths[episode_id] = int(row["length"])
    return lengths


def _load_dataset_info(dataset_root: Path) -> dict[str, Any]:
    with (dataset_root / "meta" / "info.json").open() as f:
        return json.load(f)


class RobotSubtaskDataset(Dataset):
    """Balanced marker handover subtask dataset backed by timed video windows."""

    def __init__(self, processor, data_args, dataset_config: Dict[str, Any]):
        super().__init__()
        self.processor = update_processor_pixels(processor, data_args)
        self.tokenizer = processor.tokenizer
        self.data_args = data_args
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.get_rope_index = get_rope_index_fn(data_args.model_type)

        self.root = Path(getattr(data_args, "robot_subtask_root", "") or dataset_config["data_path"])
        self.csv_path = Path(
            getattr(data_args, "robot_subtask_csv", "") or dataset_config["annotation_path"]
        )
        self.camera = getattr(data_args, "robot_subtask_camera", "") or dataset_config.get(
            "camera", "observation.images.cam_high"
        )
        self.prompt = getattr(data_args, "robot_subtask_prompt", "") or dataset_config.get("prompt")
        if not self.prompt:
            raise ValueError(
                "No robot_subtask prompt: set it in the dataset registry entry "
                "(qwenvl/data/__init__.py) or via --robot_subtask_prompt / ROBOT_SUBTASK_PROMPT."
            )
        self.history_seconds = float(getattr(data_args, "robot_subtask_history_seconds", 1.5))
        self.num_frames = int(getattr(data_args, "robot_subtask_num_frames", 6))
        # Lookahead: the label is the gripper subtask at frame (t + lookahead_frames),
        # where t is the window's final (current) frame. This makes the model predict
        # what the gripper SHOULD do next, not just describe the current frame.
        # Clamped to the episode end when t + H runs past the last frame.
        self.lookahead_frames = int(getattr(data_args, "robot_subtask_lookahead_frames", 15))
        self.epoch_size = int(getattr(data_args, "robot_subtask_epoch_size", 4096))
        self.split = getattr(data_args, "robot_subtask_split", "train")
        self.train_episodes = _parse_episode_range(
            getattr(data_args, "robot_subtask_train_episodes", "0:28")
        )
        self.val_episodes = _parse_episode_range(
            getattr(data_args, "robot_subtask_val_episodes", "28:30")
        )

        self.segments = self._load_segments(dataset_config)
        self.segments_by_label = defaultdict(list)
        for segment in self.segments:
            self.segments_by_label[segment.subtask].append(segment)
        # Labels are derived from the CSV's `subtask` column (sorted for a stable
        # order). Switching to a new task is just a new CSV -- no code edits.
        self.labels = sorted(self.segments_by_label)
        if not self.labels:
            raise ValueError(f"Robot subtask split '{self.split}' has no segments")
        rank0_print(f"Robot subtask labels ({len(self.labels)}): {self.labels}")

        # Per-episode frame -> subtask lookup, so the lookahead label at frame (t+H)
        # can be resolved even when it falls in a different segment than timestep t.
        self.frame_subtask = defaultdict(dict)  # (dataset, episode_id) -> {frame: subtask}
        for segment in self.segments:
            key = (segment.dataset, segment.episode_id)
            table = self.frame_subtask[key]
            for frame in range(segment.start_frame, segment.end_frame + 1):
                table[frame] = segment.subtask

        self.fixed_windows = self._build_fixed_windows() if self.split == "val" else []
        rank0_print(
            f"Loaded robot subtask split={self.split}, segments={len(self.segments)}, "
            f"epoch_size={len(self)}"
        )

    def _episode_allowed(self, episode_id: str) -> bool:
        idx = int(episode_id)
        if self.split == "train":
            return idx in self.train_episodes
        if self.split == "val":
            return idx in self.val_episodes
        if self.split == "all":
            return True
        raise ValueError("robot_subtask_split must be one of: train, val, all")

    def _dataset_metadata(self, metadata: dict, dataset_name: str) -> dict:
        """Lazily load + cache the LeRobot meta (fps, chunk size, episode lengths)
        for a dataset folder. Dataset folders come from the CSV's `dataset` column,
        so a new task's CSV can reference any folder under the data root."""
        if dataset_name not in metadata:
            dataset_root = self.root / dataset_name
            info = _load_dataset_info(dataset_root)
            metadata[dataset_name] = {
                "fps": float(info["fps"]),
                "chunks_size": int(info["chunks_size"]),
                "lengths": _load_episode_lengths(dataset_root),
            }
        return metadata[dataset_name]

    def _load_segments(self, dataset_config: Dict[str, Any]) -> list[RobotSubtaskSegment]:
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Missing robot subtask CSV: {self.csv_path}")

        metadata: dict = {}
        segments = []
        with self.csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            required = {"dataset", "episode_id", "start_frame", "end_frame", "subtask"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{self.csv_path} is missing columns: {sorted(missing)}")

            for row in reader:
                dataset_name = row["dataset"]
                episode_id = f"{int(row['episode_id']):06d}"
                if not self._episode_allowed(episode_id):
                    continue
                subtask = row["subtask"]

                info = self._dataset_metadata(metadata, dataset_name)
                episode_length = info["lengths"][episode_id]
                chunk = int(episode_id) // info["chunks_size"]
                video_path = (
                    self.root
                    / dataset_name
                    / "videos"
                    / f"chunk-{chunk:03d}"
                    / self.camera
                    / f"episode_{episode_id}.mp4"
                )
                if not video_path.is_file():
                    raise FileNotFoundError(f"Missing robot subtask video: {video_path}")

                start_frame = int(row["start_frame"])
                end_frame = int(row["end_frame"])
                if start_frame < 0 or end_frame < start_frame or end_frame >= episode_length:
                    raise ValueError(
                        f"Invalid frame range for {dataset_name} episode_{episode_id}: "
                        f"{start_frame}-{end_frame}, length={episode_length}"
                    )
                segments.append(
                    RobotSubtaskSegment(
                        dataset=dataset_name,
                        episode_id=episode_id,
                        subtask=subtask,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        video_path=str(video_path),
                        fps=info["fps"],
                        episode_length=episode_length,
                    )
                )

        if not segments:
            raise ValueError(f"No robot subtask segments loaded for split '{self.split}'")
        return segments

    def _label_at_lookahead(self, segment: RobotSubtaskSegment, timestep: int) -> str:
        """Subtask at frame (timestep + lookahead_frames), clamped to episode end.

        Resolves via the per-episode frame->subtask table so the label is correct
        even when the lookahead frame crosses into a neighbouring segment.
        """
        target = min(timestep + self.lookahead_frames, segment.episode_length - 1)
        key = (segment.dataset, segment.episode_id)
        table = self.frame_subtask[key]
        label = table.get(target)
        if label is None:
            # Fall back to nearest labelled frame at or before target, else the segment's own label.
            for frame in range(target, -1, -1):
                if frame in table:
                    label = table[frame]
                    break
        return label if label is not None else segment.subtask

    def _window_for_timestep(self, segment: RobotSubtaskSegment, timestep: int) -> RobotSubtaskWindow:
        history_frames = max(2, int(round(self.history_seconds * segment.fps)))
        clip_start = max(0, timestep - history_frames + 1)
        clip_end = min(timestep, segment.episode_length - 1)
        if clip_end <= clip_start:
            if clip_start > 0:
                clip_start -= 1
            else:
                clip_end = min(segment.episode_length - 1, clip_start + 1)
        total_clip_frames = clip_end - clip_start + 1
        nframes = min(self.num_frames, total_clip_frames)
        if nframes % 2 == 1:
            nframes -= 1
        nframes = max(2, nframes)
        return RobotSubtaskWindow(
            segment=segment,
            timestep=timestep,
            clip_start_frame=clip_start,
            clip_end_frame=clip_end,
            nframes=nframes,
            label=self._label_at_lookahead(segment, timestep),
        )

    def _sample_train_window(self, index: int) -> RobotSubtaskWindow:
        # Class-balanced: cycle through the labels so every class is sampled
        # equally regardless of how many segments each has. Works for any N classes.
        label = self.labels[index % len(self.labels)]
        segment = random.choice(self.segments_by_label[label])
        min_timestep = min(max(segment.start_frame, self.num_frames - 1), segment.end_frame)
        timestep = random.randint(min_timestep, segment.end_frame)
        return self._window_for_timestep(segment, timestep)

    def _build_fixed_windows(self) -> list[RobotSubtaskWindow]:
        # Deterministic eval windows: sample evenly-spaced timesteps across every
        # segment (quarter / mid / three-quarter / end). This covers both the steady
        # part of each state and the frames near the transition boundary, where the
        # lookahead label matters most.
        #
        # We deliberately skip the segment START (fraction 0.0): for a segment at the
        # very beginning of an episode that lands at frame ~0, whose window would be
        # only 1-2 frames long (clip_start clamps to 0) -- a near-empty clip that just
        # confuses the model. We also enforce a floor so every eval window has enough
        # history to fill at least num_frames frames.
        min_timestep_floor = max(0, self.num_frames - 1)
        windows = []
        for segment in self.segments:
            span = segment.end_frame - segment.start_frame
            fractions = (0.25, 0.5, 0.75, 1.0)
            candidate_ts = [
                segment.start_frame + int(round(span * frac)) for frac in fractions
            ]
            seen = set()
            for timestep in candidate_ts:
                timestep = min(max(timestep, segment.start_frame), segment.end_frame)
                # Ensure enough preceding frames exist for a non-degenerate window.
                timestep = max(timestep, min_timestep_floor)
                timestep = min(timestep, segment.end_frame)
                if timestep < segment.start_frame or timestep in seen:
                    continue
                seen.add(timestep)
                windows.append(self._window_for_timestep(segment, timestep))
        return windows

    def _source_for_window(self, window: RobotSubtaskWindow, include_answer: bool = True) -> Dict[str, Any]:
        segment = window.segment
        video_entry = {
            "video": segment.video_path,
            "video_start": window.clip_start_frame / segment.fps,
            "video_end": window.clip_end_frame / segment.fps,
            "nframes": window.nframes,
        }
        conversations = [
            {
                "from": "human",
                "value": f"{DEFAULT_VIDEO_TOKEN}\n{self.prompt}",
            }
        ]
        if include_answer:
            conversations.append({"from": "gpt", "value": window.label})
        return {"video": [video_entry], "conversations": conversations, "data_path": ""}

    def get_eval_source(self, index: int) -> tuple[Dict[str, Any], str]:
        if self.split != "val":
            raise ValueError("get_eval_source is only defined for robot_subtask_split='val'")
        window = self.fixed_windows[index]
        return self._source_for_window(window, include_answer=False), window.label

    def __len__(self):
        if self.split == "val":
            return len(self.fixed_windows)
        return self.epoch_size

    @property
    def lengths(self):
        return np.array([1] * len(self))

    @property
    def modality_lengths(self):
        return [1] * len(self)

    @property
    def pre_calculated_length(self):
        return np.array([1] * len(self))

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        window = self.fixed_windows[index] if self.split == "val" else self._sample_train_window(index)
        source = self._source_for_window(window, include_answer=True)
        return build_supervised_item(
            [source],
            self.processor,
            self.get_rope_index,
            self.merge_size,
        )


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_mask = list(
            itertools.chain(
                *(
                    instance["attention_mask"]
                    for instance in instances
                    if "attention_mask" in instance
                )
            )
        )
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        return batch


def make_supervised_data_module(processor, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_configs = data_list(data_args.dataset_use.split(","))
    robot_configs = [
        config for config in dataset_configs if config.get("dataset_type") == "robot_subtask"
    ]
    if robot_configs:
        if len(dataset_configs) != 1:
            raise ValueError("robot_subtask datasets cannot be mixed with other datasets")
        train_dataset = RobotSubtaskDataset(
            processor, data_args=data_args, dataset_config=robot_configs[0]
        )
    else:
        train_dataset = LazySupervisedDataset(processor, data_args=data_args)
    if data_args.data_flatten or data_args.data_packing:
        data_collator = FlattenedDataCollatorForSupervisedDataset(processor.tokenizer)
        return dict(
            train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
        )
    data_collator = DataCollatorForSupervisedDataset(processor.tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


if __name__ == "__main__":
    pass
