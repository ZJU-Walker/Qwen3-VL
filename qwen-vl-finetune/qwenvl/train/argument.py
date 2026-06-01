import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    video_max_pixels: int = field(default=1024 * 28 * 28)
    video_min_pixels: int = field(default=256 * 28 * 28)
    video_fps: float = 2
    # CSV / root / camera are empty by default: they come from the dataset
    # registry entry (qwenvl/data/__init__.py). Set any of these to override
    # the registry for a one-off run.
    robot_subtask_csv: str = field(default="")
    robot_subtask_root: str = field(default="")
    robot_subtask_camera: str = field(default="")
    # Empty by default: the prompt comes from the dataset registry entry
    # (qwenvl/data/__init__.py). Set this only to override the registry per-run.
    robot_subtask_prompt: str = field(default="")
    robot_subtask_history_seconds: float = field(default=5.0)
    robot_subtask_num_frames: int = field(default=5)
    # Lookahead horizon (frames): label = subtask at (current_frame + this), clamped
    # to episode end. 15 frames = 0.5s at 30fps. Makes the model anticipate the next
    # subtask rather than describe the current frame.
    robot_subtask_lookahead_frames: int = field(default=15)
    robot_subtask_epoch_size: int = field(default=4096)
    robot_subtask_split: str = field(default="train")
    robot_subtask_train_episodes: str = field(default="0:52")
    robot_subtask_val_episodes: str = field(default="52:61")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None

    ## Lora config
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.0)
