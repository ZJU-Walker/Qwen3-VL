#!/usr/bin/env python3
"""Per-frame rollout on a RAW mp4 file (no LeRobot folder, no GT).

This is the standalone-clip variant of rollout_robot_subtask_nogt.py. Instead of
reading a LeRobot dataset by episode index, it takes a single --video path (any
mp4) and runs the SAME backward-history-window inference on every frame.

Use it to sanity-check a model on an arbitrary clip, e.g. the first 1s of an
episode, where you want to see exactly what the model predicts frame-by-frame
with no future frames available.

Example:
  python tools/rollout_robot_subtask_video.py \
    --model-name-or-path "$CKPT" \
    --processor-name-or-path Qwen/Qwen3-VL-8B-Instruct \
    --video /iris/projects/humanoid/trossen_data/plate_test/sanity_check/ep000000_cam_high_first1s.mp4 \
    --history-seconds 5 --num-frames 5 --bf16 \
    --output-dir "$CKPT/rollout_sanity_check"
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import (AutoProcessor, Qwen3VLForConditionalGeneration,
                          Qwen3VLMoeForConditionalGeneration)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

DEFAULT_PROMPT = ("Predict the subtask the robot should perform: put the green "
                  "block to the plate, or put the yellow block to the plate.")

# Populated in main() from --labels or --labels-csv. The dataset/task defines the
# label set, so nothing here is hardcoded to a specific task.
_VALID_LABELS: tuple[str, ...] = ()
# label -> (r, g, b); filled by _assign_label_colors() once labels are known.
LABEL_COLORS: dict[str, tuple[int, int, int]] = {}


def _labels_from_csv(csv_path: Path) -> list[str]:
    """Distinct `subtask` values from a segment CSV, sorted -- mirrors how the
    training dataset derives its label set, so rollout stays in sync."""
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if "subtask" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} has no 'subtask' column")
        return sorted({row["subtask"] for row in reader if row.get("subtask")})


def _assign_label_colors(labels: Sequence[str]) -> dict[str, tuple[int, int, int]]:
    """Evenly spaced, distinct hues for any number of classes."""
    colors = {}
    n = max(1, len(labels))
    for i, label in enumerate(labels):
        r, g, b = colorsys.hsv_to_rgb(i / n, 0.65, 0.9)
        colors[label] = (int(r * 255), int(g * 255), int(b * 255))
    return colors


def load_model(model_name_or_path: str, bf16: bool, attn: str):
    name = Path(model_name_or_path.rstrip("/")).name.lower()
    kwargs = {"attn_implementation": attn, "device_map": "auto",
              "dtype": torch.bfloat16 if bf16 else "auto"}
    cls = (Qwen3VLMoeForConditionalGeneration
           if ("qwen3" in model_name_or_path.lower() and "a" in name)
           else Qwen3VLForConditionalGeneration)
    model = cls.from_pretrained(model_name_or_path, **kwargs)
    model.eval()
    return model


def normalize(text: str) -> str:
    text = text.strip()
    return text if text in _VALID_LABELS else f"?{text}"


def decode_frame_at(decoder, frame_idx: int) -> np.ndarray:
    frame = decoder.get_frames_at(indices=[int(frame_idx)]).data  # (1,C,H,W)
    arr = frame[0].permute(1, 2, 0).cpu().numpy()
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _font(size: int):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_overlay(frame: np.ndarray, *, frame_idx: int, pred: str) -> np.ndarray:
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    color = LABEL_COLORS.get(pred, (235, 235, 235))
    big = _font(max(20, h // 18))
    small = _font(max(14, h // 30))

    line_h = big.size + 6
    banner_h = small.size + line_h + 14
    bar = Image.new("RGBA", (w, banner_h), (0, 0, 0, 150))
    img.paste(Image.new("RGB", (w, banner_h), (0, 0, 0)), (0, 0), mask=bar.split()[3])

    y = 6
    draw.text((12, y), f"frame {frame_idx}", fill=(235, 235, 235), font=small)
    y += small.size + 6
    draw.text((12, y), f"pred: {pred}", fill=color, font=big)
    return np.asarray(img)


def sampled_indices(clip_start: int, clip_end: int, nframes: int) -> list[int]:
    """The actual frame indices the processor samples for a window.

    Qwen samples nframes frames evenly across [video_start, video_end], i.e.
    across [clip_start, clip_end] in frame units. This mirrors that so the
    visualization shows the frames the model truly saw, not an approximation.
    """
    if nframes <= 1:
        return [clip_start]
    return [int(round(clip_start + (clip_end - clip_start) * k / (nframes - 1)))
            for k in range(nframes)]


def _label_strip(img: Image.Image, text: str, color=(235, 235, 235)) -> None:
    """Draw a small caption strip at the top-left of img (in place)."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    f = _font(max(12, h // 22))
    strip_h = f.size + 8
    bar = Image.new("RGBA", (w, strip_h), (0, 0, 0, 160))
    img.paste(Image.new("RGB", (w, strip_h), (0, 0, 0)), (0, 0), mask=bar.split()[3])
    draw.text((6, 3), text, fill=color, font=f)


def build_panel(past_frames: list[tuple[int, np.ndarray]],
                cur_idx: int, cur_frame: np.ndarray, pred: str,
                tile_h: int = 240) -> np.ndarray:
    """Composite: past frames stacked on the LEFT, current frame + pred on RIGHT.

    past_frames: list of (frame_index, frame_array) the model actually saw.
    The current frame is the last sampled frame; we still show it big on the right.
    """
    def fit(arr: np.ndarray, target_h: int) -> Image.Image:
        im = Image.fromarray(arr).convert("RGB")
        w, h = im.size
        return im.resize((max(1, int(round(w * target_h / h))), target_h))

    # LEFT column: the past frames (everything except the current frame), stacked.
    past = [pf for pf in past_frames if pf[0] != cur_idx] or past_frames
    small_h = tile_h // max(1, min(len(past), 4)) if len(past) > 1 else tile_h
    small_h = max(60, min(tile_h, small_h))
    left_tiles = []
    for fidx, arr in past:
        tile = fit(arr, small_h)
        _label_strip(tile, f"hist f{fidx}")
        left_tiles.append(np.asarray(tile))
    left_w = max(t.shape[1] for t in left_tiles)
    left_canvas = np.zeros((sum(t.shape[0] for t in left_tiles) + 4 * (len(left_tiles) - 1),
                            left_w, 3), dtype=np.uint8)
    y = 0
    for t in left_tiles:
        left_canvas[y:y + t.shape[0], 0:t.shape[1]] = t
        y += t.shape[0] + 4

    # RIGHT: big current frame with the prediction banner.
    big_h = max(left_canvas.shape[0], tile_h)
    cur = fit(cur_frame, big_h)
    cur_arr = draw_overlay(np.asarray(cur), frame_idx=cur_idx, pred=pred)

    # Pad both columns to equal height and concatenate horizontally.
    H = max(left_canvas.shape[0], cur_arr.shape[0])
    def pad_h(a):
        if a.shape[0] == H:
            return a
        out = np.zeros((H, a.shape[1], 3), dtype=np.uint8)
        out[:a.shape[0]] = a
        return out
    gap = np.zeros((H, 8, 3), dtype=np.uint8)
    return np.concatenate([pad_h(left_canvas), gap, pad_h(cur_arr)], axis=1)


def write_video(frames: list[np.ndarray], path: Path, fps: float) -> None:
    import torchvision.io as tvio
    # Panels can vary in size (nframes grows near the episode start), so pad every
    # frame to the max H,W and make dims even (libx264 / yuv420p requires it).
    H = max(f.shape[0] for f in frames)
    W = max(f.shape[1] for f in frames)
    H += H % 2
    W += W % 2
    padded = []
    for f in frames:
        out = np.zeros((H, W, 3), dtype=np.uint8)
        out[:f.shape[0], :f.shape[1]] = f
        padded.append(out)
    tensor = torch.from_numpy(np.ascontiguousarray(np.stack(padded)))
    tvio.write_video(str(path), tensor, fps=max(1, int(round(fps))))


def build_messages(video_path: str, video_start: float, video_end: float,
                   nframes: int, prompt: str) -> list[dict]:
    return [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path,
             "video_start": video_start, "video_end": video_end, "nframes": nframes},
            {"type": "text", "text": prompt},
        ],
    }]


def build_inputs_from_frames(processor, frames: list[np.ndarray],
                             frame_indices: list[int], fps: float, prompt: str):
    """Tokenize a prompt + a list of PRE-DECODED frames (exact, no re-sampling).

    Passing frames directly (videos=[...] with do_sample_frames=False) lets us
    control the precise frame count, including front-padded repeats, instead of
    letting the processor sample from the video file. We supply VideoMetadata with
    the real frame indices + fps so the per-frame timestamps in the prompt match
    where the frames actually came from.
    """
    from transformers.video_utils import VideoMetadata
    video = np.stack(frames)  # (T,H,W,C) uint8
    h, w = video.shape[1], video.shape[2]
    md = VideoMetadata(total_num_frames=len(frames), fps=float(fps),
                       frames_indices=list(frame_indices), duration=None,
                       height=h, width=w)
    messages = [{
        "role": "user",
        "content": [{"type": "video"}, {"type": "text", "text": prompt}],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], videos=[video], video_metadata=[md],
                     do_sample_frames=False, return_tensors="pt")


def window_frame_indices(t: int, *, num_frames: int, stride_frames: int | None,
                         history_frames: int, ep_len: int, pad: bool) -> list[int]:
    """The exact frame indices to feed the model for current frame t.

    Always returns `num_frames` indices when pad=True: a backward strided (or
    history-spanning) window ending at t, FRONT-padded by repeating the oldest
    available frame whenever there isn't enough real history near the episode
    start. This guarantees the model receives a constant frame count every step.

    With pad=False it returns however many real frames fit (still even-count,
    matching the training data processor's nframes rule).
    """
    if stride_frames is not None:
        # strided window ending at t, going backward in steps of stride_frames
        idx = [t - k * stride_frames for k in range(num_frames)]
        idx = [i for i in idx if i >= 0]
        idx = list(reversed(idx))  # oldest -> newest, newest == t
    else:
        # even-sampled window across [max(0,t-history+1), t]
        clip_start = max(0, t - history_frames + 1)
        clip_end = t
        total = clip_end - clip_start + 1
        nf = min(num_frames, total)
        idx = sampled_indices(clip_start, clip_end, nf)

    if pad:
        # front-pad by repeating the oldest frame up to exactly num_frames
        if len(idx) < num_frames:
            idx = [idx[0]] * (num_frames - len(idx)) + idx
    else:
        # keep an even count to match training's nframes rule
        if len(idx) % 2 == 1 and len(idx) > 2:
            idx = idx[1:]
        if len(idx) < 2:
            idx = [idx[0], min(ep_len - 1, idx[0] + 1)]
    return idx


def main() -> None:
    global _VALID_LABELS
    p = argparse.ArgumentParser()
    p.add_argument("--model-name-or-path", required=True)
    p.add_argument("--processor-name-or-path", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--video", type=Path, required=True, help="Path to a raw mp4.")
    p.add_argument("--fps", type=float, default=None,
                   help="Override fps; default reads it from the video container.")
    p.add_argument("--history-seconds", type=float, default=5.0)
    p.add_argument("--num-frames", type=int, default=5)
    p.add_argument("--sample-stride-seconds", type=float, default=None,
                   help="If set, sample exactly --num-frames frames spaced this many "
                        "seconds apart, ending at the current frame t. This keeps the "
                        "sampled frames dense and RECENT instead of smearing --num-frames "
                        "frames across the whole --history-seconds window (which pins the "
                        "window to the episode start and freezes the prediction). "
                        "Recommended: ~0.3-0.5s for 30fps data.")
    p.add_argument("--prompt", default=DEFAULT_PROMPT,
                   help="MUST match the checkpoint's training prompt.")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Valid label set. Predictions outside it are flagged with '?'. "
                        "If omitted, derived from --labels-csv.")
    p.add_argument("--labels-csv", type=Path, default=None,
                   help="Segment CSV to derive the label set from (its distinct "
                        "'subtask' values), matching how the dataset is built. Use the "
                        "same CSV the checkpoint was trained on.")
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--pad-frames", action="store_true",
                   help="Always feed exactly --num-frames frames to the model. Near the "
                        "episode start, where there isn't enough history, FRONT-pad by "
                        "repeating the oldest available frame. Removes the variable "
                        "frame-count confound; frames are passed pre-decoded so the "
                        "processor uses them verbatim (no internal re-sampling).")
    p.add_argument("--panel", action="store_true",
                   help="Render each output frame as a side-by-side panel: the past "
                        "frames the model actually saw on the LEFT, current frame + "
                        "prediction on the RIGHT.")
    p.add_argument("--save-panel-frames", action="store_true",
                   help="Also save each panel as a PNG under <output-dir>/panels/.")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--attn-implementation", default="flash_attention_2")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    if args.labels:
        labels = list(args.labels)
    elif args.labels_csv:
        labels = _labels_from_csv(args.labels_csv)
    else:
        sys.exit("Provide --labels or --labels-csv to define the valid label set.")
    _VALID_LABELS = tuple(labels)
    LABEL_COLORS.update(_assign_label_colors(labels))
    print(f"Labels ({len(labels)}): {labels}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.video.is_file():
        sys.exit(f"video not found: {args.video}")

    from torchcodec.decoders import VideoDecoder
    decoder = VideoDecoder(str(args.video))
    meta = decoder.metadata
    fps = float(args.fps) if args.fps else float(meta.average_fps)
    ep_len = int(meta.num_frames)
    history_frames = max(2, int(round(args.history_seconds * fps)))

    processor = AutoProcessor.from_pretrained(args.processor_name_or_path)
    model = load_model(args.model_name_or_path, args.bf16, args.attn_implementation)
    device = next(model.parameters()).device

    print(f"\n=== {args.video.name}  ({ep_len} frames @ {fps}fps) ===")

    out_frames = []
    pred_counts: dict[str, int] = {}
    per_frame = []
    stride_frames = (max(1, int(round(args.sample_stride_seconds * fps)))
                     if args.sample_stride_seconds else None)
    for t in range(0, ep_len, args.frame_stride):
        # Exact frame indices the model will see (oldest -> newest, newest == t).
        # With --pad-frames this is ALWAYS exactly num_frames (front-padded).
        sidx = window_frame_indices(
            t, num_frames=args.num_frames, stride_frames=stride_frames,
            history_frames=history_frames, ep_len=ep_len, pad=args.pad_frames)
        # cache decode: distinct indices only (padding repeats the oldest frame)
        cache = {i: decode_frame_at(decoder, i) for i in set(sidx)}
        frames = [cache[i] for i in sidx]

        if args.pad_frames:
            # Feed the exact (possibly padded) frames pre-decoded; no re-sampling.
            inputs = build_inputs_from_frames(processor, frames, sidx, fps, args.prompt)
        else:
            # Faithful video-segment path: let the processor sample the window.
            clip_start, clip_end, nframes = sidx[0], sidx[-1], len(sidx)
            messages = build_messages(str(args.video), clip_start / fps,
                                      clip_end / fps, nframes, args.prompt)
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt")
        inputs = {k: (v.to(device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        out_ids = gen[:, inputs["input_ids"].shape[1]:]
        pred = normalize(processor.batch_decode(out_ids, skip_special_tokens=True)[0])
        pred_counts[pred] = pred_counts.get(pred, 0) + 1
        per_frame.append({"frame": t, "sampled_frames": sidx,
                          "nframes": len(sidx), "pred": pred})

        if args.panel:
            # past frames on the LEFT, current frame + pred on the RIGHT.
            decoded = [(i, cache[i]) for i in sidx]
            panel = build_panel(decoded, t, cache[t], pred)
            out_frames.append(panel)
            if args.save_panel_frames:
                panel_dir = args.output_dir / "panels"
                panel_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(panel).save(panel_dir / f"frame_{t:06d}.png")
        else:
            out_frames.append(draw_overlay(cache[t], frame_idx=t, pred=pred))
        print(f"  frame {t:>4}/{ep_len}  frames={sidx}  nf={len(sidx)}  pred={pred}")

    out_path = args.output_dir / f"{args.video.stem}_rollout.mp4"
    write_video(out_frames, out_path, fps)
    print(f"\n  -> {out_path}  frames={len(out_frames)}  pred dist={pred_counts}")

    (args.output_dir / f"{args.video.stem}_per_frame.json").write_text(
        json.dumps({"video": str(args.video), "fps": fps,
                    "pred_distribution": pred_counts, "frames": per_frame}, indent=2))
    print("Wrote per-frame predictions JSON to", args.output_dir)


if __name__ == "__main__":
    main()
