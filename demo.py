"""
demo.py - AVIS demo for Qwen2.5-VL
    

Usage examples:
    # 1) Vanilla (no pruning, single pass, thinking on by default)
    python demo.py --model-path Qwen/Qwen2.5-VL-7B-Instruct --image-path example.jpg --prompt "What is shown in this image?"

    # 2) KDV pruning (VCS), single pass
    python demo.py --model-path Qwen/Qwen2.5-VL-7B-Instruct \
        --image-path example.jpg \
        --prompt "What is shown in this image?" \
        --enable-kdv --kdv-ratio 0.5

    # 3) Self-consistency with K rollouts (VRS), no pruning
    python demo.py --model-path Qwen/Qwen2.5-VL-7B-Instruct \
        --image-path example.jpg \
        --prompt "What is shown in this image?" \
        --num-rollouts 7

    # 4) KDV pruning + self-consistency
    python demo.py --model-path Qwen/Qwen2.5-VL-7B-Instruct \
        --image-path example.jpg \
        --prompt "What is shown in this image?" \
        --enable-kdv --kdv-ratio 0.5 --num-rollouts 7

    # 5) AVIS
    python demo.py --model-path Qwen/Qwen2.5-VL-7B-Instruct \
        --image-path example.jpg \
        --prompt "What is shown in this image?" \
        --enable-kdv --kdv-ratio 0.5 \
        --enable-policy --policy-path ./policy_checkpoint.pt
"""

from __future__ import annotations

import argparse
import os
import re
import time
from collections import Counter

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


def ensure_image_url(path: str) -> str:
    """Convert a local file path to a file:// URL if needed."""
    prefixes = ("http://", "https://", "file://", "data:image;")
    if any(path.startswith(p) for p in prefixes):
        return path
    if os.path.exists(path):
        return "file://" + os.path.abspath(path)
    raise FileNotFoundError(f"Image not found: {path}")


def ensure_video_url(path: str) -> str:
    """Convert a local file path to a file:// URL if needed."""
    prefixes = ("http://", "https://", "file://", "data:video;")
    if any(path.startswith(p) for p in prefixes):
        return path
    if os.path.exists(path):
        return "file://" + os.path.abspath(path)
    raise FileNotFoundError(f"Video not found: {path}")


def extract_answer_from_thinking(response: str) -> str:
    """Extract content inside <answer>...</answer> tags, if present."""
    matches = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if matches:
        return "".join(matches)
    return response


def majority_vote(responses: list[str]) -> str:
    """Self-consistency: return the most common answer via majority vote."""
    counts = Counter(responses)
    return counts.most_common(1)[0][0]


def load_model(args):
    """Load Qwen2.5-VL model and processor."""
    print(f"Loading model from {args.model_path} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    return model, processor


def load_policy_model(args, device):
    """Optionally load the lightweight difficulty predictor."""
    from policy.policy_model import PolicyModel

    print(f"Loading policy model from {args.policy_path} ...")
    policy = PolicyModel(args.policy_path, device=device)
    policy.eval()
    return policy


def build_messages(args):
    """Construct the chat messages list from CLI arguments."""
    content = []

    if args.image_path is not None:
        item = {"type": "image", "image": ensure_image_url(args.image_path)}
        if args.min_pixels is not None:
            item["min_pixels"] = args.min_pixels
        if args.max_pixels is not None:
            item["max_pixels"] = args.max_pixels
        content.append(item)
    elif args.video_path is not None:
        item = {"type": "video", "video": ensure_video_url(args.video_path)}
        if args.min_pixels is not None:
            item["min_pixels"] = args.min_pixels
        if args.max_pixels is not None:
            item["max_pixels"] = args.max_pixels
        if args.fps is not None:
            item["fps"] = args.fps
        if args.nframe is not None:
            item["nframes"] = args.nframe
        content.append(item)
    else:
        raise ValueError("Provide either --image-path or --video-path")

    prompt_text = args.prompt
    if args.enable_thinking:
        prompt_text += (
            "\nFirst output the thinking process in <think> </think> tags "
            "and then output the final answer in <answer> </answer> tags."
        )
    content.append({"type": "text", "text": prompt_text})

    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": content})
    return messages



@torch.no_grad()
def generate(model, processor, messages, args, policy_model=None):
    """
    Knobs:
        enable_kdv / kdv_ratio  →  Visual Context Scaling (VCS)
        num_rollouts            →  Visual Reasoning Scaling (VRS)
        enable_policy           →  Adaptive K via difficulty predictor
    """
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(
        [messages], tokenize=False, add_generation_prompt=True
    )
    images, videos = process_vision_info([messages])
    inputs = processor(
        text=text, images=images, videos=videos,
        padding=True, return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    K = args.num_rollouts

    if args.enable_policy and policy_model is not None:
        # Policy predicts adaptive K from visual embeddings
        if "pixel_values" in inputs:
            embeds, _, _ = model.model.get_image_features(
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs.get("image_grid_thw"),
                enable_kdv=args.enable_kdv,
            )
        elif "pixel_values_videos" in inputs:
            embeds, _, _ = model.model.get_image_features(
                pixel_values=inputs["pixel_values_videos"],
                image_grid_thw=inputs.get("image_grid_thw"),
                enable_kdv=args.enable_kdv,
            )
        else:
            raise ValueError("No visual input found in processed inputs.")
        embeds = torch.stack(embeds, dim=0)
        K = int(policy_model(embeds))
        print(f"  Policy predicted K = {K}")

    if K > 1:
        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=None,
            num_return_sequences=K,
            enable_kdv=args.enable_kdv,
            kdv_ratio=args.kdv_ratio,
            enable_policy=args.enable_policy,
        )
    else:
        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=0.01,
            top_p=0.001,
            top_k=1,
            num_return_sequences=1,
            enable_kdv=args.enable_kdv,
            kdv_ratio=args.kdv_ratio,
            enable_policy=args.enable_policy,
        )

    t0 = time.time()
    generated_ids = model.generate(**inputs, **gen_kwargs)
    elapsed = time.time() - t0

    input_len = inputs.input_ids.shape[1]
    generated_ids = [
        out[input_len:] for out in generated_ids
    ]
    responses = processor.tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )

    if args.enable_thinking:
        responses = [extract_answer_from_thinking(r) for r in responses]
    responses = [r.strip() for r in responses]

    return responses, K, elapsed



def main():
    parser = argparse.ArgumentParser(
        description="AVIS Demo for Qwen2.5-VL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- model ---
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path or HF hub ID for Qwen2.5-VL model")

    # --- input ---
    parser.add_argument("--image-path", type=str, default=None,
                        help="Path or URL to input image")
    parser.add_argument("--video-path", type=str, default=None,
                        help="Path or URL to input video")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt / question")
    parser.add_argument("--system-prompt", type=str, default=None,
                        help="Optional system prompt")

    parser.add_argument("--enable-kdv", action="store_true", default=False,
                        help="Enable KDV visual token pruning")
    parser.add_argument("--kdv-ratio", type=float, default=0.5,
                        help="Fraction of visual tokens to RETAIN (0, 1]. "
                             "Lower = more aggressive pruning. Default: 0.5")

    parser.add_argument("--num-rollouts", type=int, default=1,
                        help="Number of self-consistency rollouts K. "
                             "K=1 is single-pass; K>1 triggers majority vote. Default: 1")

    parser.add_argument("--enable-policy", action="store_true", default=False,
                        help="Use the learned difficulty predictor to set K adaptively")
    parser.add_argument("--policy-path", type=str, default=None,
                        help="Path to the policy model checkpoint (required if --enable-policy)")

    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="Maximum number of new tokens to generate per rollout")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True,
                        help="Append chain-of-thought instruction with <think>/<answer> tags "
                             "(enabled by default; use --no-enable-thinking to disable)")

    parser.add_argument("--min-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=16384 * 28 * 28)

    parser.add_argument("--fps", type=float, default=None,
                        help="Frames per second for video sampling")
    parser.add_argument("--nframe", type=int, default=None,
                        help="Max number of frames to sample from video")

    args = parser.parse_args()

    if args.image_path is None and args.video_path is None:
        parser.error("Provide either --image-path or --video-path")
    if args.image_path is not None and args.video_path is not None:
        parser.error("Provide only one of --image-path or --video-path, not both")
    if args.enable_policy and not args.policy_path:
        parser.error("--policy-path is required when --enable-policy is set")

    mode_parts = []
    if args.enable_kdv:
        mode_parts.append(f"KDV, retain {args.kdv_ratio:.0%}")
    if args.enable_policy:
        mode_parts.append("Policy-adaptive K")
    elif args.num_rollouts > 1:
        mode_parts.append(f"K={args.num_rollouts})")
    mode_str = " + ".join(mode_parts) if mode_parts else "Vanilla"

    print("=" * 60)
    print("  AVIS Demo — Adaptive Test-Time Scaling for Vision-Language Models")
    print("=" * 60)
    print(f"  Model       : {args.model_path}")
    print(f"  Mode        : {mode_str}")
    if args.image_path:
        print(f"  Image       : {args.image_path}")
    if args.video_path:
        print(f"  Video       : {args.video_path}")
    print(f"  Prompt      : {args.prompt}")
    print(f"  Thinking    : {args.enable_thinking}")
    print(f"  Max tokens  : {args.max_new_tokens}")
    print("=" * 60)

    model, processor = load_model(args)
    policy_model = None
    if args.enable_policy:
        policy_model = load_policy_model(args, device=model.device)

    messages = build_messages(args)

    print("\nGenerating ...")
    responses, K, elapsed = generate(
        model, processor, messages, args, policy_model=policy_model,
    )

    print(f"\nDone in {elapsed:.2f}s  (K={K} rollout{'s' if K > 1 else ''})\n")

    if K > 1:
        print(f"--- {K} rollout responses ---")
        for i, r in enumerate(responses, 1):
            print(f"  [{i}] {r}")
        final = majority_vote(responses)
        print(f"\n--- Majority vote answer ---")
        print(f"  {final}")
    else:
        print(f"--- Response ---")
        print(f"  {responses[0]}")

    print()


if __name__ == "__main__":
    main()