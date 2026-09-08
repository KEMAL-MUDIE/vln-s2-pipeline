#!/usr/bin/env python3
"""
Gate 4: Gemma vLLM Backend
OpenAI-compatible client for the Gemma 4 31B AWQ server at 10.77.32.231:8000.
Supports both text-only and vision (image) modes.
"""
import asyncio
import base64
import json
import re
from pathlib import Path
from typing import List, Optional, Dict

from openai import AsyncOpenAI, OpenAI

VLLM_BASE_URL = "http://10.77.32.231:8000/v1"
VLLM_MODEL = "cyankiwi/gemma-4-31B-it-AWQ-4bit"
VLLM_API_KEY = "EMPTY"
MAX_CONTEXT = 4096
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.3


def _load_prompts() -> dict:
    prompts_path = Path(__file__).parent.parent / "configs" / "vlm_prompts.yaml"
    try:
        import yaml
        with open(prompts_path) as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_text_only_messages(motion_sequence: str, scene_context: str) -> List[dict]:
    """Build chat messages for text-only instruction generation."""
    prompts = _load_prompts()
    template = prompts.get("instruction_generation_text_only", "")
    if not template:
        template = (
            "You write R2R VLN navigation instructions from path geometry.\n\n"
            "PATH: {motion_sequence}\n\n"
            "Write a 1-4 sentence navigation instruction (15-40 words). "
            "Include specific indoor landmarks and a stop condition. "
            "Write ONLY the instruction:"
        )
    prompt = template.format(motion_sequence=motion_sequence, scene_context=scene_context)
    return [{"role": "user", "content": prompt}]


def build_gate3_messages(motion_sequence: str) -> List[dict]:
    """Build chat messages using the gate3-specific GT-aligned prompt template."""
    prompts = _load_prompts()
    template = prompts.get("instruction_generation_gate3", "")
    if not template:
        template = (
            "You write R2R VLN navigation instructions.\n\n"
            "PATH:\n{motion_sequence}\n\n"
            "Write 1-4 sentences using room transitions, minimal explicit turns. "
            "Include stop condition. Write ONLY the instruction:"
        )
    prompt = template.format(motion_sequence=motion_sequence)
    return [{"role": "user", "content": prompt}]


def build_gate3_v7_messages(motion_sequence: str) -> List[dict]:
    """Build chat messages using the gate3 v7 prompt (dynamic sentence count + 2-turn coverage)."""
    prompts = _load_prompts()
    template = prompts.get("instruction_generation_gate3_v7", "")
    if not template:
        template = (
            "You write R2R VLN navigation instructions.\n\n"
            "PATH:\n{motion_sequence}\n\n"
            "Write EXACTLY the number of sentences in Target_sentences. "
            "Use room transitions, include stop condition. Write ONLY the instruction:"
        )
    prompt = template.format(motion_sequence=motion_sequence)
    return [{"role": "user", "content": prompt}]


def build_gate3_v8_messages(motion_sequence: str) -> List[dict]:
    """Build chat messages using the gate3 v8 prompt (probabilistic sentences + 4-sentence structure)."""
    prompts = _load_prompts()
    template = prompts.get("instruction_generation_gate3_v8", "")
    if not template:
        template = (
            "You write R2R VLN navigation instructions.\n\n"
            "PATH:\n{motion_sequence}\n\n"
            "Write EXACTLY the number of sentences in Target_sentences. "
            "For 4 sentences: each Waypoint gets its own sentence, last sentence is Stop. "
            "Use room transitions, include stop condition. Write ONLY the instruction:"
        )
    prompt = template.format(motion_sequence=motion_sequence)
    return [{"role": "user", "content": prompt}]


def build_vision_messages(motion_sequence: str, scene_context: str, image_paths: List[str]) -> List[dict]:
    """Build chat messages with embedded key-frame images."""
    prompts = _load_prompts()
    template = prompts.get("instruction_generation", "")
    if not template:
        template = (
            "You write R2R VLN navigation instructions.\n\n"
            "PATH: {motion_sequence}\n\nSCENE: {scene_context}\n\n"
            "Write a 1-4 sentence instruction (15-40 words) with landmarks and stop condition:"
        )
    prompt = template.format(motion_sequence=motion_sequence, scene_context=scene_context)

    content = []
    for img_path in image_paths[:3]:  # max 3 images within context limit
        b64 = image_to_base64(img_path)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def generate_one(motion_sequence: str, scene_context: str = "", image_paths: List[str] = None) -> str:
    """Synchronous single-call generation."""
    client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    if image_paths:
        messages = build_vision_messages(motion_sequence, scene_context, image_paths)
    else:
        messages = build_text_only_messages(motion_sequence, scene_context)
    resp = client.chat.completions.create(
        model=VLLM_MODEL, messages=messages,
        max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
    )
    return resp.choices[0].message.content.strip()


async def generate_one_async(
    client: AsyncOpenAI,
    episode_id: int,
    motion_sequence: str,
    scene_context: str = "",
    image_paths: List[str] = None,
    semaphore: asyncio.Semaphore = None,
    prompt_type: str = "text_only",
    temperature: float = None,
) -> Dict:
    """Async single-call generation with semaphore throttling."""
    temp = temperature if temperature is not None else TEMPERATURE
    async with semaphore:
        if image_paths:
            messages = build_vision_messages(motion_sequence, scene_context, image_paths)
        elif prompt_type == "gate3":
            messages = build_gate3_messages(motion_sequence)
        elif prompt_type == "gate3_v7":
            messages = build_gate3_v7_messages(motion_sequence)
        elif prompt_type == "gate3_v8":
            messages = build_gate3_v8_messages(motion_sequence)
        else:
            messages = build_text_only_messages(motion_sequence, scene_context)
        try:
            resp = await client.chat.completions.create(
                model=VLLM_MODEL, messages=messages,
                max_tokens=MAX_NEW_TOKENS, temperature=temp,
            )
            text = resp.choices[0].message.content.strip()
            return {"episode_id": episode_id, "text": text, "ok": True, "error": None}
        except Exception as e:
            return {"episode_id": episode_id, "text": "", "ok": False, "error": str(e)}


async def generate_batch_async(
    tasks: List[Dict],   # [{"episode_id": int, "motion_sequence": str, "scene_context": str}]
    concurrency: int = 16,
    progress_every: int = 50,
    prompt_type: str = "text_only",
) -> Dict[int, str]:
    """
    Generate instructions for all tasks concurrently.
    Returns {episode_id: instruction_text}.
    prompt_type: "text_only" | "gate3" | "vision" (vision needs image_paths in task)
    """
    import time
    client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()

    coros = [
        generate_one_async(
            client, t["episode_id"], t["motion_sequence"],
            t.get("scene_context", ""), t.get("image_paths"),
            sem, prompt_type=t.get("prompt_type", prompt_type),
            temperature=t.get("temperature"),
        )
        for t in tasks
    ]

    results = {}
    errors = 0
    for i, coro in enumerate(asyncio.as_completed(coros)):
        r = await coro
        if r["ok"]:
            results[r["episode_id"]] = r["text"]
        else:
            errors += 1
            results[r["episode_id"]] = ""
        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(tasks) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(tasks)}] ok={len(results)-errors} err={errors} "
                  f"rate={rate:.1f}/s ETA={eta/60:.1f}m")

    elapsed = time.time() - t0
    print(f"Completed {len(tasks)} episodes in {elapsed:.1f}s ({len(tasks)/elapsed:.1f}/s). Errors: {errors}")
    return results
