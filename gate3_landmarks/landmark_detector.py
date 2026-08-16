#!/usr/bin/env python3
"""
Gate 3: Landmark Detector
Uses a VLM to identify navigable landmarks at key frames along the path.
These landmarks feed into Gate 4's instruction generator.

Inputs:  rendered frames (from Gate 1) + key_frame_indices (from Gate 2)
Outputs: {frame_idx: {room, landmarks: [], direction}}
"""
import base64
import json
import re
from pathlib import Path
from typing import List, Dict, Optional

LANDMARK_PROMPT = """\
You are helping generate navigation instructions for an indoor robot.

Look at this image from a navigation path and identify:
1. ROOM TYPE: What kind of room or space is this? (bedroom, living room, kitchen, hallway, dining room, bathroom, staircase, etc.)
2. LANDMARKS: Name up to 4 prominent objects or features that could serve as navigation landmarks. Focus on: furniture, doorways, rugs, stairs, counters, columns, appliances, distinctive decor.
3. PATH DIRECTION: Is the navigable path going mostly straight-ahead, or is there a visible turn?

Respond ONLY in this exact format (one line each):
ROOM: <room_type>
LANDMARKS: <item1>, <item2>, <item3>
DIRECTION: <straight | left | right | unclear>
"""


def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class LandmarkDetector:
    def __init__(self, backend: str = "gemma"):
        self.backend = backend
        self._model = None

    def _load_gemma(self):
        if self._model is not None:
            return
        from transformers import AutoProcessor, AutoModelForImageTextToText
        import torch
        model_id = "google/gemma-3-27b-it"
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )

    def _detect_gemma(self, image_path: str) -> Dict:
        from PIL import Image
        import torch
        self._load_gemma()
        image = Image.open(image_path).convert("RGB")
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": LANDMARK_PROMPT},
            ]}
        ]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[image], return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=128, do_sample=False)
        response = self._processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return self._parse_response(response)

    def _detect_gpt4o(self, image_path: str) -> Dict:
        import openai
        client = openai.OpenAI()
        b64 = image_to_base64(image_path)
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": LANDMARK_PROMPT},
                ],
            }],
            max_tokens=128,
        )
        return self._parse_response(resp.choices[0].message.content)

    @staticmethod
    def _parse_response(response: str) -> Dict:
        result = {"room": "unknown", "landmarks": [], "direction": "unclear"}
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("ROOM:"):
                result["room"] = line[5:].strip().lower()
            elif line.startswith("LANDMARKS:"):
                raw = line[10:].strip()
                landmarks = [l.strip().lower() for l in re.split(r"[,;]", raw) if l.strip()]
                result["landmarks"] = landmarks[:4]
            elif line.startswith("DIRECTION:"):
                result["direction"] = line[10:].strip().lower()
        return result

    def detect(self, image_path: str) -> Dict:
        if not Path(image_path).exists():
            return {"room": "unknown", "landmarks": [], "direction": "unclear", "error": "image not found"}
        if self.backend == "gemma":
            return self._detect_gemma(image_path)
        elif self.backend == "gpt4o":
            return self._detect_gpt4o(image_path)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def detect_key_frames(
        self,
        frames_dir: Path,
        key_frame_indices: List[int],
        frame_prefix: str = "frame",
    ) -> Dict[int, Dict]:
        """
        Detect landmarks at all key frame indices.
        Returns {frame_idx: landmark_dict}
        """
        results = {}
        for idx in key_frame_indices:
            image_path = frames_dir / f"{frame_prefix}_{idx:04d}_rgb.png"
            if not image_path.exists():
                # Try without _rgb suffix
                image_path = frames_dir / f"{frame_prefix}_{idx:04d}.png"
            if not image_path.exists():
                results[idx] = {"room": "unknown", "landmarks": [], "direction": "unclear", "error": "not found"}
                continue
            results[idx] = self.detect(str(image_path))
            print(f"  Frame {idx}: room={results[idx]['room']}, landmarks={results[idx]['landmarks']}")
        return results


def landmark_context_for_prompt(
    frame_results: Dict[int, Dict],
    primitives: List[Dict],
    key_frame_indices: List[int],
) -> str:
    """
    Build structured context string for Gate 4's instruction generator.
    Maps key frames to motion primitives for coherent description.
    """
    lines = []
    turn_frame_map = {}

    # Map turn primitives to frames
    turn_count = 0
    for p in primitives:
        if "turn" in p.get("type", ""):
            if turn_count < len(key_frame_indices) - 2:
                turn_frame_map[key_frame_indices[turn_count + 1]] = p
            turn_count += 1

    for i, frame_idx in enumerate(key_frame_indices):
        fr = frame_results.get(frame_idx, {})
        room = fr.get("room", "unknown")
        landmarks = fr.get("landmarks", [])
        landmark_str = ", ".join(landmarks) if landmarks else "no clear landmarks"

        if i == 0:
            lines.append(f"START: {room} | visible: {landmark_str}")
        elif i == len(key_frame_indices) - 1:
            lines.append(f"GOAL: {room} | stop near: {landmark_str}")
        elif frame_idx in turn_frame_map:
            turn = turn_frame_map[frame_idx]
            direction = "left" if "left" in turn["type"] else "right"
            lines.append(f"TURN {direction.upper()} ({turn['angle_deg']:.0f}°): {room} | landmark: {landmark_str}")
        else:
            lines.append(f"WAYPOINT: {room} | visible: {landmark_str}")

    return "\n".join(lines)
