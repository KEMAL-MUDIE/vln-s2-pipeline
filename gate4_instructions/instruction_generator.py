#!/usr/bin/env python3
"""
Gate 4: Instruction Generator
Core component: given images + motion primitives + landmark detections,
generates GT-quality VLN navigation instructions.

The VLM receives:
  - Images at key decision points (embedded in prompt)
  - Motion primitive sequence from Gate 2
  - Landmark context from Gate 3

It generates a 1-4 sentence instruction that matches GT style.
NO ground truth instruction is ever shown — this is pure generation.
"""
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from gate2_path.path_analyzer import analyze_path, primitives_to_text
from gate3_landmarks.landmark_detector import LandmarkDetector, landmark_context_for_prompt

# Quality thresholds (from GT statistics in Gate 0)
MIN_WORDS = 10
MAX_WORDS = 60
REQUIRE_STOP = True
REQUIRE_LANDMARK = True
MAX_RETRIES = 3

INSTRUCTION_PROMPT_TEMPLATE = """\
You are a navigation instruction writer for the R2R (Room-to-Room) Vision-Language Navigation dataset.

Your task: write a concise, natural navigation instruction for a human or robot to follow the path described below.

PATH DESCRIPTION (motion sequence):
{motion_sequence}

SCENE CONTEXT (what is visible at key points):
{scene_context}

REQUIREMENTS for the instruction:
- Write 1-4 short sentences, total 15-40 words
- Reference specific visible landmarks (furniture, rooms, doorways, rugs, etc.)
- Use natural turn language: "turn left", "make a left", "go left at"
- MUST include a stop condition: "stop at/near/by [landmark]" or "wait at [landmark]"
- Do NOT mention specific distances in meters or numbers
- Write as if giving directions to a person walking through the space
- Match the concise, direct style of these GT examples:
  * "Exit the bedroom, turn left, and walk past the gray couch. Stop near the rug."
  * "Walk through the kitchen doorway, turn right at the dining table, and stop in the hallway."
  * "Go straight through the living room past the couch. Turn left at the doorway and stop."

Write ONLY the instruction text, nothing else:
"""


def quality_check(instruction: str) -> Tuple[bool, List[str]]:
    """
    Check if generated instruction meets quality requirements.
    Returns (passes, list_of_failures).
    """
    failures = []
    words = instruction.split()

    if len(words) < MIN_WORDS:
        failures.append(f"too short ({len(words)} words, min {MIN_WORDS})")
    if len(words) > MAX_WORDS:
        failures.append(f"too long ({len(words)} words, max {MAX_WORDS})")

    if REQUIRE_STOP:
        stop_words = ["stop", "wait", "halt", "stand", "pause"]
        if not any(w in instruction.lower() for w in stop_words):
            failures.append("missing stop condition")

    if REQUIRE_LANDMARK:
        # Check for at least one non-generic landmark reference
        generic = {"there", "here", "it", "room", "place", "area", "space", "point"}
        words_lower = set(instruction.lower().split())
        has_landmark = len(words_lower - generic) > 5  # heuristic
        if not has_landmark:
            failures.append("no specific landmark referenced")

    return len(failures) == 0, failures


class InstructionGenerator:
    def __init__(self, backend: str = "gemma"):
        self.backend = backend
        self._model = None
        self._processor = None

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

    def _generate_gemma(self, prompt: str, images: List) -> str:
        import torch
        self._load_gemma()
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=images if images else None, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=256, do_sample=True,
                temperature=0.3, top_p=0.9,
            )
        response = self._processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()

    def _generate_gpt4o(self, prompt: str, image_paths: List[str]) -> str:
        import openai, base64
        client = openai.OpenAI()
        content = []
        for img_path in image_paths:
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        content.append({"type": "text", "text": prompt})
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": content}],
            max_tokens=256, temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    def generate(
        self,
        episode: Dict,
        frames_dir: Optional[Path] = None,
        path_analysis: Optional[Dict] = None,
        landmark_results: Optional[Dict] = None,
    ) -> Dict:
        """
        Generate navigation instruction for one episode.

        Args:
            episode: episode dict (needs reference_path, start_rotation)
            frames_dir: path to rendered frames directory (from Gate 1)
            path_analysis: output from Gate 2 (or computed here)
            landmark_results: output from Gate 3 (or None if no images available)

        Returns:
            {
              "instruction_text": str,
              "quality_ok": bool,
              "quality_failures": [],
              "attempts": int,
              "motion_sequence": str,
              "scene_context": str,
            }
        """
        # Gate 2: path analysis
        if path_analysis is None:
            path_analysis = analyze_path(
                episode["reference_path"],
                episode.get("start_rotation"),
            )

        motion_sequence = primitives_to_text(path_analysis["primitives"])

        # Gate 3: landmark detection (if frames available)
        scene_context = "No visual information available."
        images_for_prompt = []
        image_paths_for_prompt = []

        if frames_dir is not None and frames_dir.exists():
            detector = LandmarkDetector(backend=self.backend)
            key_indices = path_analysis["key_frame_indices"]
            landmark_results = detector.detect_key_frames(frames_dir, key_indices)
            scene_context = landmark_context_for_prompt(
                landmark_results, path_analysis["primitives"], key_indices
            )
            # Load images for vision-language prompt
            from PIL import Image
            for idx in key_indices:
                img_path = frames_dir / f"frame_{idx:04d}_rgb.png"
                if img_path.exists():
                    images_for_prompt.append(Image.open(img_path).convert("RGB"))
                    image_paths_for_prompt.append(str(img_path))
        elif landmark_results is not None:
            scene_context = landmark_context_for_prompt(
                landmark_results, path_analysis["primitives"], path_analysis["key_frame_indices"]
            )

        prompt = INSTRUCTION_PROMPT_TEMPLATE.format(
            motion_sequence=motion_sequence,
            scene_context=scene_context,
        )

        # Generate with retries
        best_instruction = ""
        best_failures = ["no attempts made"]

        for attempt in range(MAX_RETRIES):
            if self.backend == "gemma":
                raw = self._generate_gemma(prompt, images_for_prompt)
            else:
                raw = self._generate_gpt4o(prompt, image_paths_for_prompt)

            # Clean: take only first coherent sentence group
            instruction = self._clean_output(raw)
            ok, failures = quality_check(instruction)

            if ok:
                return {
                    "instruction_text": instruction,
                    "quality_ok": True,
                    "quality_failures": [],
                    "attempts": attempt + 1,
                    "motion_sequence": motion_sequence,
                    "scene_context": scene_context,
                }

            if len(failures) < len(best_failures):
                best_instruction = instruction
                best_failures = failures

        # Return best attempt even if quality check fails
        return {
            "instruction_text": best_instruction,
            "quality_ok": False,
            "quality_failures": best_failures,
            "attempts": MAX_RETRIES,
            "motion_sequence": motion_sequence,
            "scene_context": scene_context,
        }

    @staticmethod
    def _clean_output(raw: str) -> str:
        """Extract clean instruction from potentially noisy VLM output."""
        # Remove common prefixes from models that don't follow instructions cleanly
        for prefix in ["Instruction:", "Navigation instruction:", "Here is", "Here's", "Answer:"]:
            if raw.lower().startswith(prefix.lower()):
                raw = raw[len(prefix):].strip()
        # Take first 4 sentences max
        sentences = re.split(r"(?<=[.!?])\s+", raw.strip())
        clean = " ".join(sentences[:4]).strip()
        # Ensure ends with period
        if clean and clean[-1] not in ".!?":
            clean += "."
        return clean


def generate_text_only(episode: Dict, backend: str = "gemma") -> str:
    """
    Text-only generation (no rendered frames): uses path geometry + motion primitives only.
    Useful as a fast baseline before Gate 1 rendering is ready.
    """
    gen = InstructionGenerator(backend=backend)
    result = gen.generate(episode, frames_dir=None)
    return result["instruction_text"]


if __name__ == "__main__":
    import gzip, sys

    GT_PATH = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz"

    print("=== Gate 4: Instruction Generator — Text-Only Test ===")
    print("(No rendered frames — uses path geometry only)\n")
    print("NOTE: This test generates instructions WITHOUT seeing GT text.")
    print("The model reasons from path geometry (turns, distances) only.\n")

    with gzip.open(GT_PATH, "rt") as f:
        data = json.load(f)

    # Test on 3 episodes — path analysis only, no images
    for ep in data["episodes"][:3]:
        path_result = analyze_path(ep["reference_path"], ep.get("start_rotation"))
        motion_text = primitives_to_text(path_result["primitives"])
        print(f"Episode {ep['episode_id']}:")
        print(f"  Path:           {motion_text}")
        print(f"  GT instruction: {ep['instruction']['instruction_text'].strip()}")
        print(f"  (Would generate with: InstructionGenerator(backend='gemma').generate(ep))")
        print()
