#!/usr/bin/env python3
"""
Gate 5: VLN Vocabulary Tokenizer
Converts generated instruction text to GT-compatible instruction_tokens array.
Uses the same vocabulary (word2idx_dict) as the GT VLN-CE dataset.
"""
import gzip
import json
import re
from pathlib import Path
from typing import List, Optional

VOCAB_SOURCE = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz"
PAD_LENGTH = 200
PAD_INDEX = 0
UNK_INDEX = 1
EOS_INDEX = 3   # </s> in GT vocab


class VLNTokenizer:
    def __init__(self, vocab_source: str = VOCAB_SOURCE):
        self.word2idx = {}
        self.idx2word = {}
        self.pad_length = PAD_LENGTH
        self._load_vocab(vocab_source)

    def _load_vocab(self, vocab_source: str):
        with gzip.open(vocab_source, "rt") as f:
            data = json.load(f)
        vocab = data.get("instruction_vocab", {})
        self.word2idx = vocab.get("word2idx_dict", {})
        word_list = vocab.get("word_list", [])
        self.idx2word = {i: w for i, w in enumerate(word_list)}
        self.num_vocab = vocab.get("num_vocab", len(word_list))
        self.unk_index = vocab.get("UNK_INDEX", UNK_INDEX)
        self.pad_index = vocab.get("PAD_INDEX", PAD_INDEX)

    @staticmethod
    def _tokenize_text(text: str) -> List[str]:
        """Basic tokenization matching GT dataset preprocessing."""
        text = text.lower().strip()
        # Split on whitespace and punctuation (keep punctuation as tokens)
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return tokens

    def encode(self, text: str, pad: bool = True) -> List[int]:
        """
        Encode instruction text to token IDs.
        Matches GT format: word tokens + EOS + zero padding to PAD_LENGTH.
        """
        tokens = self._tokenize_text(text)
        ids = [self.word2idx.get(tok, self.unk_index) for tok in tokens]
        # Add end-of-sentence
        ids.append(EOS_INDEX)
        if pad:
            # Truncate if over limit, then pad
            ids = ids[:self.pad_length]
            ids += [self.pad_index] * (self.pad_length - len(ids))
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text (for verification)."""
        words = []
        for idx in ids:
            if idx == self.pad_index:
                break
            if idx == EOS_INDEX:
                break
            words.append(self.idx2word.get(idx, "<unk>"))
        return " ".join(words)

    def coverage(self, text: str) -> float:
        """Fraction of tokens in text that are in vocabulary (not UNK)."""
        tokens = self._tokenize_text(text)
        if not tokens:
            return 0.0
        known = sum(1 for t in tokens if t in self.word2idx)
        return known / len(tokens)


def verify_roundtrip(tokenizer: VLNTokenizer, gt_path: str = VOCAB_SOURCE):
    """Verify that GT instructions round-trip correctly through the tokenizer."""
    with gzip.open(gt_path, "rt") as f:
        data = json.load(f)
    episodes = data["episodes"][:20]
    errors = 0
    for ep in episodes:
        text = ep["instruction"]["instruction_text"]
        gt_tokens = ep["instruction"]["instruction_tokens"]
        our_tokens = tokenizer.encode(text)
        if our_tokens != gt_tokens:
            errors += 1
            if errors <= 2:
                print(f"  Mismatch ep {ep['episode_id']}: GT={gt_tokens[:10]} OURS={our_tokens[:10]}")
    print(f"Round-trip: {len(episodes)-errors}/{len(episodes)} match exactly")


if __name__ == "__main__":
    print("Loading VLN tokenizer...")
    tok = VLNTokenizer()
    print(f"Vocabulary size: {tok.num_vocab}")

    # Test on a generated instruction
    test_text = "Exit the bedroom and turn left. Walk past the gray couch and stop near the rug."
    tokens = tok.encode(test_text)
    decoded = tok.decode(tokens)
    coverage = tok.coverage(test_text)
    print(f"\nTest instruction: {test_text}")
    print(f"Tokens (first 20): {tokens[:20]}")
    print(f"Decoded: {decoded}")
    print(f"Vocabulary coverage: {coverage:.1%}")

    print("\nVerifying round-trip on GT episodes:")
    verify_roundtrip(tok)
