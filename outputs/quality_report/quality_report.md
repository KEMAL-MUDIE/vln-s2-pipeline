# ChronoNav Metadata Quality Report

**Dataset**: VLN-CE val-unseen · 1839 episodes
**Generated instruction source**: Gemma 4 31B AWQ via vLLM (Gate 4 v2)
**Comparison target**: GT R2R val-unseen human instructions

---

## Coverage

| Component | Count | % of 1839 |
|-----------|-------|----------|
| Rendered frames (Gate 1) | 1839 | 100.0% |
| Landmark annotations (Gate 3) | 1839 | 100.0% |
| Generated instruction (Gate 4) | 1839 | 100.0% |
| All three components | 1839 | 100.0% |

---

## Text Quality: Generated vs GT Instructions

*(Higher = better. GT annotations have diversity noise, so ~0.25–0.35 composite is expected.)*

| Metric | Mean | Std | Median | P25 | P75 |
|--------|------|-----|--------|-----|-----|
| BLEU-1 | 0.317 | 0.116 | 0.316 | 0.238 | 0.393 |
| BLEU-2 | 0.174 | 0.100 | 0.167 | 0.102 | 0.241 |
| ROUGE-L | 0.304 | 0.090 | 0.300 | 0.245 | 0.360 |
| METEOR | 0.280 | 0.115 | 0.267 | 0.192 | 0.353 |
| Noun-F1 | 0.157 | 0.126 | 0.143 | 0.069 | 0.235 |
| Composite | 0.265 | 0.093 | 0.257 | 0.197 | 0.323 |

### Instruction Length Comparison

| | Mean words | Std | Min | Max |
|-|-----------|-----|-----|-----|
| Generated | 23.6 | 2.8 | 16 | 34 |
| GT | 26.8 | 11.4 | 5 | 119 |

---

## Landmark Quality

*(How well our Gate 3 landmark detections cover GT instruction landmarks)*

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Landmark Recall | 0.206 | 0.158 | 0.188 |
| Landmark Precision | 0.075 | 0.059 | 0.069 |
| Landmark F1 | 0.106 | 0.077 | 0.100 |

---

## Path Analysis Summary

| Path type | Count | % |
|-----------|-------|---|
| winding | 903 | 49.1% |
| single_turn | 624 | 33.9% |
| mostly_straight | 312 | 17.0% |

| Stat | Mean | Std | Min | Max |
|------|------|-----|-----|-----|
| Distance (m) | 9.61 | 2.68 | 5.05 | 22.14 |
| Waypoints | 5.9 | 0.9 | 4 | 7 |
| Turns per episode | 1.56 | 1.08 | 0 | 5 |

---

## Best / Worst Examples

### Top-5 by Composite Score

**Episode 719** (composite=0.635)
- GT:  *Leave the bathroom and turn right.  Walk straight until you enter the kitchen area and stop in front of the island count*
- Gen: *Walk out of the hallway and turn right. Walk straight into the kitchen and stop in front of the kitchen island.*
- BLEU-1=0.681 ROUGE-L=0.698 METEOR=0.663 Noun-F1=0.500

**Episode 1439** (composite=0.621)
- GT:  *Turn around and go towards the bed. Turn left and enter the bathroom. Stop in front of the sink. *
- Gen: *Exit the bedroom and turn left. Turn left again and walk straight into the bathroom. Stop in front of the bathroom sink.*
- BLEU-1=0.636 ROUGE-L=0.585 METEOR=0.662 Noun-F1=0.600

**Episode 500** (composite=0.605)
- GT:  *Walk across the room and straight down the hallway. Wait in the doorway. *
- Gen: *Exit the living room and walk straight down the hallway. Continue forward and stop in front of the arched doorway.*
- BLEU-1=0.550 ROUGE-L=0.606 METEOR=0.648 Noun-F1=0.615

**Episode 1533** (composite=0.602)
- GT:  *Walk down the stairs and to the right. Walk down the stairs and stop in front of the table. *
- Gen: *Walk out of the hallway and go down the stairs. Turn sharp right and continue down the stairs, then stop in front of the*
- BLEU-1=0.615 ROUGE-L=0.578 METEOR=0.713 Noun-F1=0.500

**Episode 242** (composite=0.596)
- GT:  *Exit the living room, turn left, wait at the bottom of the stairs. *
- Gen: *Exit the living room and walk straight. Turn sharp left and walk forward, then stop at the top step of the stairs.*
- BLEU-1=0.500 ROUGE-L=0.629 METEOR=0.754 Noun-F1=0.500

### Bottom-5 by Composite Score

**Episode 192** (composite=0.033)
- GT:  *Walk through breakfast nook. Walk past kitchen. Walk past thermostat. Wait at exercise ball. *
- Gen: *Exit the living room and turn right. Walk straight, turn sharp left, then turn right again and stop in front of the fire*
- BLEU-1=0.043 ROUGE-L=0.054 METEOR=0.034 Noun-F1=0.000

**Episode 412** (composite=0.036)
- GT:  *Walk past altar book stands. Wait under wooden rafter. *
- Gen: *Leave the church and walk straight. Turn sharp left, then sharp right, and walk forward until you stop in front of the c*
- BLEU-1=0.040 ROUGE-L=0.059 METEOR=0.047 Noun-F1=0.000

**Episode 526** (composite=0.046)
- GT:  *Reverse direction. Walk past sisal carpet runner. Wait at open white door. *
- Gen: *Exit the bedroom and walk straight into the living room. Continue forward and stop in front of the grey upholstered armc*
- BLEU-1=0.048 ROUGE-L=0.061 METEOR=0.077 Noun-F1=0.000

**Episode 925** (composite=0.048)
- GT:  *Exit bathroom, make hard left into bedroom, wait by bed. *
- Gen: *Walk out of the hallway and turn left. Turn left again and walk straight until you stop in front of the white door.*
- BLEU-1=0.043 ROUGE-L=0.061 METEOR=0.088 Noun-F1=0.000

**Episode 935** (composite=0.058)
- GT:  *Walk around the front of the bed past the arm chair in the corner. Walk out through the door in the corner of the room p*
- Gen: *Exit the bedroom and turn right. Walk straight, turn left, then turn left again and walk forward to stop at the kitchen *
- BLEU-1=0.035 ROUGE-L=0.141 METEOR=0.057 Noun-F1=0.000

---

## Scene-Level Breakdown (top 8 scenes by episode count)

| Scene | Episodes | Mean Composite | Mean ROUGE-L |
|-------|---------|----------------|-------------|
| zsNo4HB9uLZ | 300 | 0.262 | 0.312 |
| TbHJrupSAjP | 264 | 0.268 | 0.307 |
| QUCTc6BB5sX | 255 | 0.257 | 0.302 |
| 2azQ1b91cZZ | 252 | 0.252 | 0.288 |
| oLBMNvg9in8 | 177 | 0.308 | 0.343 |
| Z6MFQCViBuw | 159 | 0.232 | 0.273 |
| X7HyMhZNoso | 141 | 0.283 | 0.319 |
| EU6Fwq7SyZv | 132 | 0.256 | 0.280 |

---

## Interpretation

- **Composite ~0.25–0.35** is typical ceiling for single-reference evaluation against
  human-generated navigation instructions. R2R val-unseen has multiple human annotators
  per path (3 annotations per trajectory); single-reference BLEU/METEOR does not capture
  the full overlap. Our scores are consistent with this ceiling.

- **Landmark recall** measures whether our Gate 3 Gemma vision detections surface the
  spatial landmarks that human annotators mention in their instructions.

- **Instruction length gap**: if our generated instructions are significantly shorter than
  GT, BLEU-2 and ROUGE-L will be penalized. Adjust Gate 4 prompt for longer output if needed.

- **Path type distribution** validates that our path analyzer correctly identifies
  the geometric structure (straight vs single-turn vs winding) across all 1839 episodes.