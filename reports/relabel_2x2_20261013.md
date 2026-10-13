# Relabel 2x2 실험 보고서 (2026-10-13 14:30 KST 기준)

R2R 사람 지시문(GT) 대신 **자동 생성한 지시문(label)** 으로 InternVLA-N1 을 학습·평가했을 때
성능이 얼마나 유지되는지 측정한 결과다. 시각은 모두 KST.

---

## 0. 용어 (먼저 읽기)

| 용어 | 뜻 |
|---|---|
| **GT** | R2R 사람이 쓴 원래 지시문. 경로 하나에 3개씩 있다 |
| **label** | 같은 경로에 대해 자동 생성기로 새로 쓴 지시문 세트 (train / val_seen / val_unseen 3개 split) |
| **2x2** | {학습 문장: GT \| label} x {평가 문장: GT \| label} 4칸. 경로·장면·평가 에피소드는 같고 **문장만** 다르다 |
| **s1 (stage1)** | System2 단독 학습. Qwen2.5-VL-7B 를 시작점으로 VLM 전체를 학습해 "이미지 속 다음 목표 픽셀"을 예측 |
| **s2 (stage2)** | s1 결과 위에 System1(로컬 정책 `nextdit_async`)을 붙여 학습. VLM 은 고정 |
| **half** | s1 을 1 epoch 만 학습한 설정 (`relabel_half/`). 이전 설정(`relabel/`)은 2 epoch |
| **SR** | 성공률. 멈춘 위치가 목표 3 m 이내인 비율. **주 지표** |
| **SPL** | 성공에 경로 효율(최단거리 / 실제 이동거리)을 곱한 평균 |
| **OS** | 경로 중 한 번이라도 목표 3 m 안에 들어간 비율. OS − SR 이 크면 "근처까지 가서 엉뚱한 데 멈춤" |
| **NE** | 멈춘 위치와 목표 사이 거리 (m). 낮을수록 좋다 |

2x2 각 칸이 답하는 질문:

| 칸 | 질문 |
|---|---|
| label 로 학습 → **GT 로 평가** | 이 label 이 **학습 데이터로서** 얼마나 좋은가 (주 비교 지표) |
| **GT 로 학습** → label 로 평가 | 이 label 문장이 **경로 정보를 얼마나 담고 있는가** |
| label 로 학습 → label 로 평가 | 같은 분포 안의 성능. 템플릿 label 은 이 값이 부풀려진다 |

---

## 1. 결론

1. **가장 좋은 label 도 GT 보다 SR 8.9%p 낮다.** s1 기준 GT 학습 42.5 vs 최고 label(v278fix_fewshot-3) 33.6.
2. **label 문장은 경로 정보를 12~15%p 잃는다.** GT 로 학습한 모델을 label 문장으로 평가하면 42.5 → 27.8~30.9.
3. **v278fix 위에 얹은 ablation 은 기준선과 거의 같다** (29.7~33.6, 기준선 31.9). 차이가 1회 학습의 오차 범위(~±1~2%p 추정) 안이라
   "확실히 좋아진 것"은 아직 없다. fewshot-3 이 +1.7 이지만 fewshot-5 는 +0.0 이라 같은 방향으로 이어지지 않았다.
4. **템플릿 label(v218 계열)은 자기 문장에 과적합한다.** 자기 label 평가 40.5 / GT 평가 27.3. VLM label(v278 계열)은 두 평가가 ±2%p 로 일치.
5. **s2 는 label 간 차이를 구분하지 못한다** (모든 label 23~26%). 2026-10-13 부터 2x2 는 s1 만 진행.

---

## 2. 기준선

### 2-1. 공개 checkpoint (full 학습, 같은 평가 하네스)

| 모델 | GT 문장 평가 SR / SPL / OS / NE | v218 계열 label 문장 평가 SR |
|---|---|---|
| InternVLA-N1-System2 (공개 s1) | **61.7** / 55.7 / 68.2 / 4.18 | v218 39.4 · v218_reproduce 39.0 · v219_reproduce 39.5 |
| InternVLA-N1-DualVLN (공개 s2) | **63.7** / 58.1 / 69.4 / 4.12 | v218 38.6 |

→ 공개 모델조차 label 문장으로 평가하면 22%p 떨어진다. label 문장의 정보 손실 상한선 참고값.

### 2-2. 이 실험의 기준선 (mini 데이터, GT 로 학습)

| 설정 | s1 SR / SPL / OS / NE | s2 SR / SPL / OS / NE |
|---|---|---|
| **relabel_half (s1 1 epoch)** | **42.5** / 37.5 / 52.5 / 6.79 | **33.7** / 29.5 / 42.2 / 7.06 |
| relabel (s1 2 epoch) | 39.4 / 35.0 / 49.3 / 7.10 | 28.5 / 24.6 / 37.5 / 7.45 |

→ mini 데이터에서는 1 epoch 가 2 epoch 보다 3.1%p 높다 (2 epoch 는 과적합). 이후 비교는 **relabel_half 의 42.5 / 33.7** 을 기준으로 한다.

---

## 3. 실험 설정

### 3-1. 데이터·평가

| 항목 | 값 | 왜 |
|---|---|---|
| 학습 데이터 | InternData-N1-v0.5-mini `vln_ce` R2R train, **61 scene / 10,684 episode** | GT 학습과 같은 이미지·궤적. label 은 문장만 교체 (`build_relabel_dataset.py`, 이미지는 GT symlink) |
| 평가 split | R2R **val_unseen 1,839 episode** (11 scene) | 표준 평가. 모든 셀 동일 |
| 평가 환경 | habitat, `vln_r2r_mini_ld30.yaml` — RGB 640x480, HFOV 79°, tilt 15°x2, 성공 반경 3.0 m | 셀끼리 `data_path`(문장) 한 줄만 다름 |
| 하드웨어 | H200 x8 노드 (node1~4) | |

### 3-2. 학습 인자 (relabel_half)

| 인자 | s1 | s2 | 뜻 / 왜 이 값인가 |
|---|---|---|---|
| `model_name_or_path` | Qwen2.5-VL-7B-Instruct | **같은 label 의 half s1** | s2 는 s1 위에 쌓는다. s1 이 없으면 config 가 일부러 실패 (공개 ckpt 로 조용히 대체 방지) |
| `vln_dataset_use` | r2r_125cm_0_30, 125cm_0_45, 60cm_15_15, 60cm_30_30 | r2r_125cm_0_30%30, 60cm_15_15%30 | 카메라 높이·pitch 조합별 R2R 렌더. s2 는 30% 샘플 |
| `num_train_epochs` | **1.0** | 3.0 | s1 2 epoch 는 과적합 (2-2 표) |
| `learning_rate` | 2e-5 (cosine) | 1e-4 (cosine_with_min_lr) | 공개 레시피 |
| 실효 batch | 4 x grad_accum 4 x 8 GPU = 128 | 16 x 1 x 8 = 128 | 공개 레시피 |
| 학습 대상 (`tune_mm_*`) | VLM 전체 (llm·vision·mlp) | VLM 고정, System1 만 | s1 은 언어→목표 학습, s2 는 실행 정책 학습 |
| `system1` | none | nextdit_async | |
| `num_history` / `max_pixels` | 8 / 313,600 | 8 / 313,600 | |
| 소요 시간 | 학습 ~16.7 h + 평가 ~1.6 h | 학습 ~2.3 h + 평가 ~4 h | |

모든 label 셀은 label 이름 외에 config 가 **바이트 단위로 동일**하고, s1 학습 인자 96개가 완료된 run 의 wandb 기록과 일치함을 확인했다 (2026-10-11).

### 3-3. label 목록

| label | 생성 방식 | 바꾼 것 |
|---|---|---|
| v6 | 초기 VLM 생성기 (gate3 Gemma v6) | — (하한선) |
| v218 / v218_reproduce | MetaReproducer: **LLM 없이** 경로 기하 + gate3 랜드마크로 템플릿 문장 | v218_reproduce 는 재실행본 |
| v218fix | v218 + 렌더 수정 (HFOV 79, 검은 프레임 제거) | |
| **v278fix** | Kemal v278 **Gemma-4-31B VLM 생성기** + 수정 3가지 (waypoint 기준 key frame, HFOV 79, 검은 프레임 제거), seed 0 | **ablation 기준선** |
| v278fix_stop-gtdist | | 끝맺음 동사를 GT 비율로 뽑음 (stop 55 / wait 32 / 없음 10 / stand 3 %). 기준선은 "stop" 강제 |
| v278fix_fewshot-3 / -5 | | 다른 경로의 GT train 문장 3 / 5개를 문체 예시로 프롬프트에 넣음 (같은 경로 GT 는 제외) |
| v278fix_div-on | | 같은 경로의 앞서 쓴 문장을 보여주고 "다르게 쓰라" |
| v278fix_len-gtdist | | 길이 목표를 GT 길이 분포에서 뽑음 (기준선 15~40 단어) |
| v278fix_len-free | | 길이 제한 없음 |
| v278fix_g3temp-070 | | 장면 묘사(gate3) temperature 0.4 → 0.7 |
| v278fix_g4temp-100 | | 문장 생성(gate4) temperature 0.35 → 1.0 |

생성 코드: `vln-s2-pipeline/annotate.py` + `configs/labels/<label>.yaml` (`base: v278fix` 상속). GT 문장은 어떤 label 에도 섞이지 않았다 (episode 별 모든 문자열 필드 대조, 0건).

---

## 4. 결과

### 4-1. s1 2x2 — relabel_half (주 결과)

SR / SPL / OS (%). 기준: GT 학습 → GT 평가 **42.5**.

| 학습 label | label 학습 → **GT 평가** | 기준선 대비 (v278fix 31.9) | label 학습 → 자기 label 평가 | GT 학습 → 이 label 평가 |
|---|---|---|---|---|
| **GT** | **42.5** / 37.5 / 52.5 | — | — | 42.5 |
| v218fix | 27.3 / 23.9 / 37.0 | −4.6 | 40.5 / 36.6 / 57.6 | 30.9 / 26.7 / 47.6 |
| **v278fix** (기준선) | 31.9 / 28.5 / 40.0 | 0 | 31.9 / 28.6 / 41.9 | 30.0 / 24.9 / 43.4 |
| stop-gtdist | 32.4 / 29.2 / 40.9 | +0.5 | 32.6 / 28.9 / 42.3 | 30.1 / 25.2 / 44.6 |
| fewshot-3 | **33.6** / 30.4 / 43.3 | **+1.7** | 32.7 / 28.9 / 43.0 | 29.3 / 24.7 / 43.4 |
| fewshot-5 | 31.9 / 28.5 / 42.1 | 0.0 | 31.2 / 27.6 / 42.3 | 29.9 / 24.8 / 44.5 |
| div-on | 29.7 / 26.6 / 39.9 | −2.2 | 30.9 / 26.8 / 41.3 | 27.8 / 23.6 / 42.0 |
| len-gtdist | 29.7 / 26.1 / 40.6 | −2.2 | 27.9 / 24.6 / 39.7 | 28.6 / 24.5 / 42.6 |
| g3temp-070 | 30.1 / 27.1 / 38.3 | −1.8 | 진행 중 (node2) | 진행 중 (node2) |
| len-free | 진행 중 (node3, ~10/14 09시) | | | |
| g4temp-100 | 진행 중 (node4, ~10/14 09시) | | | |

### 4-2. s2 2x2 — relabel_half (2026-10-13 부터 생략)

SR (%). 기준: GT 학습 → GT 평가 **33.7**.

| 학습 label | → GT 평가 | → 자기 label 평가 | GT 학습 → 이 label 평가 |
|---|---|---|---|
| GT | **33.7** | — | 33.7 |
| v218fix | 19.6 | 31.6 | 24.5 |
| v278fix | 24.9 | 25.2 | 25.4 |
| stop-gtdist | 25.8 | 23.9 | 25.5 |
| fewshot-3 | 25.2 | 25.2 | 24.2 |
| fewshot-5 | 23.0 | 25.9 | 25.6 |
| div-on | 25.5 | 23.0 | 23.7 |
| len-gtdist | 24.1 | 25.8 | 24.4 |
| g3temp-070 | 23.5 | (생략) | (생략) |

→ v278 계열이 전부 23~26 에 몰려 ablation 을 구분하지 못한다. 셀당 ~10 h 가 들어 s2 는 중단.

### 4-3. 이전 2x2 — relabel (s1 2 epoch)

SR (%).

| 학습 label | s1 → GT | s1 → 자기 label | s2 → GT | s2 → 자기 label |
|---|---|---|---|---|
| GT | **39.4** | (GT 학습 → v218 31.6 · v218_repro 32.0 · v6 24.7) | **28.5** | (→ v218 23.4 · v218_repro 23.2 · v6 19.2) |
| v218 | 21.4 | 34.4 | 16.4 | 27.2 |
| v218_reproduce | 28.5 | 42.4 | 20.0 | 31.6 |
| v6 | 8.7 | 11.6 | 8.0 | 9.1 |

### 4-4. label 문장 지표 (val_unseen, 학습 전 품질 점검용)

| label | 단어 수 (평균±표준편차) | 방향 정합성 | "Go"로 시작 % | 같은 경로 문장 간 F1 | GT 와 단어 F1 | METEOR | CIDEr |
|---|---|---|---|---|---|---|---|
| **GT** | 26.8 ± 11.4 | 0.856 | 18.5 | 0.393 | 1.000 | 0.355* | **0.286*** |
| v218fix | 27.5 ± 8.8 | 0.925 | 4.0 | 0.667 | 0.360 | 0.350 | 0.102 |
| v278fix | 27.1 ± 4.1 | 0.877 | 96.1 | 0.829 | 0.372 | 0.370 | 0.136 |
| stop-gtdist | 27.2 ± 4.1 | 0.877 | 94.7 | 0.796 | 0.368 | 0.360 | 0.138 |
| fewshot-3 | 27.7 ± 4.2 | 0.892 | 46.4 | 0.786 | 0.384 | 0.379 | 0.137 |
| fewshot-5 | 27.8 ± 4.2 | 0.884 | 37.3 | 0.789 | 0.387 | 0.381 | 0.139 |
| div-on | 26.8 ± 3.9 | 0.862 | 35.0 | 0.653 | 0.360 | 0.352 | 0.124 |
| len-gtdist | 23.6 ± 6.7 | 0.890 | 96.2 | 0.754 | 0.363 | 0.348 | 0.131 |

\* GT 행은 같은 경로의 다른 GT 2개를 정답으로 둔 값 (사람끼리의 일치도).

지표 뜻:
- **방향 정합성**: 문장의 마지막 "turn left/right" 가 경로 기하의 마지막 회전과 같은 비율.
- **같은 경로 문장 간 F1**: 한 경로에 달린 문장 3개끼리의 단어 F1. 높을수록 세 문장이 비슷하다 (다양성 부족).
- **GT 와 단어 F1**: 같은 에피소드 GT 1개와의 단어 F1.
- **METEOR / CIDEr**: GT 3개를 정답으로 한 문장 유사도. CIDEr 는 흔한 단어를 낮게, 그 경로에만 나오는 단어(랜드마크)를 높게 친다.
- 계산: 구조 지표 `vln-s2-pipeline/annotate.py:metrics()`, METEOR/CIDEr `pycocoevalcap` + `nltk`.

---

## 5. 해석

1. **방향 정보는 이미 GT 이상이다.** v278 계열 방향 정합성 0.86~0.89 ≥ GT 0.856. 그런데도 SR 이 9%p 낮다.
   남은 차이는 (a) **문체 획일성** — 96% 가 "Go" 로 시작, 길이 표준편차 4 (GT 11), 같은 경로 세 문장이 거의 같음(F1 0.83 vs 0.39) —
   과 (b) **랜드마크 구체성** — CIDEr 0.13 으로 사람(0.29)의 절반 — 으로 보인다.
2. **프롬프트 ablation 은 (a)는 일부 고쳤지만 SR 로 이어지지 않았다.** fewshot·div-on 은 "Go" 시작과 문장 유사도를 GT 쪽으로 옮겼으나
   SR 은 fewshot-3 만 +1.7, fewshot-5 0.0, div-on −2.2. **문장 지표가 좋아져도 SR 이 따라오지 않는다** — fewshot-5 는 문장 지표 1위인데 SR 은 기준선과 같다.
3. **(b) 랜드마크 구체성은 어떤 프롬프트 옵션으로도 바뀌지 않았다** (CIDEr 0.12~0.14 고정). 원인은 gate3(장면 묘사) 쪽이므로 프롬프트가 아니라 gate3 를 바꿔야 한다.
4. **템플릿 label 은 학습 데이터로 부적합하다.** v218fix 는 자기 문장 평가 40.5 인데 GT 평가 27.3 — 템플릿 패턴을 외워 사람 문장에 일반화하지 못한다.
5. **v6 는 실패 사례다** (SR 8.7). 고유 문장 82%, 방향 정합성 0.48(무작위 수준). label 품질 게이트 없이 학습하면 안 된다는 근거.

---

## 6. 한계

- **seed 1회.** 모든 셀이 한 번씩만 학습했다. ±1~2%p 차이는 판단 불가. fewshot-3 의 +1.7 도 재현 확인 전이다.
- **mini 데이터.** 61 scene / 10,684 episode. full 데이터(공개 System2 61.7) 와 절대값이 다르다.
- **s1 half(1 epoch) 기준.** 2 epoch 나 full 데이터에서 순위가 유지되는지는 미확인.
- 진행 중: g3temp-070 의 label 평가 2셀 (node2), len-free · g4temp-100 의 s1 2x2 (node3·node4, ~2026-10-14 09시 KST).

## 7. 다음 단계 제안

1. **seed 반복**: v278fix 와 fewshot-3 를 학습 seed 만 바꿔 1회씩 더 → 오차 범위를 먼저 확정.
2. **gate3 개선**: 랜드마크 구체성(CIDEr) 이 유일하게 안 움직인 축. 장면 묘사 프롬프트·해상도·key frame 수를 바꾼 label.
3. 조합 label (fewshot-3 + stop-gtdist) 은 1 의 오차 범위 확인 후 판단.

---

## 8. 재현 위치

| 항목 | 위치 |
|---|---|
| label 파일 | `/home/irteam/data-vol2/relabel_half/labels/{train,val_seen,val_unseen}/{split}_<label>.json.gz` |
| label 생성 | `vln-s2-pipeline/annotate.py <label>`, 설정 `configs/labels/<label>.yaml` |
| 학습 데이터 빌드 | `InternNav/scripts/dataset_converters/relabel_vlnce/build_relabel_dataset.py --labels <L> --emit data,yaml` |
| 셀 config | `InternNav/scripts/train_eval/qwenvl_train/relabel_half/dualvln_mini_half_s{1,2}.train.<L>_eval.<E>.py` |
| 실행 | `python3 scripts/train_eval/qwenvl_train/runner.py --machine h200 --config <셀 config>` (평가만: `--no-train --model-path <ckpt>`) |
| 결과 | `/home/irteam/data-vol2/checkpoints/relabel{,_half}/<run>/logs/<cell>/result_h200.json` |
| 노드 큐 | `/home/irteam/data-vol2/queues/node{1..4}.txt` |
