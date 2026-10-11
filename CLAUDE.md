# vln-s2-pipeline

VLN instruction 자동 생성 파이프라인. Kemal(`kemal-mudie`, ChronoNav) 작성.
`InternNav/data/vln/mp3d/r2r/v1/` 의 label 248개를 **실제로 만든 코드**가 여기 있다.

---

## 1. 목적

### 1-1. InternNav relabel 2x2 에서 GT 수준 성능 내기 (주 목표)

```
          평가 GT      평가 relabel
학습 GT     61.7          ?            ← 공개 InternVLA-N1-System2 기준점
학습 relabel  ?           ?            ← 이 칸들을 61.7 에 붙이는 게 목표
```

이 레포 README 의 자체 측정 기준점:

| | val_unseen SR |
|---|---|
| GT Human (R2R) | **63.77%** |
| v24 Gate4-Visual | 40.24% |
| v213 MetaReproducer | ~34% |
| v211 MetaReproducer | 29.3% |
| v6 Gate3-Gemma | **predicted 58–67%** (eval pending) |

InternNav 쪽 실측(`InternNav/.claude/memory/relabel_v218_eval_result.md`)과 숫자 체계가
다르다 (공개 System2 GT = 61.7). **같은 축으로 재측정하기 전까지 두 표를 섞지 말 것.**

### 1-2. 추후: guidedog / 공장 순찰 로봇 label (GT 재현 이후)

`gate8_adapters/` 에 rosbag·video 어댑터가 이미 있다. `gate4_vi_style/` 은 시각장애인용
라인. **GT 재현이 끝나기 전에는 손대지 않는다.**

---

## 2. 구조 — gate 0~8

```
gate0_analysis/      analyze_gt.py              GT 통계 추출 → gt_statistics.json
gate1_renderer/      renderer.py                Habitat-Sim 으로 경로 따라 RGB/depth 렌더
                     make_manifest.py, render_batch.sh, render_train.sh
gate2_path/          path_analyzer.py           3D waypoint → motion primitive (회전/직진/정지)
                     scene_room_mapper.py
gate3_landmarks/     landmark_detector.py       프레임 → VLM → room + landmark
                     run_landmark_batch.py
gate4_instructions/  instruction_generator.py   primitive + landmark → 문장
                     gemma_vllm_backend.py      vLLM async 클라이언트
gate4_vi_style/      vi_instruction_generator.py  시각장애인용 (별도 라인)
gate5_tokenizer/     tokenizer.py               GT vocab 으로 instruction_tokens
gate6_assembler/     assembler.py               GT 구조 그대로 .json.gz
gate7_eval/          run_habitat_eval.sh        Habitat 평가
gate8_adapters/      rosbag_adapter.py, video_adapter.py   실로봇/영상 입력
```

규모: 루트 `.py` 243개 + gate `.py` 18개, 총 **42만 줄**, 22 MB.

### ⚠️ `pipeline.py` 는 미완성이다

```python
elif args.mode == "full":
    print("Not yet implemented. Run text_only mode first.")
```

`--mode full` (= 렌더 + 비전 포함) 이 **구현돼 있지 않다.** 동작하는 건 `--mode text_only`
뿐이다. **실제로 label 을 만든 것은 루트의 독립 스크립트들**이다:

| 파일군 | 개수 | 역할 |
|---|---|---|
| `run_gate4_visual_v*.py` | **202개** | v2~v203. 각각 자기완결, 10~180 KB |
| `run_gate3_gemma_v*.py` | 7개 | v1~v8. **v6 이 최고 성적** |
| `generate_v2*.py` | 6개 | v207/v212/v217/v218/v219 (MetaReproducer 계열) |
| `run_train_*`, `run_valseen_*` | 3개 | train / val_seen split 전용 |
| `metadata_reproducer.py` | 1개 (62 KB) | 규칙 기반 문장 조립 엔진 |

각 버전 파일은 이전 버전을 `import` 하거나 통째로 복사해 만들어졌다. 예:
`run_gate3_gemma_v6.py` 는 `run_gate3_gemma_v5` 에서 `build_context_v5`, `find_triplets`,
`TRIPLET_TEMPS` 를 가져온다. **버전 간 공통 기반 모듈이 없다.**

---

## 3. 버전 계보 — 세 갈래다

### (A) MetaReproducer 계열 — `metadata_reproducer.py`
`v207, v211, v212, v213, v217, v218, v219` (= InternNav 의 `auto_v2xx`)

```python
# metadata_reproducer.py — LLM 호출 0건. grep 으로 확인됨.
def reproduce_instruction(episode_id, reference_path, start_rotation,
                          perframe: dict, landmark: dict, rng) -> str:
```

**순수 규칙 기반 조립기다.** 모델을 전혀 호출하지 않는다. 시각 정보는
`outputs/gate3_perframe/episode_NNNNNN.json` 과 `outputs/gate3_landmarks/` 에서
**미리 만들어 둔 캐시를 읽을 뿐**이다.

→ **v218 을 "Gemma 로 재현"한다는 말은 성립하지 않는다.** v218 실행에는 GPU 도 모델도
필요 없다. 필요한 건 gate3 캐시다. 캐시가 없으면 `perframe={}` 로 조용히 떨어지고
랜드마크 없는 템플릿 문장이 나온다 — 이것이 `train_v218` 의 내용어가 27종뿐인 이유다
(GT 2,228종).

SR 실적: v211 29.3%, v213 ~34%. **계열 자체가 낮다.**

### (B) Gate3-Gemma 계열 — `run_gate3_gemma_v*.py`
`v1~v8`. InternNav 의 `v4`(=gate3_gemma_v4), `v5`, `v6` 이 여기 해당.

key frame(시작+회전+목표)만 VLM 에 보내고, room-transition 스타일 + temperature
다양화(`TRIPLET_TEMPS`)를 쓴다. **v6 이 README 가 지목한 BEST ANNOTATOR.**

```
v5 pct_zero_explicit=45.6%  →  v6 56.3%  (GT=56%)
v6 변경: 90°+ 에피소드의 20% 에서 init_turn 억제 (episode_id % 10 < 2)
         212/1839 만 재생성, 나머지는 v5 체크포인트 재사용
```

### (C) Gate4-Visual 계열 — `run_gate4_visual_v*.py`
`v2~v203`. InternNav 의 `v12`~`v203` 대부분. 모든 프레임을 VLM 에 보내는 방식.
`v24` 가 40.24% SR 로 이 계열 최고. 파일이 180 KB 까지 커진 건 프롬프트·예외처리가
누적된 결과다.

### 세 계열 ↔ InternNav label 이름 대응

| InternNav label | 이 레포 | 계열 | 시각 |
|---|---|---|:--:|
| `v4` | `run_gate3_gemma_v4.py` | B | ✅ |
| `v5` | `run_gate3_gemma_v5.py` | B | ✅ |
| **`v6`** | `run_gate3_gemma_v6.py` | B | ✅ |
| `v24` | `run_gate4_visual_v24.py` | C | ✅ |
| `v100`~`v203` | `run_gate4_visual_vN.py` | C | ✅ |
| `auto_v207`~`auto_v219`, `v218` | `generate_vN_*.py` | A | 캐시 경유 |
| `gate3style_v5` | `run_train_gate3style_v5.py` | B(text-only) | ❌ |

---

## 4. 어떤 버전을 쓸 것인가 (InternNav 투입 기준)

**학습 축은 v218, 평가 축은 v6 — split 마다 답이 다르다.**
README 의 SR 표는 **val_unseen 평가 전용**이라 학습 적합성을 말해주지 않는다.

### InternNav 실측 (학습 결과)

| | train 고유문장 | train 내용어 | loss @ep0.5 | 학습 SR |
|---|---|---|---|---|
| GT | 10,815/10,819 (99.9%) | 2,228 | 0.485 | 61.7 |
| **v218** | **10,430/10,819 (96.4%)** | 27 | **0.524** | — |
| v6 | **1,856/10,819 (17.2%)** | 53 | 0.721 | **8.7** ← 붕괴 |

**v6 는 학습에 쓸 수 없다.** train split 고유 문장이 17.2%, 같은 문장이 최대 510회
반복되고 `"Pass the landmark"` placeholder 가 485건이다. SR 8.7 로 학습이 실패했다.

**v218 은 반대로 고유성이 96.4% 로 GT 에 근접한다.** 문제는 어휘가 27종뿐이라는 것 —
그리고 그 원인이 코드에 정확히 적혀 있다 (§4-1).

### 4-1. v218 train 의 내용어가 27종인 이유

`metadata_reproducer.py:361`

```python
_has_context = bool(perframe) and bool(perframe.get("start"))
```

`perframe` (gate3 캐시) 가 없으면 `_has_context=False` 가 되고, **23개 분기**가 전부
폴백 경로로 간다:

```python
# metadata_reproducer.py:393-395
_PATH_ONLY_ROOMS = ["room", "room", "room", "hallway"]
_PATH_ONLY_LMS   = ["room", "room", "hallway", "corridor", "room"]
_PATH_ONLY_STOP  = ["room", "room", "hallway", "corridor", "room"]
```

`train_v218` 의 상위 내용어가 정확히 `room, hallway, corridor` 다.
**label 품질 문제가 아니라 입력 누락이다.**

| | perframe | 내용어 | 고유문장 |
|---|---|---|---|
| `val_unseen_v218` | ✅ 있었음 | 506 | 96.4% |
| `train_v218` | ❌ 없었음 | **27** | 96.4% |

→ **train 에 gate3 캐시만 공급하면 같은 생성기가 랜드마크 있는 문장을 만든다.**
생성기 코드는 한 줄도 고치지 않는다. 이것이 2x2 의 빈 칸을 메우는 가장 짧은 경로다.

### 4-2. 그래서 목표 label — `v218_reproduce` (2026-10-05 실행됨)

```
v218_reproduce = v218 생성기(복원) + train/val_seen/val_unseen 전 split 의 gate3 캐시
```

v6 은 **평가축 비교군**으로만 남긴다 (val_unseen 품질은 좋음).

### 4-3. 재현 검증 결과 (2026-10-05, 이 환경)

**현재 git 의 `metadata_reproducer.py` 는 v218 이 아니라 v219 다** (독스트링 확인).
v219 는 기존 분기 앞에 `if <조건> and rng.random() < P:` 를 4곳 삽입한 것이고,
그 `rng.random()` 이 난수 스트림을 밀어 뒷문장까지 바꾼다. 복원 = 그 분기를 제거.
보존본은 `metadata_reproducer_v219.py`, 복원본 사본은 `metadata_reproducer_v218.py`.

| 검증 | 결과 | 기준 |
|---|---|---|
| **코드 복원** — 같은 perframe 으로 복원판 vs v219 | **116/1839 = 6.3%** | 배포본 v218↔v219 와 **동일한 116건** ✅ |
| 차이 패턴 | 65 / 51 건 | 배포본 66 / 50 건 ✅ |
| **gate3 재현** — 문장 완전일치 vs 배포 v218 | **83.8%** (1,542/1,839) | temperature 0.1 샘플링 감안 ✅ |
| avg_explicit_turns | **1.97** | 배포 v218 = 1.97 ✅ |
| avg_words | 27.5 | 배포 v218 = 27.5 ✅ |

gate3 재현이 되는지는 **내용어 수 하나로 바로 보인다** (27 이면 perframe 미적용).

| | ep | 내용어 | 고유% | 시각% |
|---|---|---|---|---|
| GT val_seen | 778 | 706 | 100.0 | 22.0 |
| 기존 `val_seen_v218` | 778 | **26** | 99.6 | **0.0** |
| 신규 `val_seen_v218_reproduce` | 778 | **404** | 100.0 | **82.0** |
| GT val_unseen | 1839 | 987 | 100.0 | 21.7 |
| 기존 `val_unseen_v218` | 1839 | 506 | 99.8 | 82.6 |
| 신규 `val_unseen_v218_reproduce` | 1839 | 500 | 99.8 | 82.7 |

**val_unseen 은 기존과 사실상 동일(재현 성공), val_seen 은 템플릿에서 랜드마크 기반으로 전환.**

### 4-4. 이 환경의 실측 처리량

| 단계 | 대상 | 시간 |
|---|---|---|
| gate1 렌더 | val_unseen 1,839 ep / 6,870 프레임 | **112초** (16.4 ep/s, 11 scene) |
| gate1 렌더 | val_seen 778 ep | 443초 (1.8 ep/s, scene 수가 많아 로드가 지배) |
| gate1 렌더 | train 10,819 ep / 40,676 프레임 | ~50분 (61 scene) |
| gate3 | val_unseen 1,839 ep | **~2분** (8 replica × concurrency 24) |
| gate3 | val_seen 778 ep | ~1분 (7 replica) |
| gate3 | train 10,819 ep | ~20분 (7 replica) |
| 문장 생성 | split 당 | **0.2초** (LLM 미사용) |

렌더링은 scene 로드(7.3초/scene)가 지배한다. 프레임 자체는 0.049초.

---

## 5. 이 환경에서 지금 할 수 있는 것

### 있는 것

| | 상태 |
|---|---|
| `habitat_sim` | ✅ **0.3.3 설치됨** |
| mp3d scene mesh | ✅ 90개 (`data-vol1/InternData-N1/scene_data/mp3d`, 21 GB) |
| → val_unseen 필요 11 scene | ✅ 전부 보유 |
| → train 필요 61 scene | ✅ 전부 보유 |
| GT json.gz | ✅ `data-vol1/vln/mp3d/r2r/v1/{split}/` |
| 기존 label 248개 | ✅ 같은 경로 |
| 사전 렌더 프레임 (InternData) | ✅ 329 GB, RGB 640×480 ×5 카메라 + depth |
| GPU | ✅ H200 ×8 (여유 GPU3·GPU7 각 66 GB) |
| Gemma 4 31B AWQ | ✅ HF 에 존재, 20.9 GB (미다운로드) |
| `openai`, `numpy` | ✅ |

### 없는 것

| | 영향 |
|---|---|
| `outputs/` 전체 (`.gitignore` 됨) | **gate3 캐시 0개** → A·B 계열 모두 선행 생성 필요 |
| `outputs/gate3_perframe/`, `gate3_landmarks/` | v218·v6 둘 다 이게 입력 |
| `outputs/gate3_gemma_v5_checkpoint.json` | v6 가 212개 빼고 재사용하려는 것 |
| 원격 vLLM `10.77.32.231:8000` | rc=000 (타임아웃). 로컬 서빙 필요 |
| `vllm` 패키지 | 미설치 |

### 하드코딩된 Kemal 머신 경로 (전부 수정 대상)

```python
HABITAT_BASE = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1"
SCENES_ROOT  = "/mnt/nvme0/vln_habitat/habitat_data/scene_datasets"
OUTPUT_DIR   = "/home/kemal/VLNav/s2_pipeline_new/outputs"
VLLM_BASE_URL = "http://10.77.32.231:8000/v1"
VLLM_MODEL    = "cyankiwi/gemma-4-31B-it-AWQ-4bit"
```

`generate_v218_valunseen.py` 는 `DEPLOY_PATH` 로 **GT 디렉터리에 직접 쓴다.**
이 환경에서는 반드시 출력 경로를 먼저 바꾼다. GT 트리는 읽기 전용.

---

## 6. 실행 — 전체 체인

`pipeline.py --mode full` 이 미구현이므로 gate 를 수동으로 잇는다.

```bash
cd /home/irteam/git/vln-s2-pipeline

# ── 0. 모델 서빙 (터미널 A) ─────────────────────────────────
vllm serve cyankiwi/gemma-4-31B-it-AWQ-4bit \
  --tensor-parallel-size 2 --max-model-len 4096 --port 8000
# gate4_instructions/gemma_vllm_backend.py 의 VLLM_BASE_URL 을 localhost 로

# ── 1. Gate1: 프레임 렌더 (habitat_sim) ─────────────────────
python3 gate1_renderer/run_renderer.py        # SCENES_ROOT / OUTPUT_DIR 수정 선행
#   → outputs/rendered_frames/episode_NNNNNN/{*.png, poses.json}

# ── 2. Gate3: 랜드마크 추출 (VLM) ───────────────────────────
python3 gate3_landmarks/run_landmark_batch.py \
  --frames-dir outputs/rendered_frames
#   → outputs/gate3_landmarks/episode_NNNNNN.json
#   (perframe 은 run_gate3_perframe_v2.py)

# ── 3. 문장 생성 ────────────────────────────────────────────
python3 run_gate3_gemma_v6.py                 # 권장. val_unseen
python3 run_train_gate3style_v5.py            # train  (현재 text-only)
python3 run_valseen_gate3style_v5.py          # val_seen

# 참고: v218 (규칙 기반, 모델 불필요 — gate3 캐시만 있으면 됨)
python3 generate_v218_valunseen.py
```

### InternNav 투입

```bash
cd /home/irteam/git/InternNav
# json.gz 를 data/vln/mp3d/r2r/v1/{split}/{split}_<label>.json.gz 에 둔 뒤
/usr/bin/python3 scripts/dataset_converters/relabel_vlnce/build_relabel_dataset.py --self_check
/usr/bin/python3 scripts/dataset_converters/relabel_vlnce/build_relabel_dataset.py \
  --labels <label> --emit data,yaml,config
```

---

## 7. 설정 — YAML 이 이미 있다 (다만 안 쓰인다)

```
configs/pipeline_config.yaml   렌더러·path·landmark·generator·tokenizer·eval 전 구간
configs/vlm_prompts.yaml       landmark_detection / instruction_generation /
                               instruction_generation_text_only 프롬프트 템플릿
```

**`pipeline.py` 만 `vlm_prompts.yaml` 을 읽는다.** 실제로 label 을 만든
`run_gate3_gemma_v*.py` / `run_gate4_visual_v*.py` / `generate_v2*.py` 는 전부
**프롬프트와 상수를 파일 안에 직접 박아 두었다.** 그래서 202개 버전이 서로 다른 파일이다.

`pipeline_config.yaml` 은 `google/gemma-3-27b-it` 을 적어 두었으나 실제 실행은
`cyankiwi/gemma-4-31B-it-AWQ-4bit` 였다 (`_generation_meta` 와 `gemma_vllm_backend.py`
가 일치). **config 를 신뢰하지 말고 코드를 보라.**

→ YAML 분기를 실제로 쓰려면 `run_gate3_gemma_v6.py` 의 상수를
`configs/` 로 끌어내는 작업이 선행돼야 한다.

---

## 8. 불변식 / 함정

- **GT 트리는 읽기 전용.** `data-vol1/.../val_unseen.json.gz`, `val_unseen_patched.json.gz`
  는 절대 덮어쓰지 않는다. `generate_*.py` 의 `DEPLOY_PATH` 가 바로 거기를 가리킨다
- **`outputs/` 는 gitignore 다.** 체크포인트를 지우면 버전 재현이 영영 불가능해진다.
  v6 는 v5 체크포인트에 의존한다
- **버전 파일을 수정하지 말고 복사해서 새 번호를 붙인다.** 이 레포의 기존 규약이다
- **`metadata_reproducer.py` 는 모델을 호출하지 않는다.** A 계열 결과가 나쁘면
  모델 문제가 아니라 gate3 캐시 또는 규칙 문제다
- **gate3 캐시가 없으면 조용히 템플릿으로 떨어진다.** 에러가 안 난다.
  생성 후 반드시 내용어 수를 센다 (GT train 2,228 / val_unseen 987)
- **label episode 의 키는 GT episode 키의 부분집합이어야 한다.** habitat 이 `VLNEpisode(**episode)` 로
  만들어서 `_generated_instruction` 같은 키가 있으면 평가가 즉시 죽는다 (학습은 통과해서 늦게 발견됨,
  2026-10-10 eval 18건). `annotate.py deploy` 가 검사한다. 생성 기록은 `{split}_meta/ep_*.json` 에 둔다
- **InternNav 평가기는 문장 마지막 글자를 자른다** (`habitat_vln_evaluator.py:614`).
  문장은 `". "` (마침표+공백) 로 끝나야 한다
- **일부 variant 는 `instruction_vocab` 이 빈 dict** 다. `word_list` 유효성으로 판단해
  공식본에서 주입해야 한다 (빌더가 처리 중)

---

## 9. 품질 게이트 (생성 직후 확인)

| 지표 | GT train | GT val_unseen | 불합격 신호 |
|---|---|---|---|
| 내용어 종류 | 2,228 | 987 | < 300 → 템플릿 |
| 고유 문장 비율 | 99.9% | 100% | < 50% → 중복 폭발 |
| 시각 어휘(색·재질) | 22.8% | 21.7% | 0% → 이미지 미사용 |
| 방향 정합성 | 0.80 | 0.84 | < 0.6 → 좌우 뒤집힘 |
| avg_words | 26.8 | 26.8 | — |
| avg_explicit_turns | 0.66 | 0.66 | > 2 → 과다 명시 |

검사 도구: `python tools/check_label.py`(vln-annotator 레포) 또는
`InternNav/scripts/dataset_converters/relabel_vlnce/check_label_quality.py`

---

## 10. vln-annotator 와의 관계

`~/git/vln-annotator` 는 **이 레포의 재구현본**이다. 최초 커밋 메시지:
*"Complete clean implementation of the VLN auto-annotation pipeline"* (ChronoNav, 2026-08-20).

| | vln-s2-pipeline (이 레포) | vln-annotator |
|---|---|---|
| 구조 | gate 0~8 | Phase 1a/1b/1c + Phase 2 |
| 버전 | 202+ 개 파일 | 단일 패키지 (v62/63 시점) |
| 실제 label 생산 | ✅ 248개 전부 | ❌ |
| 설정 | 파일 내 상수 | `AnnotatorConfig` dataclass |
| 렌더러 | ✅ `gate1_renderer/` | ❌ 없음 |
| 줄수 | 42만 | 2,404 |

**label 을 재현·확장하려면 이 레포를 쓴다.** vln-annotator 는 코드가 깨끗하지만
v218 을 만든 코드가 아니고 렌더러도 없다.

---

## 11. 미해결 질문

1. **v6 의 실제 SR** — README 는 `predicted 58–67%, eval pending`. 실측값이 없다
2. **train split 을 왜 text-only 로 두었나** — 렌더 프레임이 val_unseen 1,839개분만
   있었을 가능성. `render_train.sh` 가 존재하므로 의도는 있었던 것으로 보인다
3. **v6 val_unseen 방향 정합성 0.47** — 무작위 수준인데 SR 예측은 높다. 둘 중 하나가 틀렸다
4. **README SR 표(GT 63.77) vs InternNav 실측(GT 61.7)** — 평가 설정이 다르다.
   어느 쪽으로 통일할지 정해야 한다
5. **ChronoNav v264~v287 (SR 66.2%)** — 이건 생성이 아니라 기존 GT 문장의
   sibling 치환이다. 2x2 relabel 실험의 대상인지 확인 필요
