#!/usr/bin/env bash
# Gate 7: Habitat Eval on Generated Dataset
# Measures SR/SPL/NE for model navigating with auto-generated instructions.
# Compare against GT baseline: SR=63.89%, SPL=58.55%, NE=4.027m (1839 eps)
#
# Usage: bash run_habitat_eval.sh [generated_dataset.json.gz]
set -euo pipefail

GENERATED="${1:-/home/kemal/VLNav/s2_pipeline_new/outputs/datasets/val_unseen_generated_gemma.json.gz}"

if [ ! -f "$GENERATED" ]; then
    echo "ERROR: Generated dataset not found: $GENERATED"
    echo "Run pipeline.py first: python3 pipeline.py --mode text_only --backend gemma"
    exit 1
fi

INTERNAV=/home/kemal/VLNav/VLNav/workspaces/model/InternNav
CHECKPOINTS=/home/kemal/VLNav/VLNav/checkpoints
HABITAT_DATA=/mnt/nvme0/vln_habitat/habitat_data
LOGS=/home/kemal/VLNav/s2_pipeline_new/logs
mkdir -p "$LOGS"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOGS/gate7_habitat_eval_${TIMESTAMP}.log"
RESULT_DIR="$LOGS/gate7_results_${TIMESTAMP}"

# Write a temp eval config that points to the generated dataset
TEMP_CONFIG=$(mktemp /tmp/gate7_eval_XXXXXX.py)
cat > "$TEMP_CONFIG" << PYEOF
# Gate 7: Habitat eval config for auto-generated instructions
# Generated dataset: $GENERATED
try:
    import diffusers.models.attention as _attn_mod
    import torch.nn as _nn
    class _PatchedLuminaFFN(_attn_mod.LuminaFeedForward.__bases__[0]):
        def __init__(self, dim, inner_dim, multiple_of=256, ffn_dim_multiplier=None):
            super().__init__()
            if ffn_dim_multiplier is not None:
                inner_dim = int(ffn_dim_multiplier * inner_dim)
            inner_dim = multiple_of * ((inner_dim + multiple_of - 1) // multiple_of)
            self.linear_1 = _nn.Linear(dim, inner_dim, bias=False)
            self.linear_2 = _nn.Linear(inner_dim, dim, bias=False)
            self.linear_3 = _nn.Linear(dim, inner_dim, bias=False)
            try:
                from diffusers.models.activations import FP32SiLU
                self.silu = FP32SiLU()
            except ImportError:
                self.silu = _nn.SiLU()
        def forward(self, hidden_states):
            return self.linear_2(self.silu(self.linear_1(hidden_states)) * self.linear_3(hidden_states))
    _attn_mod.LuminaFeedForward = _PatchedLuminaFFN
except Exception as _e:
    print(f"LuminaFFN patch skipped: {_e}")

from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='internvla_n1',
        model_settings={
            'mode': 'dual_system',
            'model_path': '/checkpoints/InternVLA-N1-DualVLN',
            'num_history': 8, 'resize_w': 384, 'resize_h': 384,
            'max_new_tokens': 1024, 'vis_debug': False, 'attn_implementation': 'sdpa',
        },
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={'config_path': 'scripts/eval/configs/vln_r2r_unseen_generated.yaml'},
    ),
    eval_type='habitat_vln',
    eval_settings={
        'output_path': '$RESULT_DIR',
        'save_video': False, 'epoch': 0, 'max_steps_per_episode': 500,
        'port': '2340', 'dist_url': 'env://', 'use_wandb': False,
    },
)
PYEOF

# Write habitat env config pointing to generated dataset
HABITAT_CFG_DIR="$INTERNAV/scripts/eval/configs"
GENERATED_YAML="$HABITAT_CFG_DIR/vln_r2r_unseen_generated.yaml"

# Copy the standard unseen config and patch the dataset path
if [ -f "$HABITAT_CFG_DIR/vln_r2r_unseen.yaml" ]; then
    sed "s|val_unseen_patched.json.gz|$(basename $GENERATED)|g" \
        "$HABITAT_CFG_DIR/vln_r2r_unseen.yaml" > "$GENERATED_YAML"
    echo "Created habitat config: $GENERATED_YAML"
fi

echo "=== Gate 7: Habitat Eval on Auto-Generated Instructions ==="
echo "  Generated dataset: $GENERATED"
echo "  GT baseline: SR=63.89%  SPL=58.55%  NE=4.027m  (1839 eps)"
echo "  Success criterion: SR within ±3pp of GT (≥60.9%)"
echo "  Log: $LOGFILE"
echo "=========================================================="

docker run --rm \
  --name vlnav_gate7_eval \
  --gpus '"device=1"' \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display,video \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e HABITAT_SIM_EGL_DEVICE_ID=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --network host \
  --shm-size 32g \
  --ipc host \
  -v "$INTERNAV":/workspace/InternNav:rw \
  -v "$CHECKPOINTS":/checkpoints:ro \
  -v "$HABITAT_DATA":/workspace/InternNav/data:rw \
  -v "$HABITAT_DATA/scene_datasets":/workspace/InternNav/data/InternData-N1/scene_data/mp3d_ce:ro \
  -v "$LOGS":/workspace/InternNav/logs:rw \
  -v "$(dirname $GENERATED)":/workspace/generated_data:ro \
  -w /workspace/InternNav \
  vlnav/habitat-eval:rebuilt \
  bash -c "
set -e
cd /workspace/InternNav
export PATH=/opt/venv/bin:\$PATH
pip install -e '.[habitat]' --no-build-isolation --no-deps -q 2>&1 | tail -2
ln -sfn /checkpoints /workspace/InternNav/checkpoints

# Copy generated dataset to expected location
cp /workspace/generated_data/$(basename $GENERATED) \
   /workspace/InternNav/data/datasets/vln/mp3d/r2r/v1/val_unseen/$(basename $GENERATED)

echo '=== Starting Gate 7 Habitat eval (auto-generated instructions) ==='
torchrun --nproc_per_node=1 --master_port=29540 \
    scripts/eval/eval.py --config $TEMP_CONFIG
" 2>&1 | tee "$LOGFILE"

# Compare results
echo ""
echo "=== Gate 7 Comparison ==="
if [ -f "$RESULT_DIR/result.json" ]; then
    python3 - "$RESULT_DIR/result.json" << 'PYEOF'
import json, sys
with open(sys.argv[1]) as f: r = json.load(f)
split = list(r.keys())[0]
res = r[split]
gt_sr, gt_spl, gt_ne = 0.6389, 0.5855, 4.027
delta_sr = (res['SR'] - gt_sr) * 100
delta_spl = (res['SPL'] - gt_spl) * 100
delta_ne = res['NE'] - gt_ne
print(f"Generated | SR={res['SR']*100:.2f}%  SPL={res['SPL']*100:.2f}%  NE={res['NE']:.3f}m  ({res['Count']} eps)")
print(f"GT        | SR=63.89%  SPL=58.55%  NE=4.027m  (1839 eps)")
print(f"Delta     | SR={delta_sr:+.2f}pp  SPL={delta_spl:+.2f}pp  NE={delta_ne:+.3f}m")
passed = abs(delta_sr) <= 3.0
print(f"Gate 7    | {'PASS ✓' if passed else 'FAIL ✗'} (criterion: SR within ±3pp of GT)")
PYEOF
else
    echo "Result file not found: $RESULT_DIR/result.json"
fi

rm -f "$TEMP_CONFIG"
echo "Done. Log: $LOGFILE"
