````md
# Quick Start

This project uses a 3-stage workflow:

- `nq/`: VQ tokenizer
- `dt/`: Dynamics Transformer backbone
- `ft/`: behavior decoding downstream task

---

# 1. VQ Quick Start

## Relevant files

```text
conf/nq/
  data/
  loss/
  model/
  optim/
  tokenize/
  trainer/
  vq_train_Tseng.yaml
  vq_test_Tseng.yaml

dataset/
  nq_dataset.py

model/neural_quantizer/
  VQ_quantizer.py
  nq_layers.py
  nq_utility.py

task/
  nq_train_Tseng.py
  nq_test_Tseng.py
  nq_vq_configs_train_Tseng.py
  nq_vq_configs_test_Tseng.py
````

## 1.1 Train VQ

Edit the config groups under `conf/nq/` as needed, then run:

```bash
python -m task.nq_train_Tseng
```

Example with overrides:

```bash
python -m task.nq_train_Tseng data=Tseng_trial_small trainer.epochs=100 optim.lr=5e-4
```

## 1.2 Inspect the merged Hydra config

```bash
python -m task.nq_train_Tseng --cfg job --resolve
```

Use this to verify:

* the correct YAMLs are loaded
* CLI overrides are applied
* output paths are correct

## 1.3 Evaluate / reconstruct with a trained VQ

Set the checkpoint path in `conf/nq/vq_test_Tseng.yaml`, then run:

```bash
python -m task.nq_test_Tseng
```

Or override from CLI:

```bash
python -m task.nq_test_Tseng paths.ckpt_path=/path/to/checkpoint.pth
```

## 1.4 Export tokens for downstream AR training

Use the tokenize-related settings in:

```text
conf/nq/tokenize/tokenize.yaml
```

Then run:

```bash
python -m task.nq_test_Tseng tokenize=tokenize
```

## 1.5 Important note for token export

Disable Gumbel during inference / tokenization:

```yaml
model:
  use_gumbel: false
  use_gumbel_hard: false
```

Do this in the **test / tokenize config**, not in the training config.

---

# 2. Dynamics Transformer (DT)

## Relevant files

```text
conf/dt/
  data/Tseng_train.yaml
  model/model.yaml
  nq/nq.yaml
  optim/adamw.yaml
  trainer/trainer.yaml
  train_dt_Tseng.yaml

dataset/
  dt_dataset.py

model/dynamics_transformer/
  dt_layers.py
  dt_utility.py
  dual_axis_transformer.py

task/
  DT_configs.py
  dt_train_Tseng.py
```

## 2.1 Main entry

```bash
task/dt_train_Tseng.py
```

## 2.2 Required defaults in `conf/dt/train_dt_Tseng.yaml`

```yaml
defaults:
  - dt_train
  - data: Tseng_train
  - model: model
  - optim: adamw
  - nq@vq: nq
  - trainer@train: trainer
  - _self_
```

## 2.3 Main configs to edit

### `conf/dt/data/Tseng_train.yaml`

```yaml
data_root: /path/to/held_in_tokens
heldout_root: /path/to/held_out_tokens
registry_json: /path/to/base_registry.json
registry_json_heldout: /path/to/heldout_registry.json

token_key: token
vocab: 128
batch_size: 8
eval_batch_size: 8
num_workers: 8

train_split: train
val_split: val
test_split: test
```

### `conf/dt/nq/nq.yaml`

```yaml
vq_state: /path/to/vq_checkpoint.pth
```

### `conf/dt/trainer/trainer.yaml`

```yaml
mode: train                    # train / eval_test / finetune_heldout
eval_target: base              # base / heldout / both
eval_ckpt: null

init_ckpt: null
heldout_init_ckpt: null
heldout_epochs: 160
heldout_lr: 2e-4
heldout_train_mode: embed_only # embed_only / full
heldout_init_use_ema_weights: false

epochs: 160
compile: false
compile_dynamic: false
eval_detail_every: 5
```

## 2.4 Held-in training

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 task/dt_train_Tseng.py \
  train.mode=train \
  train.epochs=160 \
  train.compile=false \
  train.compile_dynamic=false \
  data.data_root=/path/to/held_in_tokens \
  data.registry_json=/path/to/base_registry.json \
  vq.vq_state=/path/to/vq_checkpoint.pth \
  runtime.device=cuda
```

## 2.5 Held-in evaluation

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 task/dt_train_Tseng.py \
  train.mode=eval_test \
  train.eval_target=base \
  train.eval_ckpt=/path/to/base_ckpt.pth \
  train.compile=false \
  train.compile_dynamic=false \
  data.data_root=/path/to/held_in_tokens \
  data.registry_json=/path/to/base_registry.json \
  vq.vq_state=/path/to/vq_checkpoint.pth \
  runtime.device=cuda
```

## 2.6 Held-out fine-tuning

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 task/dt_train_Tseng.py \
  train.mode=finetune_heldout \
  train.heldout_init_ckpt=/path/to/base_ckpt.pth \
  train.heldout_epochs=160 \
  train.heldout_lr=2e-4 \
  train.heldout_train_mode=embed_only \
  train.heldout_init_use_ema_weights=false \
  train.compile=false \
  train.compile_dynamic=false \
  data.heldout_root=/path/to/held_out_tokens \
  data.registry_json_heldout=/path/to/heldout_registry.json \
  vq.vq_state=/path/to/vq_checkpoint.pth \
  runtime.device=cuda
```

## 2.7 Held-out evaluation

### Held-out only

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 task/dt_train_Tseng.py \
  train.mode=eval_test \
  train.eval_target=heldout \
  train.eval_ckpt=/path/to/heldout_ft_ckpt.pth \
  train.compile=false \
  train.compile_dynamic=false \
  data.heldout_root=/path/to/held_out_tokens \
  data.registry_json_heldout=/path/to/heldout_registry.json \
  vq.vq_state=/path/to/vq_checkpoint.pth \
  runtime.device=cuda
```

### Base + held-out together

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 task/dt_train_Tseng.py \
  train.mode=eval_test \
  train.eval_target=both \
  train.eval_ckpt=/path/to/heldout_ft_ckpt.pth \
  train.compile=false \
  train.compile_dynamic=false \
  data.data_root=/path/to/held_in_tokens \
  data.heldout_root=/path/to/held_out_tokens \
  data.registry_json=/path/to/base_registry.json \
  data.registry_json_heldout=/path/to/heldout_registry.json \
  vq.vq_state=/path/to/vq_checkpoint.pth \
  runtime.device=cuda
```

---

# 3. Behavior Decoding (FT)

## Relevant files

```text
conf/ft/
  backbone/ar_backbone.yaml
  cache/feature_cache.yaml
  data/tseng_behavior.yaml
  eval/eval.yaml
  head/lowrank_uv.yaml
  train/base.yaml
  train/heldout.yaml
  ft_behavior_tseng.yaml

dataset/
  ft_behavior_dataset.py
  session_registry.py

model/behavior_decoder/
  distributed.py
  feature_cache.py
  ft_eval.py
  ft_trainer.py
  heads.py

task/
  FT_configs.py
  ft_behavior_decode.py
```

## 3.1 Main entry

```bash
task/ft_behavior_decode.py
```

## 3.2 Required defaults in `conf/ft/ft_behavior_tseng.yaml`

```yaml
defaults:
  - ft_behavior
  - data: tseng_behavior
  - backbone: ar_backbone
  - head: lowrank_uv
  - train@train_base: base
  - train@train_heldout: heldout
  - cache: feature_cache
  - eval: eval
  - _self_
```

## 3.3 Main configs to edit

### `conf/ft/data/tseng_behavior.yaml`

```yaml
registry_json: /path/to/global_registry.json
heldin_root: /path/to/held_in_root
heldout_root: /path/to/held_out_root

token_key: token
beh_key: behavior
vocab: 128
beh_channels: 3
beh_up: 2
batch_size: 8
num_workers: 8
```

### `conf/ft/backbone/ar_backbone.yaml`

```yaml
ar_ckpt: /path/to/backbone_ckpt.pth
prefer_ema_weights: true
compile: false
compile_dynamic: false

init_head_ckpt: null
init_head_registry_json: null
```

### `conf/ft/train/base.yaml`

```yaml
enabled: true
epochs: 200
lr: 0.0036
weight_decay: 1e-2
warmup_frac: 0.1
eval_every_epochs: 5
do_test_during_train: true
```

### `conf/ft/train/heldout.yaml`

```yaml
enabled: false
epochs: 100
lr: 0.0072
weight_decay: 1e-3
warmup_frac: 0.1
eval_every_epochs: 5
do_test_during_train: true
ft_mode: all    # all / newrows
```

### `conf/ft/eval/eval.yaml`

```yaml
eval_held_in_init: false
eval_held_out_init: false
max_eval_batches: -1
print_each_session: false
print_limit_sessions: -1
```

### `conf/ft/cache/feature_cache.yaml`

```yaml
enabled: true
build: true
use: true
force: false
dir: /path/to/cache_dir
dtype: fp16
max_batches: -1
allow_mismatch: false
allow_empty_eval: true
```

### `conf/ft/ft_behavior_tseng.yaml`

```yaml
remark: ft_behavior_tseng
mode: full_pipeline       # cache_only / train_base / finetune_heldout / eval_only / full_pipeline
run_name: heldout_ft

runtime:
  device: auto
  seed: 0
  tf32: true
  matmul_precision: high
  use_bf16: true
  ddp: true
  dist_backend: nccl
  dist_url: env://
  local_rank: 0

paths:
  output_root: /path/to/output_root
  save_dir: ${hydra:runtime.output_dir}
```

## 3.4 Held-in head training

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 task/ft_behavior_decode.py \
  mode=train_base \
  run_name=base_train \
  data.registry_json=/path/to/global_registry.json \
  data.heldin_root=/path/to/held_in_root \
  data.heldout_root=null \
  backbone.ar_ckpt=/path/to/backbone_ckpt.pth \
  backbone.compile=false \
  train_base.enabled=true \
  train_heldout.enabled=false \
  eval.eval_held_in_init=true \
  eval.eval_held_out_init=false \
  cache.enabled=true \
  cache.build=true \
  cache.use=true \
  cache.dir=/path/to/base_cache \
  paths.save_dir=/path/to/base_save_dir
```

## 3.5 Held-in head evaluation

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 task/ft_behavior_decode.py \
  mode=eval_only \
  run_name=base_eval \
  data.registry_json=/path/to/global_registry.json \
  data.heldin_root=/path/to/held_in_root \
  data.heldout_root=null \
  backbone.ar_ckpt=/path/to/backbone_ckpt.pth \
  backbone.init_head_ckpt=/path/to/best_base.pt \
  train_base.enabled=false \
  train_heldout.enabled=false \
  eval.eval_held_in_init=true \
  eval.eval_held_out_init=false \
  cache.enabled=false
```

## 3.6 Held-out fine-tuning

Edit these before running:

* `data.registry_json`
* `data.heldin_root`
* `data.heldout_root`
* `backbone.ar_ckpt`
* `backbone.init_head_ckpt=/path/to/best_base.pt`
* `backbone.init_head_registry_json=/path/to/base_registry.json` if remapping is needed
* `train_base.enabled=false`
* `train_heldout.enabled=true`

Run:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 task/ft_behavior_decode.py \
  mode=finetune_heldout \
  run_name=heldout_ft \
  data.registry_json=/path/to/global_registry.json \
  data.heldin_root=/path/to/held_in_root \
  data.heldout_root=/path/to/held_out_root \
  backbone.ar_ckpt=/path/to/backbone_ckpt.pth \
  backbone.compile=false \
  backbone.init_head_ckpt=/path/to/best_base.pt \
  backbone.init_head_registry_json=/path/to/base_registry.json \
  train_base.enabled=false \
  train_heldout.enabled=true \
  train_heldout.epochs=100 \
  train_heldout.lr=0.0072 \
  train_heldout.weight_decay=1e-3 \
  train_heldout.ft_mode=all \
  eval.eval_held_in_init=false \
  eval.eval_held_out_init=true \
  cache.enabled=true \
  cache.build=true \
  cache.use=true \
  cache.dir=/path/to/heldout_cache \
  paths.save_dir=/path/to/heldout_save_dir
```

### Train only new held-out rows

```bash
train_heldout.ft_mode=newrows
```

## 3.7 Held-out evaluation

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 task/ft_behavior_decode.py \
  mode=eval_only \
  run_name=heldout_eval \
  data.registry_json=/path/to/global_registry.json \
  data.heldin_root=/path/to/held_in_root \
  data.heldout_root=/path/to/held_out_root \
  backbone.ar_ckpt=/path/to/backbone_ckpt.pth \
  backbone.init_head_ckpt=/path/to/best_heldout.pt \
  train_base.enabled=false \
  train_heldout.enabled=false \
  eval.eval_held_in_init=false \
  eval.eval_held_out_init=true \
  cache.enabled=false
```

---

# 4. Full Pipeline Summary

## VQ

1. Train VQ
2. Test / reconstruct VQ
3. Export tokens for DT

## DT

1. Train on held-in sessions
2. Evaluate on held-in sessions
3. Fine-tune on held-out sessions if needed
4. Evaluate held-out or both

## FT

1. Train base behavior head on held-in sessions
2. Save `best_base.pt`
3. Fine-tune held-out behavior head from base head
4. Save `best_heldout.pt` or `final_head.pt`
5. Evaluate held-in / held-out as needed
