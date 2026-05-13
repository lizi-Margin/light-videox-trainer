# Light VideoX Trainer

A lightweight training harness for Wan/VideoX-Fun style video diffusion fine-tuning.

The supported tasks are text-to-video and random-mask inpainting training with
`Wan2.1-Fun-1.3B-InP`.

## Layout

- `main.py`: unified CLI for training and sanity checks.
- `configs/t2v_smoke.jsonc`: T2V smoke-test config.
- `configs/inpaint_smoke.jsonc`: inpainting smoke-test config.
- `trainer/`: class-based Wan trainers, task condition builders, scheduler helpers, checkpointing.
- `data/`: video dataset, collate logic, and `VideoDataModule`.
- `models/`: Wan model loading bundle and forward utilities.
- `utils/`: config, path, device, and video IO helpers.
- `tools/`: metadata preparation and validation helpers.
- `third_party/videox_fun/`: copied from `gen-omnimatte-public`.

## Install

Use the existing environment:

```bash
micromamba activate video_gen
pip install -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

## Train

```bash
micromamba activate video_gen
python main.py train --config configs/inpaint_smoke.jsonc
```

Training metrics are logged with the vendored UHTK visualizer by default. The
default log directory is `<output_dir>/visualizer`, and the live/static outputs are
written under that directory, including `rec.jpg`, `rec.json`, and `rec.csv`.
Logged scalar metrics include `train_loss`, `learning_rate`, `grad_norm`,
`step_time_sec`, `samples_per_sec`, `samples_seen`, `timestep_mean`,
`sigma_mean`, `mask_fraction`, validation metrics such as
`validation_masked_psnr`, and CUDA memory metrics when training on GPU.

Relevant config keys:

```json
"visualizer_enabled": true,
"visualizer_smooth": true,
"visualizer_dpi": 120,
"visualizer_font_size": 9,
"visualizer_figsize": null
```

For inpainting runs, low-cost training samples and fixed validation are separate.
Training samples are for quick visual health checks. Validation uses fixed cases,
fixed masks, and fixed inference seeds so checkpoint outputs can be compared.
Validation outputs are written under `<output_dir>/validation`, including
`source.mp4`, `mask.mp4`, `masked_input.mp4`, `generated.mp4`, and `summary.json`
for each validation step.

Relevant validation and inpainting keys:

```json
"validation_every_steps": 500,
"validation_metadata": "",
"validation_count": 8,
"validation_seed": 2026,
"masked_loss_weight": 4.0,
"unmasked_loss_weight": 1.0,
"mask_modes": ["rectangle", "moving_rectangle", "brush"]
```

Run a lightweight data/config sanity check before training:

```bash
python main.py sanity --config configs/inpaint_smoke.jsonc
```

Before training, create and validate metadata:

```bash
python tools/prepare_openvid_metadata.py \
  --video-root /path/to/extracted/openvid/videos \
  --caption-table /path/to/openvid/captions.json \
  --output /path/to/openvid_metadata.json

python tools/validate_metadata.py \
  --metadata /path/to/openvid_metadata.json \
  --write-valid /path/to/openvid_metadata.valid.json

python tools/inspect_batch.py \
  --config configs/inpaint_smoke.jsonc \
  --output outputs/inspect_batch.mp4
```

Metadata format:

```json
[
  {
    "file_path": "/abs/path/to/video.mp4",
    "text": "caption text",
    "type": "video"
  }
]
```
