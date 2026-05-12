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
