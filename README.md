# Light VideoX Trainer

A lightweight training harness for Wan/VideoX-Fun style video diffusion fine-tuning.

The supported tasks are text-to-video smoke/full training and random-mask inpainting training
with `Wan2.1-Fun-1.3B-InP`.

## Layout

- `train_wan_t2v.py`: root training entrypoint.
- `sanity_check.py`: validates config, metadata, batch decode, and optional model loading.
- `configs/*.jsonc`: commented JSON configuration files.
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
python train_wan_t2v.py --config configs/wan_t2v_1.3b_debug.jsonc
```

Use `"task": "inpaint"` in the config for the inpainting trainer. Without a task field, the
factory uses the T2V trainer.

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
  --config configs/wan_t2v_1.3b_debug.jsonc \
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
