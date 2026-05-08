# Light VideoX Trainer

A lightweight training harness for Wan/VideoX-Fun style video diffusion fine-tuning.

The first supported path is full-parameter text-to-video training with `Wan2.1-Fun-1.3B-InP`.
Inpainting support is intentionally left out of the initial training path, but the vendored
`third_party/videox_fun` code is kept so the inpaint branch can be added later without changing
the model loading foundation.

## Layout

- `train_wan_t2v.py`: root training entrypoint.
- `sanity_check.py`: validates config, metadata, batch decode, and optional model loading.
- `configs/*.jsonc`: commented JSON configuration files.
- `trainer/`: training loop, scheduler helpers, checkpointing.
- `data/`: video dataset and collate logic.
- `models/`: Wan model loading and forward utilities.
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
