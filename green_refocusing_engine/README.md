# The “Green” Refocusing Engine

A research-grade hybrid computational imaging pipeline for sustainable refocusing.

## 1) Super-quick install (macOS/Linux)
```bash
cd green_refocusing_engine
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Run immediately (one command)
```bash
python run_pipeline.py
```

`run_pipeline.py` is self-contained (no cross-module runtime imports required).

## 3) If your data is in a custom folder
```bash
python run_pipeline.py \
  --rgb_dir "/Users/sarthakjain/Desktop/ML Projects/GreenComputing/nyu_data/rbg_images" \
  --output_root outputs
```

## 4) What the script generates
- `outputs/refocused_images/*_green.png`
- `outputs/refocused_images/*_sweep.gif`
- `outputs/depth_maps/*_depth.png`
- `outputs/depth_maps/*_sigma.png`

## 5) Common issues
### `ModuleNotFoundError`
Use the self-contained command:
```bash
python run_pipeline.py
```
Do **not** run `src/main.py` directly unless you are intentionally using package mode.

### Slow first run
The first run downloads the Hugging Face model (`LiheYoung/depth-anything-base-hf`), which can take a few minutes.

### No images found
Make sure `--rgb_dir` points to a folder containing `.png/.jpg/.jpeg` files.

## 6) Optional advanced run
If you want package-style execution:
```bash
python -m src.main \
  --rgb_dir "/Users/sarthakjain/Desktop/ML Projects/GreenComputing/nyu_data/rbg_images" \
  --depth_dir "/Users/sarthakjain/Desktop/ML Projects/GreenComputing/nyu_data/depth_images" \
  --sigma_max 8.0 --num_sigma_levels 12 --alpha 12.0
```
