# The “Green” Refocusing Engine

A research-grade hybrid computational imaging pipeline for sustainable refocusing.

## Important clarification
- **No training is required** for this repository.
- We use a **pretrained Depth Anything model** for inference only.
- NYU depth maps are used for pairing/evaluation and analysis.

## Dataset paths configured for your machine
- Depth: `/Users/sarthakjain/Desktop/ML Projects/GreenComputing/nyu_data/depth_images`
- RGB: `/Users/sarthakjain/Desktop/ML Projects/GreenComputing/nyu_data/rbg_images`

## Dependency note for your error
If you see `ModuleNotFoundError: No module named 'depth_anything'`, this repo now supports two backends:
1. Official Depth Anything package (`depth_anything.dpt`), OR
2. Hugging Face Transformers depth-estimation fallback.

Install either backend (both also work):
```bash
pip install -r requirements.txt
pip install git+https://github.com/LiheYoung/Depth-Anything.git
```


## Using your locally cloned Depth-Anything repo
This code automatically adds this path to `sys.path` before importing:
`/Users/sarthakjain/Desktop/ML Projects/GreenComputing/Depth-Anything`

If your clone is elsewhere, set:
```bash
export DEPTH_ANYTHING_LOCAL_REPO="/path/to/Depth-Anything"
```

## Installation
```bash
cd green_refocusing_engine
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run at once (Python only)
```bash
python run_pipeline.py
```

## Alternative run (custom args)
```bash
python -m src.main \
  --rgb_dir "/Users/sarthakjain/Desktop/ML Projects/GreenComputing/nyu_data/rbg_images" \
  --depth_dir "/Users/sarthakjain/Desktop/ML Projects/GreenComputing/nyu_data/depth_images" \
  --sigma_max 8.0 --num_sigma_levels 12 --alpha 12.0
```

## Method summary
1. Predict relative depth once from RGB.
2. Build sigma map: `sigma(x,y)=alpha*|D(x,y)-F|`.
3. Precompute FFT Gaussian pyramid.
4. Blend spatially for refocused outputs.
5. Compare against repeated-inference baseline.

## Outputs
- `outputs/depth_maps/`
- `outputs/refocused_images/`
- `outputs/comparisons/`
- `outputs/metrics/quality_metrics.csv`
- `outputs/plots/*.png`
