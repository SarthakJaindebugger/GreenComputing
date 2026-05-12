# The “Green” Refocusing Engine

A research-grade hybrid computational imaging pipeline for sustainable refocusing.

## Overview
This repository compares:
1. **Baseline:** repeated neural depth inference per focus change.
2. **Green method:** single depth inference + FFT spatially varying blur reuse.

## Installation
```bash
cd green_refocusing_engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset setup (NYU Depth V2)
Place files as:
```
data/nyu_depth_v2/
  test/
    rgb/*.png
    depth/*.png
```
Download source: NYU Depth V2 official pages/Kaggle mirrors, then convert to PNG pairs if needed.

## Run
```bash
./run_pipeline.sh
# or
python -m src.main --image_path data/sample_images/example.jpg --sigma_max 10 --num_sigma_levels 16
```

## Mathematical core
- Relative depth map: \(D(x,y)\in[0,1]\).
- Blur radius map:
  \[\sigma(x,y)=\alpha |D(x,y)-F|\]
- FFT blur:
  \[I_\sigma = \mathcal{F}^{-1}(\mathcal{F}(I)\odot\mathcal{F}(G_\sigma))\]
- Spatially varying synthesis via interpolation between precomputed sigma pyramid levels.

## Sustainability motivation
"Compute Once, Filter Many" removes repeated transformer inference and substantially lowers runtime, memory pressure, and estimated energy per focus sweep.

## Outputs
- `outputs/depth_maps/`
- `outputs/refocused_images/`
- `outputs/comparisons/`
- `outputs/metrics/quality_metrics.csv`
- `outputs/plots/*.png`

## Methodology
1. Load RGB(+depth GT when available).
2. Run Depth Anything once (green) or per-focus (baseline).
3. Generate sigma map(s), synthesize refocus.
4. Evaluate MSE/PSNR/SSIM + efficiency metrics.
5. Plot quality/efficiency trade-offs.

## Future work
- metric-depth calibration,
- learned defocus kernels,
- temporal consistency for video,
- full hardware power logging via CodeCarbon/NVML integration.
