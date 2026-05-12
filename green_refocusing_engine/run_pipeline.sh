#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -m src.main --dataset_path data/nyu_depth_v2 --sigma_max 8.0 --num_sigma_levels 12 --save_outputs
