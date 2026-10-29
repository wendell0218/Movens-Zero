cd "$(dirname "${BASH_SOURCE[0]}")/.."
conda activate movens-zero
python evaluation/download_assets.py --output-dir checkpoints/evaluation
