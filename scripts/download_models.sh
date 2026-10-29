cd "$(dirname "${BASH_SOURCE[0]}")/.."
conda activate movens-zero
python scripts/download_models.py "$@"
