cd "$(dirname "${BASH_SOURCE[0]}")/.."
conda activate movens-zero
python evaluation/setup.py "$@"
