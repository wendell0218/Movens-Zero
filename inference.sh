cd "$(dirname "${BASH_SOURCE[0]}")"
conda activate movens-zero
python -m movens_zero.inference "$@"
