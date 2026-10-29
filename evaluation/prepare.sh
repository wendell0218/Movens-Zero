cd "$(dirname "${BASH_SOURCE[0]}")/.."
conda activate movens-zero
python evaluation/prepare.py --track text --output-dir data/intent/text
