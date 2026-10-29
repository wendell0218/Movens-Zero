cd "$(dirname "${BASH_SOURCE[0]}")/.."
conda activate movens-zero
python evaluation/score.py --track text --data-dir data/intent/text --generation-dir outputs/intent/text --output-dir outputs/intent/text/metrics
